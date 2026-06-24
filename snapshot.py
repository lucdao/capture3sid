#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

import requests
import yaml


# Example:
# python3 vehicle_capture_consumer.py \
#   --broker 127.0.0.1:9092 \
#   --topic ai.metadata.v1 \
#   --group-id vehicle-capture-service \
#   --confidence-threshold 0.5 \
#   --cooldown-sec 60 \
#   --output-root server_assets/vehicle_captures \
#   --snapshot-base-url http://192.168.100.154:9999/snapshot.jpg \
#   --snapshot-token pb_9f3k2lsdgf32hnjrk23j2k55j322kjnw3


DEFAULT_BROKER = "127.0.0.1:9092"
DEFAULT_TOPIC = "ai.metadata.v1"
DEFAULT_GROUP_ID = "vehicle-capture-service"
DEFAULT_CONFIDENCE_THRESHOLD = 0.96
DEFAULT_COOLDOWN_SEC = 60.0
DEFAULT_OUTPUT_ROOT = "server_assets/vehicle_captures"
DEFAULT_REQUEST_TIMEOUT_SEC = 3.0
DEFAULT_RETRIES = 1
DEFAULT_POLL_TIMEOUT_SEC = 1.0
DEFAULT_MAX_EVENT_AGE_SEC = 120.0
DEFAULT_VEHICLE_TYPES = ("car", "truck", "bus", "vehicle")

DEFAULT_SNAPSHOT_BASE_URL = "http://192.168.100.154:9999/snapshot_rtsp"
DEFAULT_SNAPSHOT_TOKEN = ""


@dataclass(frozen=True)
class CaptureEvent:
    plate: str
    trigger_cam_id: str
    capture_cam_id: str
    capture_role: str
    group_id: str
    message_id: str
    frame_num: Optional[int]
    ntp_timestamp: Optional[int]
    produced_at: Optional[str]
    snapshot_at: str
    tracking_object_id: Optional[int]
    confidence: float
    ai_result: Dict[str, Any]


@dataclass(frozen=True)
class CameraTarget:
    role: str
    cam_id: str


@dataclass(frozen=True)
class CameraGroup:
    group_id: str
    ai_camera_role: Optional[str]
    ai_cam_id: Optional[str]
    cameras: Tuple[CameraTarget, ...]


@dataclass(frozen=True)
class ServiceConfig:
    camera_groups: Tuple[CameraGroup, ...]


def decode_message_value(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            logging.warning("Failed to decode JSON message: %s", exc)
            return None
        return decoded if isinstance(decoded, dict) else None
    return value if isinstance(value, dict) else None


def normalize_plate(value: Any) -> str:
    if value is None:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9]", "", str(value).strip()).upper()
    return normalized


def safe_folder_name(plate: str) -> str:
    normalized = normalize_plate(plate)
    return normalized or "UNKNOWN"


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def event_age_sec(event: "CaptureEvent", *, now: Optional[datetime] = None) -> Optional[float]:
    produced_at = parse_iso_datetime(event.produced_at)
    if produced_at is None:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - produced_at).total_seconds()


def load_service_config(path: Optional[str]) -> ServiceConfig:
    if not path:
        return ServiceConfig(camera_groups=())

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw_groups = raw.get("camera_groups") or {}
    if not isinstance(raw_groups, dict):
        raise ValueError("config camera_groups must be a mapping")

    groups: List[CameraGroup] = []
    seen_group_ids: set[str] = set()
    seen_cam_ids: Dict[str, str] = {}
    for group_id, group_cfg in raw_groups.items():
        group_id_text = str(group_id).strip()
        if not group_id_text:
            raise ValueError("camera group id must not be empty")
        if group_id_text in seen_group_ids:
            raise ValueError(f"duplicate camera group id {group_id_text!r}")
        seen_group_ids.add(group_id_text)

        if not isinstance(group_cfg, dict):
            raise ValueError(f"camera group {group_id!r} must be a mapping")

        cameras_cfg = group_cfg.get("cameras") or {}
        if not isinstance(cameras_cfg, dict):
            raise ValueError(f"camera group {group_id!r} cameras must be a mapping")

        targets: List[CameraTarget] = []
        seen_roles: set[str] = set()
        for role, cam_cfg in cameras_cfg.items():
            role_text = str(role).strip()
            if not role_text:
                raise ValueError(f"camera group {group_id!r} contains an empty camera role")
            if role_text in seen_roles:
                raise ValueError(f"camera group {group_id!r} contains duplicate role {role_text!r}")
            seen_roles.add(role_text)

            cam_id = ""
            if isinstance(cam_cfg, dict):
                cam_id = str(cam_cfg.get("cam_id") or "").strip()
            elif cam_cfg is not None:
                cam_id = str(cam_cfg).strip()
            if cam_id:
                owner = seen_cam_ids.get(cam_id)
                if owner and owner != group_id_text:
                    raise ValueError(
                        f"camera {cam_id!r} is configured in multiple groups: {owner!r} and {group_id_text!r}"
                    )
                seen_cam_ids[cam_id] = group_id_text
                targets.append(CameraTarget(role=role_text, cam_id=cam_id))

        ai_role = str(group_cfg.get("ai_camera") or "").strip() or None
        ai_cam_id = ""
        if ai_role:
            for target in targets:
                if target.role == ai_role:
                    ai_cam_id = target.cam_id
                    break
        if ai_role and not ai_cam_id:
            raise ValueError(
                f"camera group {group_id!r} ai_camera={ai_role!r} has no cam_id"
            )
        if not targets:
            raise ValueError(f"camera group {group_id!r} must contain at least one camera")

        groups.append(
            CameraGroup(
                group_id=group_id_text,
                ai_camera_role=ai_role,
                ai_cam_id=ai_cam_id or None,
                cameras=tuple(targets),
            )
        )

    return ServiceConfig(camera_groups=tuple(groups))


def plate_from_ai_result(result: Dict[str, Any]) -> str:
    # Prefer the license-plate-like fields. Fall back to empty string.
    for key in ("detected_object_ids", "name", "object_name"):
        plate = normalize_plate(result.get(key))
        if plate:
            return plate
    return ""


def iter_capture_events(
    payload: Dict[str, Any],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    allow_missing_plate: bool = False,
    vehicle_types: Tuple[str, ...] = DEFAULT_VEHICLE_TYPES,
) -> Iterable[CaptureEvent]:
    ai_results = payload.get("ai_results")
    if not isinstance(ai_results, list):
        return

    trigger_cam_id = str(payload.get("cam_id") or "")
    message_id = str(payload.get("message_id") or "")
    frame_num = coerce_int(payload.get("frame_num"))
    ntp_timestamp = coerce_int(payload.get("ntp_timestamp"))
    produced_at = payload.get("produced_at")

    for result in ai_results:
        if not isinstance(result, dict):
            continue

        if str(result.get("meta_type") or "").strip().lower() not in vehicle_types:
            continue

        confidence = coerce_float(result.get("confidence"), 0.0)
        if confidence < float(confidence_threshold):
            continue

        plate = plate_from_ai_result(result)
        tracking_object_id = coerce_int(result.get("tracking_object_id"))

        if not plate and allow_missing_plate:
            fallback_id = tracking_object_id if tracking_object_id is not None else result.get("id", "unknown")
            plate = safe_folder_name(f"TRACK{fallback_id}")

        if not plate:
            continue

        yield CaptureEvent(
            plate=plate,
            trigger_cam_id=trigger_cam_id,
            capture_cam_id=trigger_cam_id,
            capture_role="trigger",
            group_id="",
            message_id=message_id,
            frame_num=frame_num,
            ntp_timestamp=ntp_timestamp,
            produced_at=str(produced_at) if produced_at is not None else None,
            snapshot_at=str(produced_at) if produced_at is not None else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            tracking_object_id=tracking_object_id,
            confidence=confidence,
            ai_result=dict(result),
        )


class CooldownDeduper:
    def __init__(self, cooldown_sec: float = DEFAULT_COOLDOWN_SEC, clock: Callable[[], float] = time.monotonic):
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self.clock = clock
        self._last_seen: Dict[Tuple[str, str], float] = {}

    def should_process(self, cam_id: str, plate: str) -> bool:
        key = (str(cam_id), normalize_plate(plate))
        now = float(self.clock())
        last_seen = self._last_seen.get(key)
        if last_seen is not None and now - last_seen < self.cooldown_sec:
            return False
        self._last_seen[key] = now
        return True

    def in_cooldown(self, cam_id: str, plate: str) -> bool:
        key = (str(cam_id), normalize_plate(plate))
        now = float(self.clock())
        last_seen = self._last_seen.get(key)
        return last_seen is not None and now - last_seen < self.cooldown_sec

    def mark_processed(self, cam_id: str, plate: str) -> None:
        key = (str(cam_id), normalize_plate(plate))
        self._last_seen[key] = float(self.clock())


def sanitize_timestamp(value: str) -> str:
    # Safe for filenames: 2026-06-24T04:44:45.179449Z -> 2026-06-24T04-44-45_179449Z
    return (
        str(value)
        .strip()
        .replace(":", "-")
        .replace("/", "-")
        .replace(" ", "_")
        .replace(".", "_")
    )


def build_snapshot_url(
    *,
    base_url: str,
    device_id: str,
    at_time: Optional[str],
    token: str,
) -> str:
    params = {
        "device_id": device_id,
        "token": token,
    }
    if at_time:
        params["at_time"] = at_time
    return f"{base_url}?{urlencode(params)}"


def redact_secret(text: Any, secret: str) -> str:
    value = str(text)
    if secret:
        value = value.replace(secret, "<redacted>")
    return value


def snapshot_filename(event: CaptureEvent, *, role: str = "snapshot") -> str:
    ts = sanitize_timestamp(event.snapshot_at or event.produced_at or time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime()))
    plate = safe_folder_name(event.plate)
    cam = normalize_plate(event.capture_cam_id) or "CAM"
    capture_role = safe_folder_name(event.capture_role or role)
    return f"{capture_role}_{plate}_{cam}_{ts}.jpg"


class SnapshotClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_SNAPSHOT_BASE_URL,
        token: str = DEFAULT_SNAPSHOT_TOKEN,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
        retries: int = DEFAULT_RETRIES,
        dry_run: bool = False,
        include_at_time: bool = False,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = str(base_url).rstrip("?")
        self.token = str(token)
        self.output_root = Path(output_root)
        self.timeout_sec = float(timeout_sec)
        self.retries = max(0, int(retries))
        self.dry_run = bool(dry_run)
        self.include_at_time = bool(include_at_time)
        self.session = session or requests.Session()

    def plate_dir(self, plate: str) -> Path:
        return self.output_root / safe_folder_name(plate)

    def real_images_in_plate_dir(self, plate: str) -> List[Path]:
        output_dir = self.plate_dir(plate)
        if not output_dir.is_dir():
            return []
        return [
            path
            for path in output_dir.glob("*.jpg")
            if self.is_valid_image_file(path)
        ]

    def has_existing_capture(self, plate: str, *, min_images: int = 1) -> bool:
        return len(self.real_images_in_plate_dir(plate)) >= max(1, int(min_images))

    def has_existing_capture_for_events(self, events: List[CaptureEvent]) -> bool:
        if not events:
            return False

        for event in events:
            if not self.has_existing_capture_for_event(event):
                return False
        return True

    def has_existing_capture_for_event(self, event: CaptureEvent) -> bool:
        output_dir = self.plate_dir(event.plate)
        if not output_dir.is_dir():
            return False
        role = safe_folder_name(event.capture_role)
        cam = normalize_plate(event.capture_cam_id)
        matches = list(output_dir.glob(f"{role}_*_{cam}_*.jpg"))
        return any(self.is_valid_image_file(path) for path in matches)

    def is_valid_image_file(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            if path.stat().st_size <= 0:
                return False
            with open(path, "rb") as f:
                header = f.read(16)
        except OSError:
            return False
        return (
            header.startswith(b"\xff\xd8\xff")
            or header.startswith(b"\x89PNG\r\n\x1a\n")
            or header.startswith(b"GIF87a")
            or header.startswith(b"GIF89a")
            or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
        )

    def validate_image_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        if size <= 0:
            path.unlink(missing_ok=True)
            raise RuntimeError("snapshot response produced empty image file")
        if content_type and "image" not in content_type:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"snapshot response is not an image: {content_type}")
        with open(path, "rb") as f:
            header = f.read(16)
        looks_like_image = (
            header.startswith(b"\xff\xd8\xff")  # JPEG
            or header.startswith(b"\x89PNG\r\n\x1a\n")
            or header.startswith(b"GIF87a")
            or header.startswith(b"GIF89a")
            or header.startswith(b"RIFF") and header[8:12] == b"WEBP"
        )
        if not looks_like_image:
            path.unlink(missing_ok=True)
            raise RuntimeError("snapshot response does not look like an image")

    def capture(self, event: CaptureEvent) -> Dict[str, Any]:
        output_dir = self.plate_dir(event.plate)

        if not event.capture_cam_id:
            raise RuntimeError("Missing cam_id / device_id in message")
        if not event.snapshot_at:
            raise RuntimeError("Missing snapshot timestamp")

        snapshot_url = build_snapshot_url(
            base_url=self.base_url,
            device_id=event.capture_cam_id,
            at_time=event.snapshot_at if self.include_at_time else None,
            token=self.token,
        )
        output_path = output_dir / snapshot_filename(event)

        if self.dry_run:
            logging.info("Dry-run snapshot url=%s output=%s", redact_secret(snapshot_url, self.token), output_path)
            return {
                "url": snapshot_url,
                "file": str(output_path),
                "status": "dry_run",
            }

        last_error: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            tmp_path: Optional[Path] = None
            try:
                with self.session.get(snapshot_url, timeout=self.timeout_sec, stream=True) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "").lower()

                    output_dir.mkdir(parents=True, exist_ok=True)
                    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
                    tmp_path.unlink(missing_ok=True)
                    with open(tmp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    self.validate_image_file(tmp_path, content_type)
                    os.replace(tmp_path, output_path)

                logging.info(
                    "Snapshot saved group=%s role=%s plate=%s trigger_cam=%s capture_cam=%s at_time=%s file=%s status=%s",
                    event.
                    event.group_id,
                    event.capture_role,
                    event.plate,
                    event.trigger_cam_id,
                    event.capture_cam_id,
                    event.snapshot_at,
                    output_path,
                    response.status_code,
                )
                return {
                    "url": snapshot_url,
                    "file": str(output_path),
                    "status": response.status_code,
                }

            except Exception as exc:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
                last_error = exc
                logging.warning(
                    "Snapshot request failed group=%s role=%s plate=%s trigger_cam=%s capture_cam=%s attempt=%s/%s: %s",
                    event.group_id,
                    event.capture_role,
                    event.plate,
                    event.trigger_cam_id,
                    event.capture_cam_id,
                    attempt + 1,
                    self.retries + 1,
                    redact_secret(exc, self.token),
                )

        raise RuntimeError(
            f"snapshot failed after {self.retries + 1} attempts: {redact_secret(last_error, self.token)}"
        )


class VehicleCaptureProcessor:
    def __init__(
        self,
        snapshot_client: SnapshotClient,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        allow_missing_plate: bool = False,
        max_event_age_sec: float = DEFAULT_MAX_EVENT_AGE_SEC,
        service_config: Optional[ServiceConfig] = None,
        deduper: Optional[CooldownDeduper] = None,
    ):
        self.snapshot_client = snapshot_client
        self.confidence_threshold = float(confidence_threshold)
        self.allow_missing_plate = bool(allow_missing_plate)
        self.max_event_age_sec = max(0.0, float(max_event_age_sec))
        self.service_config = service_config or ServiceConfig(camera_groups=())
        self.groups_by_trigger_cam: Dict[str, List[CameraGroup]] = {}
        for group in self.service_config.camera_groups:
            if group.ai_cam_id:
                self.groups_by_trigger_cam.setdefault(group.ai_cam_id, []).append(group)
            else:
                for target in group.cameras:
                    self.groups_by_trigger_cam.setdefault(target.cam_id, []).append(group)
        self.deduper = deduper or CooldownDeduper(cooldown_sec)

    def expand_group_events(self, event: CaptureEvent) -> List[CaptureEvent]:
        groups = self.groups_by_trigger_cam.get(event.trigger_cam_id)
        if not groups:
            if self.service_config.camera_groups:
                logging.debug("Skipping cam=%s because it is not configured in any camera_group", event.trigger_cam_id)
                return []
            return [event]

        expanded: List[CaptureEvent] = []
        for group in groups:
            for target in group.cameras:
                expanded.append(
                    CaptureEvent(
                        plate=event.plate,
                        trigger_cam_id=event.trigger_cam_id,
                        capture_cam_id=target.cam_id,
                        capture_role=target.role,
                        group_id=group.group_id,
                        message_id=event.message_id,
                        frame_num=event.frame_num,
                        ntp_timestamp=event.ntp_timestamp,
                        produced_at=event.produced_at,
                        snapshot_at=event.snapshot_at,
                        tracking_object_id=event.tracking_object_id,
                        confidence=event.confidence,
                        ai_result=event.ai_result,
                    )
                )
        return expanded

    def process_payload(self, payload: Dict[str, Any]) -> List[CaptureEvent]:
        processed: List[CaptureEvent] = []
        for event in iter_capture_events(
            payload,
            confidence_threshold=self.confidence_threshold,
            allow_missing_plate=self.allow_missing_plate,
        ):
            groups = self.groups_by_trigger_cam.get(event.trigger_cam_id)
            if self.service_config.camera_groups and not groups:
                logging.debug("Skipping cam=%s because it is not configured in any camera_group", event.trigger_cam_id)
                continue

            age = event_age_sec(event)
            if age is not None and age > self.max_event_age_sec:
                logging.info(
                    "Skipping stale event plate=%s cam_id=%s age=%.1fs max_age=%.1fs produced_at=%s",
                    event.plate,
                    event.trigger_cam_id,
                    age,
                    self.max_event_age_sec,
                    event.produced_at,
                )
                continue

            dedupe_cam = event.trigger_cam_id
            if groups:
                dedupe_cam = ",".join(group.group_id for group in groups)

            capture_events = self.expand_group_events(event)
            if not capture_events:
                continue

            required_images = len(capture_events)
            missing_events = [
                capture_event
                for capture_event in capture_events
                if not self.snapshot_client.has_existing_capture_for_event(capture_event)
            ]
            if not missing_events:
                if not self.deduper.in_cooldown(dedupe_cam, event.plate):
                    logging.info(
                        "Skipping existing capture plate=%s required_images=%d existing_images=%d",
                        event.plate,
                        required_images,
                        len(self.snapshot_client.real_images_in_plate_dir(event.plate)),
                    )
                    self.deduper.mark_processed(dedupe_cam, event.plate)
                continue

            if self.deduper.in_cooldown(dedupe_cam, event.plate) and len(missing_events) == required_images:
                logging.debug("Skipping cooldown key=%s plate=%s", dedupe_cam, event.plate)
                continue

            success_count = 0
            for capture_event in missing_events:
                try:
                    self.snapshot_client.capture(capture_event)
                    processed.append(capture_event)
                    success_count += 1
                except Exception as exc:
                    logging.error(
                        "Snapshot failed, message will be skipped without stopping consumer group=%s role=%s plate=%s trigger_cam=%s capture_cam=%s at_time=%s error=%s",
                        capture_event.group_id,
                        capture_event.capture_role,
                        capture_event.plate,
                        capture_event.trigger_cam_id,
                        capture_event.capture_cam_id,
                        capture_event.snapshot_at,
                        exc,
                    )
            if success_count == len(missing_events):
                self.deduper.mark_processed(dedupe_cam, event.plate)

        return processed


def process_kafka_message(
    message: Any,
    processor: VehicleCaptureProcessor,
    consumer: Any,
    *,
    commit_offsets: bool = True,
) -> List[CaptureEvent]:
    payload = decode_message_value(message.value())
    if payload is None:
        if commit_offsets:
            try:
                consumer.commit(message=message, asynchronous=False)
            except Exception as exc:
                logging.warning("Kafka commit failed for invalid message: %s", exc)
        return []

    events = processor.process_payload(payload)
    if commit_offsets:
        try:
            consumer.commit(message=message, asynchronous=False)
        except Exception as exc:
            logging.warning("Kafka commit failed: %s", exc)
    return events


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consume V3S Kafka vehicle events and download snapshots from snapshot.jpg API."
    )
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--group-id", default=DEFAULT_GROUP_ID)
    parser.add_argument("--offset", choices=("earliest", "latest"), default="latest")
    parser.add_argument("--poll-timeout", type=float, default=DEFAULT_POLL_TIMEOUT_SEC)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--cooldown-sec", type=float, default=DEFAULT_COOLDOWN_SEC)
    parser.add_argument(
        "--max-event-age-sec",
        type=float,
        default=DEFAULT_MAX_EVENT_AGE_SEC,
        help="Skip Kafka events older than this many seconds to avoid requesting expired snapshots.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--request-timeout-sec", type=float, default=DEFAULT_REQUEST_TIMEOUT_SEC)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--allow-missing-plate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--config", help="YAML config containing camera_groups.")

    parser.add_argument("--snapshot-base-url", default=DEFAULT_SNAPSHOT_BASE_URL)
    parser.add_argument("--snapshot-token", default=DEFAULT_SNAPSHOT_TOKEN)
    parser.add_argument(
        "--snapshot-include-at-time",
        action="store_true",
        help="Send at_time to snapshot API. Leave off for /snapshot_rtsp realtime API.",
    )

    return parser


def run_consumer(args: argparse.Namespace) -> int:
    try:
        from confluent_kafka import Consumer, KafkaError, TopicPartition
    except ImportError:
        print("confluent-kafka is not installed. Install requirements first.", file=sys.stderr)
        return 1

    snapshot_client = SnapshotClient(
        base_url=args.snapshot_base_url,
        token=args.snapshot_token,
        output_root=args.output_root,
        timeout_sec=args.request_timeout_sec,
        retries=args.retries,
        dry_run=args.dry_run,
        include_at_time=args.snapshot_include_at_time,
    )
    if not args.dry_run and not str(args.snapshot_token or "").strip():
        raise ValueError("snapshot token is required unless --dry-run is used")
    service_config = load_service_config(args.config)
    processor = VehicleCaptureProcessor(
        snapshot_client,
        confidence_threshold=args.confidence_threshold,
        cooldown_sec=args.cooldown_sec,
        max_event_age_sec=args.max_event_age_sec,
        allow_missing_plate=args.allow_missing_plate,
        service_config=service_config,
    )

    consumer = Consumer(
        {
            "bootstrap.servers": args.broker,
            "group.id": args.group_id,
            "auto.offset.reset": args.offset,
            "enable.auto.commit": False,
        }
    )
    commit_offsets = True
    if args.offset == "latest":
        commit_offsets = False
        topic_meta = None
        while topic_meta is None:
            try:
                metadata = consumer.list_topics(args.topic, timeout=10.0)
                topic_meta = metadata.topics.get(args.topic)
                if topic_meta is None or topic_meta.error is not None:
                    error = topic_meta.error if topic_meta else "missing topic"
                    logging.warning("Kafka topic metadata unavailable for %r: %s", args.topic, error)
                    topic_meta = None
                    time.sleep(5.0)
            except Exception as exc:
                logging.warning("Kafka metadata lookup failed, retrying: %s", exc)
                time.sleep(5.0)

        assignments = []
        for partition_id in sorted(topic_meta.partitions):
            probe = TopicPartition(args.topic, partition_id)
            try:
                _low, high = consumer.get_watermark_offsets(probe, timeout=10.0)
            except Exception as exc:
                logging.warning(
                    "Kafka watermark lookup failed topic=%s partition=%s, starting at stored/latest fallback: %s",
                    args.topic,
                    partition_id,
                    exc,
                )
                high = -1
            assignments.append(TopicPartition(args.topic, partition_id, high))
            logging.info(
                "Starting at Kafka latest topic=%s partition=%s offset=%s",
                args.topic,
                partition_id,
                high,
            )
        consumer.assign(assignments)
    else:
        consumer.subscribe([args.topic])

    logging.info(
        "Consuming topic=%s broker=%s group_id=%s threshold=%.3f cooldown=%.1fs max_event_age=%.1fs camera_groups=%d dry_run=%s snapshot_base_url=%s",
        args.topic,
        args.broker,
        args.group_id,
        args.confidence_threshold,
        args.cooldown_sec,
        args.max_event_age_sec,
        len(service_config.camera_groups),
        args.dry_run,
        args.snapshot_base_url,
    )

    try:
        while True:
            message = consumer.poll(float(args.poll_timeout))
            if message is None:
                continue

            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logging.warning("Kafka consumer error: %s", message.error())
                continue

            try:
                process_kafka_message(message, processor, consumer, commit_offsets=commit_offsets)
            except Exception as exc:
                logging.exception("Message processing failed without stopping service: %s", exc)

    except KeyboardInterrupt:
        logging.info("Stopping vehicle capture consumer")
    finally:
        consumer.close()

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return run_consumer(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())



