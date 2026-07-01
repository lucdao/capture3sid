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
DEFAULT_CONFIDENCE_THRESHOLD = 0.99
DEFAULT_COOLDOWN_SEC = 60.0
DEFAULT_OUTPUT_ROOT = "server_assets/vehicle_captures"
DEFAULT_REQUEST_TIMEOUT_SEC = 3.0
DEFAULT_RETRIES = 1
DEFAULT_POLL_TIMEOUT_SEC = 1.0
DEFAULT_MAX_EVENT_AGE_SEC = 120.0
DEFAULT_CROP_EXPAND_RATIO = 0.30
DEFAULT_VEHICLE_TYPES = ("car", "truck", "bus", "vehicle")
CAPTURE_EVENT_TYPES = ("object_update", "object_exist")

DEFAULT_SNAPSHOT_BASE_URL = "http://192.168.1.199:9999/snapshot_rtsp"
DEFAULT_SNAPSHOT_TOKEN = ""


@dataclass(frozen=True)
class BBox:
    left: float
    top: float
    width: float
    height: float


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
    event_type: str
    bbox: BBox
    crop_bbox: BBox
    image_width: int
    image_height: int
    received_monotonic: Optional[float]
    ai_result: Dict[str, Any]


@dataclass(frozen=True)
class CameraTarget:
    role: str
    cam_id: str
    ai_enabled: bool = False


@dataclass(frozen=True)
class CameraGroup:
    group_id: str
    ai_camera_role: Optional[str]
    ai_cam_id: Optional[str]
    ai_cam_ids: Tuple[str, ...]
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


def coerce_positive_int(value: Any) -> Optional[int]:
    parsed = coerce_int(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


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


def config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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
            ai_enabled = False
            if isinstance(cam_cfg, dict):
                cam_id = str(cam_cfg.get("cam_id") or "").strip()
                ai_enabled = config_bool(cam_cfg.get("ai"))
            elif cam_cfg is not None:
                cam_id = str(cam_cfg).strip()
            if cam_id:
                owner = seen_cam_ids.get(cam_id)
                if owner and owner != group_id_text:
                    raise ValueError(
                        f"camera {cam_id!r} is configured in multiple groups: {owner!r} and {group_id_text!r}"
                    )
                seen_cam_ids[cam_id] = group_id_text
                targets.append(CameraTarget(role=role_text, cam_id=cam_id, ai_enabled=ai_enabled))

        ai_role = str(group_cfg.get("ai_camera") or "").strip() or None
        ai_cam_id = ""
        if ai_role:
            updated_targets: List[CameraTarget] = []
            for target in targets:
                if target.role == ai_role:
                    ai_cam_id = target.cam_id
                    updated_targets.append(
                        CameraTarget(role=target.role, cam_id=target.cam_id, ai_enabled=True)
                    )
                else:
                    updated_targets.append(target)
            targets = updated_targets
        if ai_role and not ai_cam_id:
            raise ValueError(
                f"camera group {group_id!r} ai_camera={ai_role!r} has no cam_id"
            )
        if not targets:
            raise ValueError(f"camera group {group_id!r} must contain at least one camera")
        ai_cam_ids = tuple(target.cam_id for target in targets if target.ai_enabled)

        groups.append(
            CameraGroup(
                group_id=group_id_text,
                ai_camera_role=ai_role,
                ai_cam_id=ai_cam_id or None,
                ai_cam_ids=ai_cam_ids,
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


def parse_bbox(value: Any) -> Optional[BBox]:
    if isinstance(value, dict):
        left = coerce_float(value.get("left"), 0.0)
        top = coerce_float(value.get("top"), 0.0)
        width = coerce_float(value.get("width"), 0.0)
        height = coerce_float(value.get("height"), 0.0)
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        x1 = coerce_float(value[0], 0.0)
        y1 = coerce_float(value[1], 0.0)
        x2 = coerce_float(value[2], 0.0)
        y2 = coerce_float(value[3], 0.0)
        left = x1
        top = y1
        width = x2 - x1
        height = y2 - y1
    else:
        return None

    if width <= 1.0 or height <= 1.0:
        return None
    return BBox(left=left, top=top, width=width, height=height)


def expand_bbox_to_frame(
    bbox: BBox,
    *,
    image_width: int,
    image_height: int,
    expand_ratio: float = DEFAULT_CROP_EXPAND_RATIO,
) -> Optional[BBox]:
    if image_width <= 0 or image_height <= 0:
        return None

    ratio = max(0.0, float(expand_ratio))
    grow_w = bbox.width * ratio
    grow_h = bbox.height * ratio
    left = max(0.0, bbox.left - grow_w / 2.0)
    top = max(0.0, bbox.top - grow_h / 2.0)
    right = min(float(image_width), bbox.left + bbox.width + grow_w / 2.0)
    bottom = min(float(image_height), bbox.top + bbox.height + grow_h / 2.0)
    width = right - left
    height = bottom - top
    if width <= 1.0 or height <= 1.0:
        return None
    return BBox(left=left, top=top, width=width, height=height)


def bbox_to_query_params(bbox: BBox) -> Dict[str, int]:
    return {
        "crop_left": int(round(bbox.left)),
        "crop_top": int(round(bbox.top)),
        "crop_width": max(1, int(round(bbox.width))),
        "crop_height": max(1, int(round(bbox.height))),
    }


def bbox_to_normalized_query_params(
    bbox: BBox,
    *,
    image_width: int,
    image_height: int,
) -> Dict[str, str]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    return {
        "crop_left_norm": f"{bbox.left / float(image_width):.8f}",
        "crop_top_norm": f"{bbox.top / float(image_height):.8f}",
        "crop_width_norm": f"{bbox.width / float(image_width):.8f}",
        "crop_height_norm": f"{bbox.height / float(image_height):.8f}",
    }


def bbox_to_metadata(bbox: BBox) -> Dict[str, float]:
    return {
        "left": float(bbox.left),
        "top": float(bbox.top),
        "width": float(bbox.width),
        "height": float(bbox.height),
    }


def iter_capture_events(
    payload: Dict[str, Any],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    allow_missing_plate: bool = False,
    vehicle_types: Tuple[str, ...] = DEFAULT_VEHICLE_TYPES,
    crop_expand_ratio: float = DEFAULT_CROP_EXPAND_RATIO,
    received_monotonic: Optional[float] = None,
) -> Iterable[CaptureEvent]:
    ai_results = payload.get("ai_results")
    if not isinstance(ai_results, list):
        return

    image = payload.get("image") if isinstance(payload.get("image"), dict) else {}
    image_width = coerce_positive_int(image.get("width") or image.get("image_width"))
    image_height = coerce_positive_int(image.get("height") or image.get("image_height"))
    if image_width is None or image_height is None:
        logging.debug("Skipping payload without image dimensions message_id=%s", payload.get("message_id"))
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

        event_type = str(result.get("event_type") or "").strip().lower()
        if event_type not in CAPTURE_EVENT_TYPES:
            continue

        confidence = coerce_float(result.get("confidence"), 0.0)
        if confidence < float(confidence_threshold):
            continue

        plate = plate_from_ai_result(result)
        tracking_object_id = coerce_int(result.get("tracking_object_id"))
        bbox = parse_bbox(result.get("bbox"))
        if bbox is None:
            continue
        crop_bbox = expand_bbox_to_frame(
            bbox,
            image_width=image_width,
            image_height=image_height,
            expand_ratio=crop_expand_ratio,
        )
        if crop_bbox is None:
            continue

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
            event_type=event_type,
            bbox=bbox,
            crop_bbox=crop_bbox,
            image_width=image_width,
            image_height=image_height,
            received_monotonic=received_monotonic,
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
    crop_bbox: Optional[BBox] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> str:
    params = {
        "device_id": device_id,
        "token": token,
    }
    if at_time:
        params["at_time"] = at_time
    if crop_bbox is not None:
        if image_width is None or image_height is None:
            params.update(bbox_to_query_params(crop_bbox))
        else:
            params.update(
                bbox_to_normalized_query_params(
                    crop_bbox,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
    return f"{base_url}?{urlencode(params)}"


def redact_secret(text: Any, secret: str) -> str:
    value = str(text)
    if secret:
        value = value.replace(secret, "<redacted>")
    return value


def snapshot_filename(event: CaptureEvent, *, role: str = "snapshot") -> str:
    return f"{snapshot_stem(event, role=role)}.jpg"


def snapshot_stem(event: CaptureEvent, *, role: str = "snapshot") -> str:
    cam = normalize_plate(event.capture_cam_id) or "CAM"
    capture_role = safe_folder_name(event.capture_role or role)
    return f"{capture_role}_{cam}"


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

    def event_stem(self, event: CaptureEvent) -> str:
        return snapshot_stem(event)

    def capture_path(self, event: CaptureEvent) -> Path:
        return self.plate_dir(event.plate) / snapshot_filename(event)

    def metadata_path_for_image(self, image_path: Path) -> Path:
        return image_path.with_suffix(".json")

    def find_existing_record(self, event: CaptureEvent) -> Optional[Dict[str, Any]]:
        stem = self.event_stem(event)
        output_dir = self.plate_dir(event.plate)
        candidate_metadata = [output_dir / f"{stem}.json"]
        candidate_metadata.extend(sorted(output_dir.glob(f"{stem}_track*.json")))

        best_record: Optional[Dict[str, Any]] = None
        best_confidence = float("-inf")
        for metadata_path in candidate_metadata:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            file_path = Path(str(metadata.get("file") or metadata_path.with_suffix(".jpg")))
            if file_path.is_file() and self.is_valid_image_file(file_path):
                metadata["_metadata_path"] = str(metadata_path)
                metadata["_file_path"] = str(file_path)
                confidence = coerce_float(metadata.get("confidence"), float("-inf"))
                if best_record is None or confidence > best_confidence:
                    best_record = metadata
                    best_confidence = confidence
        if best_record is not None:
            return best_record

        candidate_images = [output_dir / f"{stem}.jpg"]
        candidate_images.extend(sorted(output_dir.glob(f"{stem}_track*.jpg")))
        for image_path in candidate_images:
            if self.is_valid_image_file(image_path):
                return {
                    "file": str(image_path),
                    "confidence": None,
                    "_metadata_path": str(self.metadata_path_for_image(image_path)),
                    "_file_path": str(image_path),
                }
        return None

    def should_capture_event(self, event: CaptureEvent) -> bool:
        existing = self.find_existing_record(event)
        if existing is None:
            return True
        existing_confidence = existing.get("confidence")
        if existing_confidence is None:
            return False
        return event.confidence > coerce_float(existing_confidence, 0.0)

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
        if self.find_existing_record(event) is not None:
            return True
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

    def write_capture_metadata(
        self,
        metadata_path: Path,
        event: CaptureEvent,
        output_path: Path,
        *,
        status: Any,
        request_elapsed_ms: float,
        total_latency_ms: Optional[float],
        api_crop_elapsed_ms: Optional[float],
    ) -> None:
        metadata = {
            "plate": safe_folder_name(event.plate),
            "group_id": event.group_id,
            "capture_role": event.capture_role,
            "trigger_cam_id": event.trigger_cam_id,
            "capture_cam_id": event.capture_cam_id,
            "message_id": event.message_id,
            "frame_num": event.frame_num,
            "ntp_timestamp": event.ntp_timestamp,
            "produced_at": event.produced_at,
            "snapshot_at": event.snapshot_at,
            "tracking_object_id": event.tracking_object_id,
            "confidence": event.confidence,
            "event_type": event.event_type,
            "image_width": event.image_width,
            "image_height": event.image_height,
            "bbox": bbox_to_metadata(event.bbox),
            "crop_bbox": bbox_to_metadata(event.crop_bbox),
            "crop_bbox_norm": bbox_to_normalized_query_params(
                event.crop_bbox,
                image_width=event.image_width,
                image_height=event.image_height,
            ),
            "file": str(output_path),
            "status": status,
            "request_elapsed_ms": request_elapsed_ms,
            "total_latency_ms": total_latency_ms,
            "api_crop_elapsed_ms": api_crop_elapsed_ms,
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(metadata, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(tmp_path, metadata_path)

    def cleanup_replaced_record(self, existing_record: Optional[Dict[str, Any]], output_path: Path) -> None:
        if not existing_record:
            return
        old_file_value = existing_record.get("_file_path") or existing_record.get("file")
        old_metadata_value = existing_record.get("_metadata_path")
        old_file = Path(str(old_file_value)) if old_file_value else None
        old_metadata = Path(str(old_metadata_value)) if old_metadata_value else None
        if old_file is not None and old_file != output_path:
            try:
                old_file.unlink(missing_ok=True)
            except OSError as exc:
                logging.warning("Failed to remove replaced snapshot %s: %s", old_file, exc)
        if old_metadata is not None and old_metadata != self.metadata_path_for_image(output_path):
            try:
                old_metadata.unlink(missing_ok=True)
            except OSError as exc:
                logging.warning("Failed to remove replaced metadata %s: %s", old_metadata, exc)
        for old_path in (old_file, old_metadata):
            if old_path is None:
                continue
            parent = old_path.parent
            if parent and parent != self.output_root and parent.is_dir():
                try:
                    parent.rmdir()
                except OSError:
                    pass

    def cleanup_view_duplicates(self, event: CaptureEvent, output_path: Path, metadata_path: Path) -> None:
        output_dir = self.plate_dir(event.plate)
        if not output_dir.is_dir():
            return
        stem = self.event_stem(event)
        duplicate_paths = list(output_dir.glob(f"{stem}_track*.jpg"))
        duplicate_paths.extend(output_dir.glob(f"{stem}_track*.json"))
        duplicate_paths.extend(path for path in (output_dir / f"{stem}.jpg", output_dir / f"{stem}.json"))
        for path in duplicate_paths:
            if path in {output_path, metadata_path}:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logging.warning("Failed to remove duplicate snapshot view file %s: %s", path, exc)

    def capture(self, event: CaptureEvent) -> Dict[str, Any]:
        output_dir = self.plate_dir(event.plate)
        existing_record = self.find_existing_record(event)

        if not event.capture_cam_id:
            raise RuntimeError("Missing cam_id / device_id in message")
        if not event.snapshot_at:
            raise RuntimeError("Missing snapshot timestamp")

        snapshot_url = build_snapshot_url(
            base_url=self.base_url,
            device_id=event.capture_cam_id,
            at_time=event.snapshot_at if self.include_at_time else None,
            token=self.token,
            crop_bbox=event.crop_bbox,
            image_width=event.image_width,
            image_height=event.image_height,
        )
        output_path = self.capture_path(event)
        metadata_path = self.metadata_path_for_image(output_path)

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
                request_started = time.monotonic()
                with self.session.get(snapshot_url, timeout=self.timeout_sec, stream=True) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "").lower()
                    api_crop_elapsed_ms = coerce_float(response.headers.get("X-Crop-Duration-Ms"), -1.0)
                    if api_crop_elapsed_ms < 0:
                        api_crop_elapsed_ms = None

                    output_dir.mkdir(parents=True, exist_ok=True)
                    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
                    tmp_path.unlink(missing_ok=True)
                    with open(tmp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    self.validate_image_file(tmp_path, content_type)
                    os.replace(tmp_path, output_path)
                    request_elapsed_ms = (time.monotonic() - request_started) * 1000.0
                    total_latency_ms = None
                    if event.received_monotonic is not None:
                        total_latency_ms = (time.monotonic() - event.received_monotonic) * 1000.0
                    self.write_capture_metadata(
                        metadata_path,
                        event,
                        output_path,
                        status=response.status_code,
                        request_elapsed_ms=request_elapsed_ms,
                        total_latency_ms=total_latency_ms,
                        api_crop_elapsed_ms=api_crop_elapsed_ms,
                    )
                    self.cleanup_replaced_record(existing_record, output_path)
                    self.cleanup_view_duplicates(event, output_path, metadata_path)

                logging.info(
                    "Snapshot saved group=%s role=%s plate=%s event_type=%s confidence=%.4f trigger_cam=%s capture_cam=%s at_time=%s file=%s status=%s request_ms=%.1f total_latency_ms=%s api_crop_ms=%s",
                    event.group_id,
                    event.capture_role,
                    event.plate,
                    event.event_type,
                    event.confidence,
                    event.trigger_cam_id,
                    event.capture_cam_id,
                    event.snapshot_at,
                    output_path,
                    response.status_code,
                    request_elapsed_ms,
                    f"{total_latency_ms:.1f}" if total_latency_ms is not None else "unknown",
                    f"{api_crop_elapsed_ms:.1f}" if api_crop_elapsed_ms is not None else "unknown",
                )
                return {
                    "url": snapshot_url,
                    "file": str(output_path),
                    "status": response.status_code,
                    "request_elapsed_ms": request_elapsed_ms,
                    "total_latency_ms": total_latency_ms,
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
        crop_expand_ratio: float = DEFAULT_CROP_EXPAND_RATIO,
        service_config: Optional[ServiceConfig] = None,
        deduper: Optional[CooldownDeduper] = None,
    ):
        self.snapshot_client = snapshot_client
        self.confidence_threshold = float(confidence_threshold)
        self.allow_missing_plate = bool(allow_missing_plate)
        self.max_event_age_sec = max(0.0, float(max_event_age_sec))
        self.crop_expand_ratio = max(0.0, float(crop_expand_ratio))
        self.service_config = service_config or ServiceConfig(camera_groups=())
        self.groups_by_trigger_cam: Dict[str, List[CameraGroup]] = {}
        for group in self.service_config.camera_groups:
            if group.ai_cam_ids:
                for ai_cam_id in group.ai_cam_ids:
                    self.groups_by_trigger_cam.setdefault(ai_cam_id, []).append(group)
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
            if group.ai_cam_ids:
                targets = [
                    target
                    for target in group.cameras
                    if target.ai_enabled and target.cam_id == event.trigger_cam_id
                ]
            elif group.ai_cam_id:
                targets = [target for target in group.cameras if target.cam_id == event.trigger_cam_id]
            else:
                targets = list(group.cameras)

            for target in targets:
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
                        event_type=event.event_type,
                        bbox=event.bbox,
                        crop_bbox=event.crop_bbox,
                        image_width=event.image_width,
                        image_height=event.image_height,
                        received_monotonic=event.received_monotonic,
                        ai_result=event.ai_result,
                    )
                )
        return expanded

    def process_payload(
        self,
        payload: Dict[str, Any],
        *,
        received_monotonic: Optional[float] = None,
    ) -> List[CaptureEvent]:
        processed: List[CaptureEvent] = []
        for event in iter_capture_events(
            payload,
            confidence_threshold=self.confidence_threshold,
            allow_missing_plate=self.allow_missing_plate,
            crop_expand_ratio=self.crop_expand_ratio,
            received_monotonic=received_monotonic,
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

            has_any_existing_image = any(
                self.snapshot_client.has_existing_capture_for_event(capture_event)
                for capture_event in capture_events
            )
            if event.event_type == "object_exist" and has_any_existing_image:
                logging.debug(
                    "Skipping object_exist because image already exists plate=%s tracking_object_id=%s",
                    event.plate,
                    event.tracking_object_id,
                )
                continue

            candidate_events = [
                capture_event
                for capture_event in capture_events
                if self.snapshot_client.should_capture_event(capture_event)
            ]
            if not candidate_events:
                if not self.deduper.in_cooldown(dedupe_cam, event.plate):
                    logging.info(
                        "Skipping capture plate=%s event_type=%s tracking_object_id=%s existing_images=%d confidence=%.4f",
                        event.plate,
                        event.event_type,
                        event.tracking_object_id,
                        len(self.snapshot_client.real_images_in_plate_dir(event.plate)),
                        event.confidence,
                    )
                    self.deduper.mark_processed(dedupe_cam, event.plate)
                continue

            if event.event_type == "object_exist" and self.deduper.in_cooldown(dedupe_cam, event.plate):
                logging.debug("Skipping cooldown key=%s plate=%s", dedupe_cam, event.plate)
                continue

            success_count = 0
            for capture_event in candidate_events:
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
            if success_count == len(candidate_events):
                self.deduper.mark_processed(dedupe_cam, event.plate)

        return processed


def process_kafka_message(
    message: Any,
    processor: VehicleCaptureProcessor,
    consumer: Any,
    *,
    commit_offsets: bool = True,
) -> List[CaptureEvent]:
    received_monotonic = time.monotonic()
    payload = decode_message_value(message.value())
    if payload is None:
        if commit_offsets:
            try:
                consumer.commit(message=message, asynchronous=False)
            except Exception as exc:
                logging.warning("Kafka commit failed for invalid message: %s", exc)
        return []

    events = processor.process_payload(payload, received_monotonic=received_monotonic)
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
    parser.add_argument(
        "--crop-expand-ratio",
        type=float,
        default=DEFAULT_CROP_EXPAND_RATIO,
        help="Expand the vehicle bbox by this ratio before sending crop params to the snapshot API.",
    )
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
        crop_expand_ratio=args.crop_expand_ratio,
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
