#!/usr/bin/env python3
import argparse
import hashlib
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
from urllib.parse import urljoin, urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
import yaml

from camera_config_store import (
    InspectionCameraConfigProvider,
    DEFAULT_CONFIG_RELOAD_INTERVAL_SEC,
)


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
DEFAULT_PLATE_CROP_EXPAND_RATIO = 0.0
DEFAULT_SHARPEN_PLATE_IMAGE = False
DEFAULT_PLATE_SHARPEN_AMOUNT = 5.0
DEFAULT_PLATE_SHARPEN_BLUR_KERNEL = 3
DEFAULT_PLATE_SHARPEN_UPSCALE = True
DEFAULT_PLATE_SHARPEN_DENOISE = True
DEFAULT_PLATE_SHARPEN_CLAHE = False
DEFAULT_VEHICLE_TYPES = ("car", "truck", "bus", "vehicle")
STORAGE_SOURCE_EVENT_TYPES = ("object_update", "object_appear")
CAMERA_SOURCE_EVENT_TYPES = ("object_exist",)
CAPTURE_EVENT_TYPES = STORAGE_SOURCE_EVENT_TYPES + CAMERA_SOURCE_EVENT_TYPES
DEFAULT_STORAGE_MANAGE_BASE_URL = "http://192.168.1.199:8011"
DEFAULT_STORAGE_ASSET_RETRIES = 5
DEFAULT_STORAGE_ASSET_RETRY_DELAY_SEC = 0.5
DEFAULT_STORAGE_TIMEZONE = "Asia/Ho_Chi_Minh"
REVIEW_MAX_PLATE_DISTANCE = 2
REVIEW_TIME_WINDOW_SEC = 180.0
REVIEW_CONFIDENCE_CLOSE_DELTA = 0.005

DEFAULT_SNAPSHOT_BASE_URL = "http://192.168.1.199:9999/snapshot_rtsp"
DEFAULT_SNAPSHOT_TOKEN = ""


def sharpen_image(image: Any, *, amount: float = DEFAULT_PLATE_SHARPEN_AMOUNT, blur_kernel: int = DEFAULT_PLATE_SHARPEN_BLUR_KERNEL, **_pipeline_options: Any) -> Any:
    """Apply the configured Gaussian unsharp mask to a plate crop."""
    import cv2

    kernel = int(blur_kernel)
    if kernel < 3:
        kernel = 3
    if kernel % 2 == 0:
        kernel += 1
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
    return cv2.addWeighted(image, 1.0 + float(amount), blurred, -float(amount), 0)


@dataclass(frozen=True)
class BBox:
    left: float
    top: float
    width: float
    height: float


@dataclass(frozen=True)
class PixelCrop:
    left: int
    top: int
    width: int
    height: int


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
    plate_bbox: Optional[BBox]
    image_width: int
    image_height: int
    received_monotonic: Optional[float]
    source_asset_id: Optional[str]
    source_file_name: Optional[str]
    ai_result: Dict[str, Any]
    placeholder: Dict[str, Any]


@dataclass(frozen=True)
class CameraTarget:
    role: str
    cam_id: str


@dataclass(frozen=True)
class CameraGroup:
    group_id: str
    cameras: Tuple[CameraTarget, ...]


@dataclass(frozen=True)
class ServiceConfig:
    camera_groups: Tuple[CameraGroup, ...]


def iter_camera_role_configs(role: Any, cam_cfg: Any) -> Iterable[Tuple[str, Any]]:
    role_text = str(role).strip()
    if isinstance(cam_cfg, list):
        for item in cam_cfg:
            yield role_text, item
        return
    if isinstance(cam_cfg, dict) and "cam_ids" in cam_cfg:
        for cam_id in cam_cfg.get("cam_ids") or []:
            yield role_text, {"cam_id": cam_id}
        return
    yield role_text, cam_cfg


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


def load_service_config_from_config(raw: Dict[str, Any]) -> ServiceConfig:
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
            for role_text, camera_cfg in iter_camera_role_configs(role, cam_cfg):
                if not role_text:
                    raise ValueError(f"camera group {group_id!r} contains an empty camera role")
                seen_roles.add(role_text)

                cam_id = ""
                if isinstance(camera_cfg, dict):
                    cam_id = str(camera_cfg.get("cam_id") or "").strip()
                elif camera_cfg is not None:
                    cam_id = str(camera_cfg).strip()
                if cam_id:
                    owner = seen_cam_ids.get(cam_id)
                    if owner and owner != group_id_text:
                        raise ValueError(
                            f"camera {cam_id!r} is configured in multiple groups: {owner!r} and {group_id_text!r}"
                        )
                    seen_cam_ids[cam_id] = group_id_text
                    targets.append(CameraTarget(role=role_text, cam_id=cam_id))

        if not targets:
            raise ValueError(f"camera group {group_id!r} must contain at least one camera")

        groups.append(
            CameraGroup(
                group_id=group_id_text,
                cameras=tuple(targets),
            )
        )

    return ServiceConfig(camera_groups=tuple(groups))


def load_service_config(path: Optional[str]) -> ServiceConfig:
    if not path:
        return ServiceConfig(camera_groups=())

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return load_service_config_from_config(raw)


def plate_from_ai_result(result: Dict[str, Any]) -> str:
    # Prefer the license-plate-like fields. Fall back to empty string.
    for key in ("detected_object_ids", "name", "object_name"):
        plate = normalize_plate(result.get(key))
        if plate:
            return plate
    return ""


def parse_bbox(value: Any) -> Optional[BBox]:
    if isinstance(value, dict):
        left = coerce_float(value.get("left", value.get("x")), 0.0)
        top = coerce_float(value.get("top", value.get("y")), 0.0)
        width = coerce_float(value.get("width", value.get("w")), 0.0)
        height = coerce_float(value.get("height", value.get("h")), 0.0)
        if (width <= 1.0 or height <= 1.0) and (
            "right" in value or "bottom" in value or "x2" in value or "y2" in value
        ):
            right = coerce_float(value.get("right", value.get("x2")), left)
            bottom = coerce_float(value.get("bottom", value.get("y2")), top)
            width = right - left
            height = bottom - top
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


def parse_plate_bbox_from_placeholder(value: Any) -> Optional[BBox]:
    if not isinstance(value, dict):
        return None
    for key in ("plate_bbox", "license_plate_bbox", "lp_bbox"):
        bbox = parse_bbox(value.get(key))
        if bbox is not None:
            return bbox
    return None


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


def expand_bbox_to_region(
    bbox: BBox,
    *,
    region: BBox,
    expand_ratio: float = 0.0,
) -> Optional[BBox]:
    if region.width <= 0 or region.height <= 0:
        return None

    ratio = max(0.0, float(expand_ratio))
    grow_w = bbox.width * ratio
    grow_h = bbox.height * ratio
    left = max(region.left, bbox.left - grow_w / 2.0)
    top = max(region.top, bbox.top - grow_h / 2.0)
    right = min(region.left + region.width, bbox.left + bbox.width + grow_w / 2.0)
    bottom = min(region.top + region.height, bbox.top + bbox.height + grow_h / 2.0)
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


def normalized_crop_to_pixels_for_frame(
    *,
    bbox: BBox,
    source_width: int,
    source_height: int,
    frame_width: int,
    frame_height: int,
) -> PixelCrop:
    if source_width <= 0 or source_height <= 0 or frame_width <= 0 or frame_height <= 0:
        raise ValueError("source and frame dimensions must be positive")
    left_norm = max(0.0, min(1.0, bbox.left / float(source_width)))
    top_norm = max(0.0, min(1.0, bbox.top / float(source_height)))
    width_norm = max(0.0, min(1.0 - left_norm, bbox.width / float(source_width)))
    height_norm = max(0.0, min(1.0 - top_norm, bbox.height / float(source_height)))

    x1 = max(0, min(frame_width - 1, int(round(left_norm * frame_width))))
    y1 = max(0, min(frame_height - 1, int(round(top_norm * frame_height))))
    x2 = max(x1 + 1, min(frame_width, int(round((left_norm + width_norm) * frame_width))))
    y2 = max(y1 + 1, min(frame_height, int(round((top_norm + height_norm) * frame_height))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("crop is smaller than 2 pixels after scaling")
    return PixelCrop(left=x1, top=y1, width=x2 - x1, height=y2 - y1)


def bbox_to_pixels_in_region(
    *,
    bbox: BBox,
    region: BBox,
    frame_width: int,
    frame_height: int,
) -> PixelCrop:
    if region.width <= 0 or region.height <= 0 or frame_width <= 0 or frame_height <= 0:
        raise ValueError("region and frame dimensions must be positive")

    x1_norm = (bbox.left - region.left) / region.width
    y1_norm = (bbox.top - region.top) / region.height
    x2_norm = (bbox.left + bbox.width - region.left) / region.width
    y2_norm = (bbox.top + bbox.height - region.top) / region.height

    x1 = max(0, min(frame_width - 1, int(round(x1_norm * frame_width))))
    y1 = max(0, min(frame_height - 1, int(round(y1_norm * frame_height))))
    x2 = max(x1 + 1, min(frame_width, int(round(x2_norm * frame_width))))
    y2 = max(y1 + 1, min(frame_height, int(round(y2_norm * frame_height))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("crop is smaller than 2 pixels after scaling")
    return PixelCrop(left=x1, top=y1, width=x2 - x1, height=y2 - y1)


def plate_edit_distance(left: str, right: str) -> int:
    a = normalize_plate(left)
    b = normalize_plate(right)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if ca == cb else 1),
            )
        previous = current
    return previous[-1]


def is_one_missing_character_variant(left: str, right: str) -> bool:
    a = normalize_plate(left)
    b = normalize_plate(right)
    if abs(len(a) - len(b)) != 1:
        return False
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    skipped = 0
    for char in longer:
        if i < len(shorter) and shorter[i] == char:
            i += 1
        else:
            skipped += 1
            if skipped > 1:
                return False
    return i == len(shorter) and skipped == 1


def vietnam_plate_shape_score(plate: str) -> int:
    normalized = normalize_plate(plate)
    if re.fullmatch(r"\d{2}[A-Z]{2}\d{5}", normalized):
        return 3
    if re.fullmatch(r"\d{2}[A-Z]\d{5}", normalized):
        return 2
    if re.fullmatch(r"\d{2}[A-Z]{1,2}\d{4,6}", normalized):
        return 1
    return 0


def canonical_plate_suggestion(records: List[Dict[str, Any]]) -> str:
    if not records:
        return ""

    max_confidence = max(coerce_float(record.get("confidence"), 0.0) for record in records)
    close_records = [
        record
        for record in records
        if max_confidence - coerce_float(record.get("confidence"), 0.0) <= REVIEW_CONFIDENCE_CLOSE_DELTA
    ]

    def key(record: Dict[str, Any]) -> Tuple[int, int, float, str]:
        plate = safe_folder_name(str(record.get("plate") or ""))
        return (
            len(plate),
            vietnam_plate_shape_score(plate),
            coerce_float(record.get("confidence"), 0.0),
            plate,
        )

    return safe_folder_name(str(max(close_records, key=key).get("plate") or ""))


def review_match_reason(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    time_window_sec: float = REVIEW_TIME_WINDOW_SEC,
) -> Optional[str]:
    left_track = coerce_int(left.get("tracking_object_id"))
    right_track = coerce_int(right.get("tracking_object_id"))
    if left_track is not None and right_track is not None and left_track == right_track:
        return "same_tracking_object_id"

    left_plate = str(left.get("plate") or "")
    right_plate = str(right.get("plate") or "")
    if plate_edit_distance(left_plate, right_plate) == 1:
        return "plate_distance_one"

    if not is_one_missing_character_variant(left_plate, right_plate):
        return None

    same_camera_role = (
        str(left.get("capture_cam_id") or "") == str(right.get("capture_cam_id") or "")
        and str(left.get("capture_role") or "") == str(right.get("capture_role") or "")
    )
    if not same_camera_role:
        return None

    left_time = parse_iso_datetime(left.get("produced_at") or left.get("snapshot_at") or left.get("saved_at"))
    right_time = parse_iso_datetime(right.get("produced_at") or right.get("snapshot_at") or right.get("saved_at"))
    if left_time is None or right_time is None:
        return None

    if abs((left_time - right_time).total_seconds()) <= float(time_window_sec):
        return "missing_one_character_time_window"
    return None


def optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_storage_manage_base_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    for suffix in ("/api/storage", "/api/storage/"):
        if base.lower().endswith(suffix.rstrip("/")):
            return base[: -len(suffix.rstrip("/"))].rstrip("/")
    return base


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
    #logging.info(f"Plate bbox: {ai_results}")
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
    payload_asset_id = optional_text(image.get("asset_id"))
    payload_file_name = (
        optional_text(image.get("image_path"))
        or optional_text(image.get("file_name"))
        or optional_text(payload.get("image_path"))
        or optional_text(payload.get("file_name"))
    )

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

        raw_placeholder = result.get("placeholder", {})
        placeholder = raw_placeholder if isinstance(raw_placeholder, dict) else {}
        plate_bbox = parse_plate_bbox_from_placeholder(placeholder)
        # if event_type in STORAGE_SOURCE_EVENT_TYPES:
        #     logging.info(f"Plate bbox: {placeholder}")
        source_file_name = (
            optional_text(result.get("image_path"))
            or optional_text(result.get("file_name"))
            or payload_file_name
        )
        if event_type in STORAGE_SOURCE_EVENT_TYPES:
            logging.debug(
                "Kafka event asset_id=%s image_path=%s event_type=%s plate=%s cam=%s tracking_object_id=%s",
                payload_asset_id or "",
                source_file_name or "",
                event_type,
                plate,
                trigger_cam_id,
                tracking_object_id,
            )

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
            plate_bbox=plate_bbox,
            image_width=image_width,
            image_height=image_height,
            received_monotonic=received_monotonic,
            source_asset_id=payload_asset_id,
            source_file_name=source_file_name,
            ai_result=dict(result),
            placeholder=placeholder
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


class StorageAuthorizationError(RuntimeError):
    pass


class StorageAssetClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_STORAGE_MANAGE_BASE_URL,
        token: str = "",
        timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
        retries: int = DEFAULT_STORAGE_ASSET_RETRIES,
        retry_delay_sec: float = DEFAULT_STORAGE_ASSET_RETRY_DELAY_SEC,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = normalize_storage_manage_base_url(base_url)
        self.token = str(token or "")
        self.timeout_sec = float(timeout_sec)
        self.retries = max(0, int(retries))
        self.retry_delay_sec = max(0.0, float(retry_delay_sec))
        self.session = session or requests.Session()

    def headers(self) -> Dict[str, str]:
        return {"X-Service-Token": self.token} if self.token else {}

    def get_access_url(self, asset_id: Any) -> str:
        asset_id_text = optional_text(asset_id)
        if asset_id_text is None:
            raise ValueError("asset_id is required")
        response = self.session.post(
            f"{self.base_url}/api/storage/get-access-url",
            json={
                "asset_id": asset_id_text,
                "access_scope": "file",
                "duration": 300,
                "as_attachment": False,
            },
            headers=self.headers(),
            timeout=self.timeout_sec,
        )
        if response.status_code == 401:
            raise StorageAuthorizationError("storage API returned 401 unauthorized while requesting asset access URL")
        if response.status_code == 403:
            raise StorageAuthorizationError("storage API returned 403 forbidden while requesting asset access URL")
        response.raise_for_status()
        return urljoin(f"{self.base_url}/", str(response.json()["access_url"]))

    def download_by_asset_id(self, asset_id: Any) -> Tuple[bytes, Dict[str, Any], str]:
        asset_id_text = optional_text(asset_id)
        if asset_id_text is None:
            raise ValueError("asset_id is required")
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                preview_url = self.get_access_url(asset_id_text)
                response = self.session.get(preview_url, timeout=self.timeout_sec)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if content_type and "image" not in content_type:
                    raise RuntimeError(f"storage preview response is not an image: {content_type}")
                if not response.content:
                    raise RuntimeError("storage preview returned empty body")
                return response.content, {"id": asset_id_text}, preview_url
            except StorageAuthorizationError as exc:
                raise RuntimeError(f"failed to download storage asset_id={asset_id_text}: {exc}") from exc
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_delay_sec)
        raise RuntimeError(
            f"failed to download storage asset_id={asset_id_text} "
            f"after {self.retries + 1} attempts: {last_error}"
        )


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
        storage_asset_client: Optional[StorageAssetClient] = None,
        session: Optional[requests.Session] = None,
        storage_timezone: str = DEFAULT_STORAGE_TIMEZONE,
        plate_crop_expand_ratio: float = DEFAULT_PLATE_CROP_EXPAND_RATIO,
        sharpen_plate_image: bool = DEFAULT_SHARPEN_PLATE_IMAGE,
        plate_sharpen_amount: float = DEFAULT_PLATE_SHARPEN_AMOUNT,
        plate_sharpen_blur_kernel: int = DEFAULT_PLATE_SHARPEN_BLUR_KERNEL,
        plate_sharpen_upscale: bool = DEFAULT_PLATE_SHARPEN_UPSCALE,
        plate_sharpen_denoise: bool = DEFAULT_PLATE_SHARPEN_DENOISE,
        plate_sharpen_clahe: bool = DEFAULT_PLATE_SHARPEN_CLAHE,
    ):
        self.base_url = str(base_url).rstrip("?")
        self.token = str(token)
        self.output_root = Path(output_root)
        self.timeout_sec = float(timeout_sec)
        self.retries = max(0, int(retries))
        self.dry_run = bool(dry_run)
        self.include_at_time = bool(include_at_time)
        self.session = session or requests.Session()
        self.storage_asset_client = storage_asset_client
        self.plate_crop_expand_ratio = max(0.0, float(plate_crop_expand_ratio))
        self.sharpen_plate_image = bool(sharpen_plate_image)
        self.plate_sharpen_amount = float(plate_sharpen_amount)
        self.plate_sharpen_blur_kernel = int(plate_sharpen_blur_kernel)
        self.plate_sharpen_upscale = bool(plate_sharpen_upscale)
        self.plate_sharpen_denoise = bool(plate_sharpen_denoise)
        self.plate_sharpen_clahe = bool(plate_sharpen_clahe)
        self.storage_timezone_name = str(storage_timezone or DEFAULT_STORAGE_TIMEZONE)
        try:
            self.storage_timezone = ZoneInfo(self.storage_timezone_name)
        except ZoneInfoNotFoundError:
            logging.warning(
                "Storage timezone %s is not available; falling back to UTC",
                self.storage_timezone_name,
            )
            self.storage_timezone = timezone.utc
            self.storage_timezone_name = "UTC"

    def current_storage_date(self) -> str:
        return datetime.now(self.storage_timezone).date().isoformat()

    def plate_dir(self, plate: str, *, storage_date: Optional[str] = None) -> Path:
        date_part = str(storage_date or self.current_storage_date()).strip()
        return self.output_root / date_part / safe_folder_name(plate)

    def event_stem(self, event: CaptureEvent) -> str:
        return snapshot_stem(event)

    def capture_path(self, event: CaptureEvent, *, storage_date: Optional[str] = None) -> Path:
        return self.plate_dir(event.plate, storage_date=storage_date) / snapshot_filename(event)

    def plate_capture_path(self, event: CaptureEvent, *, storage_date: Optional[str] = None) -> Path:
        image_path = self.capture_path(event, storage_date=storage_date)
        return image_path.with_name(f"{image_path.stem}_plate.jpg")

    def metadata_path_for_image(self, image_path: Path) -> Path:
        return image_path.with_suffix(".json")

    def find_existing_record(
        self,
        event: CaptureEvent,
        *,
        storage_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        stem = self.event_stem(event)
        output_dir = self.plate_dir(event.plate, storage_date=storage_date)
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

    def should_capture_event(self, event: CaptureEvent, *, storage_date: Optional[str] = None) -> bool:
        existing = self.find_existing_record(event, storage_date=storage_date)
        if existing is None:
            return True
        existing_confidence = existing.get("confidence")
        if existing_confidence is None:
            return False
        return event.confidence > coerce_float(existing_confidence, 0.0)

    def real_images_in_plate_dir(self, plate: str, *, storage_date: Optional[str] = None) -> List[Path]:
        output_dir = self.plate_dir(plate, storage_date=storage_date)
        if not output_dir.is_dir():
            return []
        return [
            path
            for path in output_dir.glob("*.jpg")
            if self.is_valid_image_file(path)
        ]

    def has_existing_capture(
        self,
        plate: str,
        *,
        min_images: int = 1,
        storage_date: Optional[str] = None,
    ) -> bool:
        return len(self.real_images_in_plate_dir(plate, storage_date=storage_date)) >= max(1, int(min_images))

    def has_existing_capture_for_events(
        self,
        events: List[CaptureEvent],
        *,
        storage_date: Optional[str] = None,
    ) -> bool:
        if not events:
            return False

        for event in events:
            if not self.has_existing_capture_for_event(event, storage_date=storage_date):
                return False
        return True

    def has_existing_capture_for_event(
        self,
        event: CaptureEvent,
        *,
        storage_date: Optional[str] = None,
    ) -> bool:
        if self.find_existing_record(event, storage_date=storage_date) is not None:
            return True
        output_dir = self.plate_dir(event.plate, storage_date=storage_date)
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
        source_method: str,
        storage_asset: Optional[Dict[str, Any]] = None,
        storage_preview_url: Optional[str] = None,
        plate_output_path: Optional[Path] = None,
        storage_date: Optional[str] = None,
    ) -> None:
        resolved_storage_date = str(storage_date or self.current_storage_date())
        metadata = {
            "plate": safe_folder_name(event.plate),
            "storage_date": resolved_storage_date,
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
            "asset_id": event.source_asset_id,
            "image_path": event.source_file_name,
            "source_file_name": event.source_file_name,
            "source_method": source_method,
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
        if event.placeholder:
            metadata["placeholder"] = event.placeholder
        if event.plate_bbox is not None:
            metadata["plate_bbox"] = bbox_to_metadata(event.plate_bbox)
            metadata["plate_bbox_norm"] = bbox_to_normalized_query_params(
                event.plate_bbox,
                image_width=event.image_width,
                image_height=event.image_height,
            )
            metadata["plate_crop_expand_ratio"] = self.plate_crop_expand_ratio
            metadata["sharpen_plate_image"] = self.sharpen_plate_image
            if self.sharpen_plate_image:
                metadata["plate_sharpen_amount"] = self.plate_sharpen_amount
                metadata["plate_sharpen_blur_kernel"] = self.plate_sharpen_blur_kernel
                metadata["plate_sharpen_upscale"] = self.plate_sharpen_upscale
                metadata["plate_sharpen_denoise"] = self.plate_sharpen_denoise
                metadata["plate_sharpen_clahe"] = self.plate_sharpen_clahe
        if plate_output_path is not None and plate_output_path.is_file():
            metadata["plate_image_file"] = str(plate_output_path)
        if storage_asset is not None:
            metadata["storage_asset"] = {
                key: storage_asset[key]
                for key in ("id", "file_path", "filename", "device_id", "timestamp")
                if storage_asset.get(key) is not None
            }
        if storage_preview_url is not None:
            metadata["storage_preview_url"] = storage_preview_url
        tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(metadata, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(tmp_path, metadata_path)

    def iter_storage_date_metadata(self, storage_date: str) -> Iterable[Tuple[Path, Dict[str, Any]]]:
        date_dir = self.output_root / str(storage_date)
        if not date_dir.is_dir():
            return
        for metadata_path in sorted(date_dir.glob("*/*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logging.warning("Failed to read capture metadata for review %s: %s", metadata_path, exc)
                continue
            if isinstance(metadata, dict):
                yield metadata_path, metadata

    def review_manifest_path(self, storage_date: str) -> Path:
        return self.output_root / "_review" / str(storage_date) / "suspicious_plates.jsonl"

    def review_group_id(self, storage_date: str, metadata_paths: List[Path]) -> str:
        digest_source = "|".join(sorted(str(path) for path in metadata_paths))
        digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
        return f"{storage_date}_{digest}"

    def review_group_already_recorded(self, storage_date: str, review_group_id: str) -> bool:
        manifest_path = self.review_manifest_path(storage_date)
        if not manifest_path.is_file():
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and record.get("review_group_id") == review_group_id:
                        return True
        except OSError as exc:
            logging.warning("Failed to read review manifest %s: %s", manifest_path, exc)
        return False

    def update_metadata_review_fields(
        self,
        metadata_path: Path,
        *,
        review_group_id: str,
        review_reason: str,
        similar_plates: List[str],
        canonical_plate: str,
    ) -> None:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Failed to annotate review metadata %s: %s", metadata_path, exc)
            return
        if not isinstance(metadata, dict):
            return

        existing_similar = metadata.get("similar_plates")
        merged_similar = set(similar_plates)
        if isinstance(existing_similar, list):
            merged_similar.update(str(value) for value in existing_similar if value)

        metadata["review_status"] = "candidate"
        metadata["review_reason"] = review_reason
        metadata["review_group_id"] = review_group_id
        metadata["similar_plates"] = sorted(merged_similar)
        metadata["canonical_plate_suggestion"] = canonical_plate

        tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        try:
            tmp_path.write_text(json.dumps(metadata, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
            os.replace(tmp_path, metadata_path)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            logging.warning("Failed to write review metadata %s: %s", metadata_path, exc)

    def flag_similar_plates_for_review(
        self,
        *,
        storage_date: str,
        event: CaptureEvent,
        metadata_path: Path,
    ) -> None:
        try:
            current_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Failed to load saved metadata for review %s: %s", metadata_path, exc)
            return
        if not isinstance(current_metadata, dict):
            return

        current_plate = safe_folder_name(str(current_metadata.get("plate") or event.plate))
        matches: List[Tuple[Path, Dict[str, Any], str, int]] = []
        for candidate_path, candidate_metadata in self.iter_storage_date_metadata(storage_date):
            if candidate_path == metadata_path:
                continue
            candidate_plate = safe_folder_name(str(candidate_metadata.get("plate") or candidate_path.parent.name))
            if not candidate_plate or candidate_plate == current_plate:
                continue
            distance = plate_edit_distance(current_plate, candidate_plate)
            if distance > REVIEW_MAX_PLATE_DISTANCE:
                continue
            reason = review_match_reason(current_metadata, candidate_metadata)
            if reason is None:
                continue
            matches.append((candidate_path, candidate_metadata, reason, distance))

        if not matches:
            return

        records = [current_metadata] + [metadata for _, metadata, _, _ in matches]
        canonical_plate = canonical_plate_suggestion(records)
        participant_paths = [metadata_path] + [path for path, _, _, _ in matches]
        group_id = self.review_group_id(storage_date, participant_paths)
        all_plates = sorted(
            {
                safe_folder_name(str(metadata.get("plate") or path.parent.name))
                for path, metadata in [(metadata_path, current_metadata)]
                + [(path, metadata) for path, metadata, _, _ in matches]
            }
        )
        reasons = sorted({reason for _, _, reason, _ in matches})
        review_reason = ",".join(reasons)

        manifest_record = {
            "review_group_id": group_id,
            "storage_date": storage_date,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "review_status": "candidate",
            "review_reason": review_reason,
            "canonical_plate_suggestion": canonical_plate,
            "plates": all_plates,
            "candidates": [
                {
                    "plate": safe_folder_name(str(current_metadata.get("plate") or event.plate)),
                    "metadata_path": str(metadata_path),
                    "confidence": current_metadata.get("confidence"),
                    "tracking_object_id": current_metadata.get("tracking_object_id"),
                    "capture_cam_id": current_metadata.get("capture_cam_id"),
                    "capture_role": current_metadata.get("capture_role"),
                    "produced_at": current_metadata.get("produced_at"),
                }
            ]
            + [
                {
                    "plate": safe_folder_name(str(metadata.get("plate") or path.parent.name)),
                    "metadata_path": str(path),
                    "confidence": metadata.get("confidence"),
                    "tracking_object_id": metadata.get("tracking_object_id"),
                    "capture_cam_id": metadata.get("capture_cam_id"),
                    "capture_role": metadata.get("capture_role"),
                    "produced_at": metadata.get("produced_at"),
                    "plate_edit_distance": distance,
                    "match_reason": reason,
                }
                for path, metadata, reason, distance in matches
            ],
        }

        if not self.review_group_already_recorded(storage_date, group_id):
            manifest_path = self.review_manifest_path(storage_date)
            try:
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(manifest_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(manifest_record, ensure_ascii=True, sort_keys=True) + "\n")
            except OSError as exc:
                logging.warning("Failed to write review manifest %s: %s", manifest_path, exc)

        for participant_path, participant_metadata in [(metadata_path, current_metadata)] + [
            (path, metadata) for path, metadata, _, _ in matches
        ]:
            participant_plate = safe_folder_name(str(participant_metadata.get("plate") or participant_path.parent.name))
            self.update_metadata_review_fields(
                participant_path,
                review_group_id=group_id,
                review_reason=review_reason,
                similar_plates=[plate for plate in all_plates if plate != participant_plate],
                canonical_plate=canonical_plate,
            )

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

    def cleanup_view_duplicates(
        self,
        event: CaptureEvent,
        output_path: Path,
        metadata_path: Path,
        *,
        storage_date: Optional[str] = None,
    ) -> None:
        output_dir = self.plate_dir(event.plate, storage_date=storage_date)
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

    def write_frame_crop_to_file(self, frame: Any, crop: PixelCrop, output_path: Path, *, sharpen: bool = False) -> None:
        import cv2

        crop_frame = frame[crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
        if sharpen:
            crop_frame = sharpen_image(
                crop_frame,
                amount=self.plate_sharpen_amount,
                blur_kernel=self.plate_sharpen_blur_kernel,
                upscale=self.plate_sharpen_upscale,
                denoise=self.plate_sharpen_denoise,
                clahe=self.plate_sharpen_clahe,
            )
        ok, output = cv2.imencode(".jpg", crop_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("failed to encode cropped image")

        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.unlink(missing_ok=True)
        tmp_path.write_bytes(output.tobytes())
        self.validate_image_file(tmp_path, "image/jpeg")
        os.replace(tmp_path, output_path)

    def crop_plate_from_region_to_file(
        self,
        frame: Any,
        event: CaptureEvent,
        *,
        region: BBox,
        output_path: Path,
    ) -> None:
        if event.plate_bbox is None:
            return
        frame_height, frame_width = frame.shape[:2]
        plate_bbox = expand_bbox_to_region(
            event.plate_bbox,
            region=region,
            expand_ratio=self.plate_crop_expand_ratio,
        )
        if plate_bbox is None:
            raise ValueError("expanded plate crop is empty")
        crop = bbox_to_pixels_in_region(
            bbox=plate_bbox,
            region=region,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self.write_frame_crop_to_file(frame, crop, output_path, sharpen=self.sharpen_plate_image)

    def crop_plate_from_image_file(self, image_path: Path, event: CaptureEvent, output_path: Path) -> bool:
        if event.plate_bbox is None:
            return False
        import cv2

        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("snapshot image could not be decoded for plate crop")
        self.crop_plate_from_region_to_file(
            frame,
            event,
            region=event.crop_bbox,
            output_path=output_path,
        )
        return True

    def crop_storage_image_to_file(
        self,
        image_bytes: bytes,
        event: CaptureEvent,
        output_path: Path,
        *,
        plate_output_path: Optional[Path] = None,
    ) -> bool:
        import cv2
        import numpy as np

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("storage image could not be decoded")

        frame_height, frame_width = frame.shape[:2]
        crop = normalized_crop_to_pixels_for_frame(
            bbox=event.crop_bbox,
            source_width=event.image_width,
            source_height=event.image_height,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self.write_frame_crop_to_file(frame, crop, output_path)

        if plate_output_path is None or event.plate_bbox is None:
            return False
        full_region = BBox(
            left=0.0,
            top=0.0,
            width=float(event.image_width),
            height=float(event.image_height),
        )
        try:
            self.crop_plate_from_region_to_file(
                frame,
                event,
                region=full_region,
                output_path=plate_output_path,
            )
            return True
        except Exception as exc:
            logging.warning(
                "Failed to crop plate image from storage plate=%s asset_id=%s file=%s plate_bbox=%s: %s",
                event.plate,
                event.source_asset_id or "",
                event.source_file_name or "",
                bbox_to_metadata(event.plate_bbox),
                exc,
            )
            return False

    def capture_from_storage_source(self, event: CaptureEvent, *, storage_date: Optional[str] = None) -> Dict[str, Any]:
        storage_date = str(storage_date or self.current_storage_date())
        if self.storage_asset_client is None:
            raise RuntimeError("storage asset client is not configured")
        if not event.source_asset_id:
            raise RuntimeError("event has no source_asset_id")

        output_dir = self.plate_dir(event.plate, storage_date=storage_date)
        existing_record = self.find_existing_record(event, storage_date=storage_date)
        output_path = self.capture_path(event, storage_date=storage_date)
        plate_output_path = self.plate_capture_path(event, storage_date=storage_date) if event.plate_bbox is not None else None
        metadata_path = self.metadata_path_for_image(output_path)

        if self.dry_run:
            logging.info("Dry-run storage snapshot asset_id=%s output=%s", event.source_asset_id, output_path)
            return {"url": f"asset:{event.source_asset_id}", "file": str(output_path), "status": "dry_run"}

        request_started = time.monotonic()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_bytes, asset, preview_url = self.storage_asset_client.download_by_asset_id(event.source_asset_id)
        crop_started = time.monotonic()
        wrote_plate_crop = self.crop_storage_image_to_file(
            image_bytes,
            event,
            output_path,
            plate_output_path=plate_output_path,
        )
        crop_elapsed_ms = (time.monotonic() - crop_started) * 1000.0
        request_elapsed_ms = (time.monotonic() - request_started) * 1000.0
        total_latency_ms = None
        if event.received_monotonic is not None:
            total_latency_ms = (time.monotonic() - event.received_monotonic) * 1000.0
        self.write_capture_metadata(
            metadata_path,
            event,
            output_path,
            status=200,
            request_elapsed_ms=request_elapsed_ms,
            total_latency_ms=total_latency_ms,
            api_crop_elapsed_ms=crop_elapsed_ms,
            source_method="storage_asset_id",
            storage_asset=asset,
            storage_preview_url=preview_url,
            plate_output_path=plate_output_path if wrote_plate_crop else None,
            storage_date=storage_date,
        )
        self.cleanup_replaced_record(existing_record, output_path)
        self.cleanup_view_duplicates(event, output_path, metadata_path, storage_date=storage_date)
        self.flag_similar_plates_for_review(
            storage_date=storage_date,
            event=event,
            metadata_path=metadata_path,
        )

        logging.info(
            "Snapshot saved from storage group=%s role=%s plate=%s event_type=%s confidence=%.4f trigger_cam=%s asset_id=%s image_path=%s file=%s request_ms=%.1f total_latency_ms=%s crop_ms=%.1f",
            event.group_id,
            event.capture_role,
            event.plate,
            event.event_type,
            event.confidence,
            event.trigger_cam_id,
            event.source_asset_id,
            event.source_file_name,
            output_path,
            request_elapsed_ms,
            f"{total_latency_ms:.1f}" if total_latency_ms is not None else "unknown",
            crop_elapsed_ms,
        )
        return {
            "url": preview_url,
            "file": str(output_path),
            "status": 200,
            "request_elapsed_ms": request_elapsed_ms,
            "total_latency_ms": total_latency_ms,
        }

    def capture_from_camera(self, event: CaptureEvent, *, storage_date: Optional[str] = None) -> Dict[str, Any]:
        storage_date = str(storage_date or self.current_storage_date())
        output_dir = self.plate_dir(event.plate, storage_date=storage_date)
        existing_record = self.find_existing_record(event, storage_date=storage_date)

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
        output_path = self.capture_path(event, storage_date=storage_date)
        plate_output_path = self.plate_capture_path(event, storage_date=storage_date) if event.plate_bbox is not None else None
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
                    wrote_plate_crop = False
                    if plate_output_path is not None:
                        try:
                            wrote_plate_crop = self.crop_plate_from_image_file(
                                output_path,
                                event,
                                plate_output_path,
                            )
                        except Exception as exc:
                            logging.warning(
                                "Failed to crop plate image from camera snapshot plate=%s file=%s plate_bbox=%s: %s",
                                event.plate,
                                output_path,
                                bbox_to_metadata(event.plate_bbox) if event.plate_bbox is not None else None,
                                exc,
                            )
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
                        source_method="camera_capture_api",
                        plate_output_path=plate_output_path if wrote_plate_crop else None,
                        storage_date=storage_date,
                    )
                    self.cleanup_replaced_record(existing_record, output_path)
                    self.cleanup_view_duplicates(event, output_path, metadata_path, storage_date=storage_date)
                    self.flag_similar_plates_for_review(
                        storage_date=storage_date,
                        event=event,
                        metadata_path=metadata_path,
                    )

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

    def capture(self, event: CaptureEvent, *, storage_date: Optional[str] = None) -> Dict[str, Any]:
        storage_date = str(storage_date or self.current_storage_date())
        if event.event_type in STORAGE_SOURCE_EVENT_TYPES and event.source_asset_id:
            return self.capture_from_storage_source(event, storage_date=storage_date)
        if event.event_type in STORAGE_SOURCE_EVENT_TYPES and not event.source_asset_id:
            logging.warning(
                "%s has no asset_id; falling back to camera capture plate=%s cam=%s tracking_object_id=%s",
                event.event_type,
                event.plate,
                event.trigger_cam_id,
                event.tracking_object_id,
            )
        return self.capture_from_camera(event, storage_date=storage_date)


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
        config_provider: Optional[InspectionCameraConfigProvider] = None,
        config_reload_interval_sec: float = DEFAULT_CONFIG_RELOAD_INTERVAL_SEC,
    ):
        self.snapshot_client = snapshot_client
        self.confidence_threshold = float(confidence_threshold)
        self.allow_missing_plate = bool(allow_missing_plate)
        self.max_event_age_sec = max(0.0, float(max_event_age_sec))
        self.crop_expand_ratio = max(0.0, float(crop_expand_ratio))
        self.service_config = service_config or ServiceConfig(camera_groups=())
        self.groups_by_trigger_cam: Dict[str, List[CameraGroup]] = {}
        self._rebuild_group_index()
        self.deduper = deduper or CooldownDeduper(cooldown_sec)
        self.config_provider = config_provider
        self.config_reload_interval_sec = max(0.1, float(config_reload_interval_sec))
        self._last_reload_check = 0.0
        self._config_version = config_provider.current_version() if config_provider else None

    def _rebuild_group_index(self) -> None:
        self.groups_by_trigger_cam = {}
        for group in self.service_config.camera_groups:
            for target in group.cameras:
                self.groups_by_trigger_cam.setdefault(target.cam_id, []).append(group)

    def reload_config_if_needed(self, *, force: bool = False) -> None:
        if self.config_provider is None:
            return
        now = time.monotonic()
        if not force and now - self._last_reload_check < self.config_reload_interval_sec:
            return
        self._last_reload_check = now
        try:
            state = self.config_provider.load()
            if not force and state.version == self._config_version:
                return
            service_config = load_service_config_from_config(state.config)
        except Exception as exc:
            logging.warning("Failed to reload camera config for capture service: %s", exc)
            return
        self.service_config = service_config
        self._rebuild_group_index()
        self._config_version = state.version
        logging.info(
            "Reloaded capture camera config version=%s camera_groups=%d",
            state.version,
            len(service_config.camera_groups),
        )

    def expand_group_events(self, event: CaptureEvent) -> List[CaptureEvent]:
        groups = self.groups_by_trigger_cam.get(event.trigger_cam_id)
        if not groups:
            if self.service_config.camera_groups:
                logging.debug("Skipping cam=%s because it is not configured in any camera_group", event.trigger_cam_id)
                return []
            return [event]

        expanded: List[CaptureEvent] = []
        for group in groups:
            targets = [target for target in group.cameras if target.cam_id == event.trigger_cam_id]

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
                        plate_bbox=event.plate_bbox,
                        image_width=event.image_width,
                        image_height=event.image_height,
                        received_monotonic=event.received_monotonic,
                        source_asset_id=event.source_asset_id,
                        source_file_name=event.source_file_name,
                        ai_result=event.ai_result,
                        placeholder=event.placeholder,
                    )
                )
        return expanded

    def process_payload(
        self,
        payload: Dict[str, Any],
        *,
        received_monotonic: Optional[float] = None,
    ) -> List[CaptureEvent]:
        self.reload_config_if_needed()
        processed: List[CaptureEvent] = []
        storage_date = self.snapshot_client.current_storage_date()
        for event in iter_capture_events(
            payload,
            confidence_threshold=self.confidence_threshold,
            allow_missing_plate=self.allow_missing_plate,
            crop_expand_ratio=self.crop_expand_ratio,
            received_monotonic=received_monotonic,
        ):
            groups = self.groups_by_trigger_cam.get(event.trigger_cam_id)
            if self.service_config.camera_groups and not groups:
                logging.debug(
                    "Skipping cam=%s because it is not configured in any camera_group plate=%s event_type=%s asset_id=%s image_path=%s",
                    event.trigger_cam_id,
                    event.plate,
                    event.event_type,
                    event.source_asset_id or "",
                    event.source_file_name or "",
                )
                continue

            age = event_age_sec(event)
            if age is not None and age > self.max_event_age_sec:
                logging.debug(
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
                self.snapshot_client.has_existing_capture_for_event(capture_event, storage_date=storage_date)
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
                if self.snapshot_client.should_capture_event(capture_event, storage_date=storage_date)
            ]
            if not candidate_events:
                if not self.deduper.in_cooldown(dedupe_cam, event.plate):
                    logging.info(
                        "Skipping capture plate=%s event_type=%s tracking_object_id=%s existing_images=%d confidence=%.4f",
                        event.plate,
                        event.event_type,
                        event.tracking_object_id,
                        len(self.snapshot_client.real_images_in_plate_dir(event.plate, storage_date=storage_date)),
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
                    self.snapshot_client.capture(capture_event, storage_date=storage_date)
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
    parser.add_argument(
        "--plate-crop-expand-ratio",
        type=float,
        default=DEFAULT_PLATE_CROP_EXPAND_RATIO,
        help="Expand the plate bbox by this ratio before saving the plate crop.",
    )
    parser.add_argument(
        "--sharpen-plate-image",
        action="store_true",
        default=str(os.environ.get("SHARPEN_PLATE_IMAGE", "false")).strip().lower() in {"1", "true", "yes", "on"},
        help="Apply unsharp masking only to saved plate crops.",
    )
    parser.add_argument("--plate-sharpen-amount", type=float, default=float(os.environ.get("PLATE_SHARPEN_AMOUNT", DEFAULT_PLATE_SHARPEN_AMOUNT)))
    parser.add_argument("--plate-sharpen-blur-kernel", type=int, default=int(os.environ.get("PLATE_SHARPEN_BLUR_KERNEL", DEFAULT_PLATE_SHARPEN_BLUR_KERNEL)))
    parser.add_argument("--plate-sharpen-upscale", action="store_true", default=str(os.environ.get("PLATE_SHARPEN_UPSCALE", "true")).lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--plate-sharpen-denoise", action="store_true", default=str(os.environ.get("PLATE_SHARPEN_DENOISE", "true")).lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--plate-sharpen-clahe", action="store_true", default=str(os.environ.get("PLATE_SHARPEN_CLAHE", "false")).lower() in {"1", "true", "yes", "on"})
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
    parser.add_argument(
        "--config-reload-interval-sec",
        type=float,
        default=float(os.environ.get("CAMERA_CONFIG_RELOAD_INTERVAL_SEC", DEFAULT_CONFIG_RELOAD_INTERVAL_SEC)),
    )
    parser.add_argument(
        "--storage-timezone",
        default=os.environ.get("STORAGE_TIMEZONE", DEFAULT_STORAGE_TIMEZONE),
        help="Timezone used to choose the YYYY-MM-DD storage folder.",
    )

    parser.add_argument("--snapshot-base-url", default=DEFAULT_SNAPSHOT_BASE_URL)
    parser.add_argument("--snapshot-token", default=DEFAULT_SNAPSHOT_TOKEN)
    parser.add_argument("--storage-manage-base-url", default=os.environ.get("STORAGE_MANAGE_BASE_URL", DEFAULT_STORAGE_MANAGE_BASE_URL))
    parser.add_argument("--storage-manage-token", default=os.environ.get("STORAGE_MANAGE_TOKEN", ""))
    parser.add_argument("--storage-asset-retries", type=int, default=int(os.environ.get("STORAGE_ASSET_RETRIES", DEFAULT_STORAGE_ASSET_RETRIES)))
    parser.add_argument(
        "--storage-asset-retry-delay-sec",
        type=float,
        default=float(os.environ.get("STORAGE_ASSET_RETRY_DELAY_SEC", DEFAULT_STORAGE_ASSET_RETRY_DELAY_SEC)),
    )
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

    storage_asset_client = StorageAssetClient(
        base_url=args.storage_manage_base_url,
        token=args.storage_manage_token,
        timeout_sec=args.request_timeout_sec,
        retries=args.storage_asset_retries,
        retry_delay_sec=args.storage_asset_retry_delay_sec,
    )

    snapshot_client = SnapshotClient(
        base_url=args.snapshot_base_url,
        token=args.snapshot_token,
        output_root=args.output_root,
        timeout_sec=args.request_timeout_sec,
        retries=args.retries,
        dry_run=args.dry_run,
        include_at_time=args.snapshot_include_at_time,
        storage_asset_client=storage_asset_client,
        storage_timezone=args.storage_timezone,
        plate_crop_expand_ratio=args.plate_crop_expand_ratio,
        sharpen_plate_image=args.sharpen_plate_image,
        plate_sharpen_amount=args.plate_sharpen_amount,
        plate_sharpen_blur_kernel=args.plate_sharpen_blur_kernel,
        plate_sharpen_upscale=args.plate_sharpen_upscale,
        plate_sharpen_denoise=args.plate_sharpen_denoise,
        plate_sharpen_clahe=args.plate_sharpen_clahe,
    )
    if not args.dry_run and not str(args.snapshot_token or "").strip():
        raise ValueError("snapshot token is required unless --dry-run is used")
    config_provider = InspectionCameraConfigProvider()
    config_state = config_provider.load(force=True)
    service_config = load_service_config_from_config(config_state.config)
    processor = VehicleCaptureProcessor(
        snapshot_client,
        confidence_threshold=args.confidence_threshold,
        cooldown_sec=args.cooldown_sec,
        max_event_age_sec=args.max_event_age_sec,
        crop_expand_ratio=args.crop_expand_ratio,
        allow_missing_plate=args.allow_missing_plate,
        service_config=service_config,
        config_provider=config_provider,
        config_reload_interval_sec=args.config_reload_interval_sec,
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
