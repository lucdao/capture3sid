#!/usr/bin/env python3
import argparse
import base64
import binascii
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from camera_config_store import (
    InspectionCameraConfigProvider,
    DEFAULT_CONFIG_RELOAD_INTERVAL_SEC,
)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8890
DEFAULT_OUTPUT_ROOT = "/data/vehicle_captures"
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
DEFAULT_STORAGE_TIMEZONE = "Asia/Ho_Chi_Minh"


class SubmitPhoto(BaseModel):
    type: str
    image: str
    plate_image: Optional[str] = None


class SubmitPayload(BaseModel):
    plate: str
    plateSource: Optional[str] = None
    deviceId: Optional[str] = None
    timestamp: str
    recordId: Optional[str] = None
    photos: List[SubmitPhoto] = Field(default_factory=list)


@dataclass(frozen=True)
class CameraGroupIndex:
    group_names: Dict[str, str]
    cam_to_group: Dict[str, str]

    def canonical_group_id(self, value: Optional[str]) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        if text in self.group_names:
            return text
        for group_id, name in self.group_names.items():
            if text == name:
                return group_id
        return None

    def group_for_camera_code(self, value: Optional[str]) -> Optional[str]:
        camera_code = str(value or "").strip()
        direct = self.cam_to_group.get(camera_code)
        if direct:
            return direct
        if camera_code.startswith("DKV3_"):
            mobile_suffix = camera_code[len("DKV3_"):]
            matches = [
                group_id for group_id in self.group_names
                if mobile_suffix == group_id or mobile_suffix.startswith(f"{group_id}_")
            ]
            if matches:
                return max(matches, key=len)
        return None


def iter_camera_role_configs(cam_cfg: Any) -> Iterable[Any]:
    if isinstance(cam_cfg, list):
        yield from cam_cfg
        return
    if isinstance(cam_cfg, dict) and "cam_ids" in cam_cfg:
        for cam_id in cam_cfg.get("cam_ids") or []:
            yield {"cam_id": cam_id}
        return
    yield cam_cfg


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


def format_iso_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_plate(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def safe_folder_name(value: Any, *, fallback: str = "UNKNOWN") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value or "").strip().upper())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def image_bytes_look_valid(data: bytes) -> bool:
    if not data:
        return False
    return (
        data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"\x89PNG\r\n\x1a\n")
        or data.startswith(b"GIF87a")
        or data.startswith(b"GIF89a")
        or (data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP")
    )


def decode_submit_image(value: str) -> bytes:
    text = str(value or "").strip()
    if not text:
        raise ValueError("photo image is required")
    if text.startswith("data:"):
        header, sep, payload = text.partition(",")
        if not sep or ";base64" not in header.lower():
            raise ValueError("photo image data URL must be base64")
        text = payload.strip()
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("photo image is not valid base64") from exc
    if not image_bytes_look_valid(data):
        raise ValueError("photo image is empty or unsupported")
    return data


def storage_date_from_timestamp(value: Any, storage_tz: ZoneInfo) -> Tuple[str, str]:
    dt = parse_iso_datetime(value)
    if dt is None:
        raise ValueError("timestamp must be an ISO datetime")
    local_dt = dt.astimezone(storage_tz)
    return local_dt.date().isoformat(), format_iso_datetime(dt)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def validate_submit_token(submit_token: str, x_api_token: str) -> None:
    if not submit_token:
        raise HTTPException(status_code=401, detail="submit token is not configured")
    if x_api_token != submit_token:
        raise HTTPException(status_code=401, detail="invalid token")


def load_camera_group_index_from_config(raw: Dict[str, Any]) -> CameraGroupIndex:
    raw_groups = raw.get("camera_groups") or {}
    if not isinstance(raw_groups, dict):
        raise ValueError("config camera_groups must be a mapping")

    group_names: Dict[str, str] = {}
    cam_to_group: Dict[str, str] = {}
    for group_id, group_cfg in raw_groups.items():
        group_id_text = str(group_id).strip()
        if not group_id_text:
            raise ValueError("camera group id must not be empty")
        if not isinstance(group_cfg, dict):
            raise ValueError(f"camera group {group_id!r} must be a mapping")
        group_names[group_id_text] = str(group_cfg.get("name") or "").strip()

        cameras_cfg = group_cfg.get("cameras") or {}
        if not isinstance(cameras_cfg, dict):
            raise ValueError(f"camera group {group_id!r} cameras must be a mapping")
        for cam_cfg in cameras_cfg.values():
            for camera_cfg in iter_camera_role_configs(cam_cfg):
                if isinstance(camera_cfg, dict):
                    cam_id = str(camera_cfg.get("cam_id") or "").strip()
                elif camera_cfg is not None:
                    cam_id = str(camera_cfg).strip()
                else:
                    cam_id = ""
                if not cam_id:
                    continue
                owner = cam_to_group.get(cam_id)
                if owner and owner != group_id_text:
                    raise ValueError(f"camera {cam_id!r} is configured in multiple groups: {owner!r} and {group_id_text!r}")
                cam_to_group[cam_id] = group_id_text

    return CameraGroupIndex(group_names=group_names, cam_to_group=cam_to_group)


def load_camera_group_index(path: Optional[str]) -> CameraGroupIndex:
    if not path:
        return CameraGroupIndex(group_names={}, cam_to_group={})

    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"camera group config not found: {path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return load_camera_group_index_from_config(raw)


def camera_group_admin_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Groups</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; max-width: 1100px; color: #182026; }
    input, button { font: inherit; }
    input { padding: 6px 8px; border: 1px solid #bcc6d0; border-radius: 4px; }
    button { padding: 6px 10px; border: 1px solid #9aa8b5; border-radius: 4px; background: #f5f7f9; cursor: pointer; }
    button.primary { background: #1366d6; color: white; border-color: #1366d6; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
    .group { border: 1px solid #d6dde5; border-radius: 6px; padding: 12px; margin: 14px 0; }
    .role { margin: 10px 0 10px 24px; padding: 10px; background: #f8fafc; border-radius: 6px; }
    .cam { margin-left: 24px; }
    .status { margin: 12px 0; min-height: 24px; }
    .error { color: #b00020; }
    .ok { color: #126b32; }
  </style>
</head>
<body>
  <h1>Camera Groups</h1>
  <div class="row">
    <label>Bearer token <input id="token" type="password" placeholder="SYNC_TOKEN"></label>
    <button id="load">Load</button>
    <button id="save" class="primary">Save</button>
    <button id="addGroup">Add location</button>
  </div>
  <div class="row">
    <label>Relay base URL <input id="relay" size="54"></label>
    <label>Default stream profile <input id="profile" size="12"></label>
  </div>
  <div id="status" class="status"></div>
  <div id="groups"></div>
<script>
let config = {relay_base_url: "", default_stream_profile: "main", camera_groups: {}};
let version = null;
const $ = id => document.getElementById(id);
function headers() {
  const token = $("token").value.trim();
  return token ? {"Authorization": "Bearer " + token, "Content-Type": "application/json"} : {"Content-Type": "application/json"};
}
function status(text, cls) {
  $("status").textContent = text;
  $("status").className = "status " + (cls || "");
}
function asArray(roleCfg) {
  if (!roleCfg) return [];
  if (Array.isArray(roleCfg)) return roleCfg.map(x => typeof x === "string" ? x : (x.cam_id || "")).filter(Boolean);
  if (roleCfg.cam_ids) return roleCfg.cam_ids.slice();
  if (roleCfg.cam_id) return [roleCfg.cam_id];
  return [];
}
function render() {
  $("relay").value = config.relay_base_url || "";
  $("profile").value = config.default_stream_profile || "main";
  const root = $("groups");
  root.innerHTML = "";
  Object.entries(config.camera_groups || {}).forEach(([gid, group]) => {
    const box = document.createElement("div");
    box.className = "group";
    box.innerHTML = `<div class="row">
      <label>Location id <input data-kind="gid" value="${gid}"></label>
      <label>Name <input data-kind="gname" value="${group.name || gid}"></label>
      <button data-action="addRole">Add role</button>
      <button data-action="removeGroup">Remove location</button>
    </div><div data-roles></div>`;
    const roles = box.querySelector("[data-roles]");
    Object.entries(group.cameras || {}).forEach(([role, roleCfg]) => {
      const roleBox = document.createElement("div");
      roleBox.className = "role";
      roleBox.innerHTML = `<div class="row">
        <label>Role <input data-kind="role" value="${role}"></label>
        <button data-action="addCam">Add camera</button>
        <button data-action="removeRole">Remove role</button>
      </div><div data-cams></div>`;
      const cams = roleBox.querySelector("[data-cams]");
      asArray(roleCfg).forEach(cam => {
        const row = document.createElement("div");
        row.className = "row cam";
        row.innerHTML = `<label>cam_id <input data-kind="cam" value="${cam}"></label><button data-action="removeCam">Remove</button>`;
        cams.appendChild(row);
      });
      roles.appendChild(roleBox);
    });
    root.appendChild(box);
  });
}
function collect() {
  const next = {relay_base_url: $("relay").value.trim(), default_stream_profile: $("profile").value.trim() || "main", camera_groups: {}};
  document.querySelectorAll(".group").forEach(groupEl => {
    const gid = groupEl.querySelector('[data-kind="gid"]').value.trim();
    if (!gid) return;
    const gname = groupEl.querySelector('[data-kind="gname"]').value.trim() || gid;
    const group = {name: gname, cameras: {}};
    groupEl.querySelectorAll(".role").forEach(roleEl => {
      const role = roleEl.querySelector('[data-kind="role"]').value.trim();
      if (!role) return;
      const ids = Array.from(roleEl.querySelectorAll('[data-kind="cam"]')).map(i => i.value.trim()).filter(Boolean);
      if (ids.length) group.cameras[role] = {cam_ids: ids};
    });
    next.camera_groups[gid] = group;
  });
  return next;
}
async function loadConfig() {
  status("Loading...");
  const res = await fetch("/api/admin/camera-groups", {headers: headers()});
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  config = data.config;
  version = data.version;
  render();
  status("Loaded version " + version, "ok");
}
async function saveConfig() {
  config = collect();
  status("Saving...");
  const res = await fetch("/api/admin/camera-groups", {method: "PUT", headers: headers(), body: JSON.stringify(config)});
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  config = data.config;
  version = data.version;
  render();
  status("Saved version " + version, "ok");
}
document.addEventListener("click", e => {
  const action = e.target.dataset.action;
  if (!action) return;
  if (action === "removeGroup") e.target.closest(".group").remove();
  if (action === "addRole") {
    config = collect();
    const gid = e.target.closest(".group").querySelector('[data-kind="gid"]').value.trim();
    config.camera_groups[gid].cameras["new_role"] = {cam_ids: [""]};
    render();
  }
  if (action === "removeRole") e.target.closest(".role").remove();
  if (action === "addCam") {
    const row = document.createElement("div");
    row.className = "row cam";
    row.innerHTML = `<label>cam_id <input data-kind="cam" value=""></label><button data-action="removeCam">Remove</button>`;
    e.target.closest(".role").querySelector("[data-cams]").appendChild(row);
  }
  if (action === "removeCam") e.target.closest(".cam").remove();
});
$("addGroup").onclick = () => { config = collect(); config.camera_groups["new_location"] = {name: "new_location", cameras: {front: {cam_ids: [""]}}}; render(); };
$("load").onclick = () => loadConfig().catch(err => status(err.message, "error"));
$("save").onclick = () => saveConfig().catch(err => status(err.message, "error"));
loadConfig().catch(err => status(err.message, "error"));
</script>
</body>
</html>"""


def camera_group_ids_for_metadata(metadata: Dict[str, Any], group_index: CameraGroupIndex) -> List[str]:
    groups: Set[str] = set()
    explicit_group = str(metadata.get("group_id") or "").strip()
    if explicit_group:
        groups.add(explicit_group)

    for key in ("capture_cam_id", "trigger_cam_id", "device_id"):
        cam_id = str(metadata.get(key) or "").strip()
        group_id = group_index.group_for_camera_code(cam_id)
        if group_id:
            groups.add(group_id)

    return sorted(groups)


def save_mobile_submit(
    root: Path,
    payload: SubmitPayload,
    *,
    storage_timezone: str = DEFAULT_STORAGE_TIMEZONE,
) -> Dict[str, Any]:
   
    plate = normalize_plate(payload.plate)
    if not plate:
        raise ValueError("plate is required")
    device_id = safe_folder_name(payload.deviceId, fallback="")
    if not device_id:
        raise ValueError("deviceId is required")
    if not payload.photos:
        raise ValueError("photos are required")
    try:
        storage_tz = ZoneInfo(storage_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"invalid storage timezone: {storage_timezone}") from exc

    storage_date, produced_at = storage_date_from_timestamp(payload.timestamp, storage_tz)
    saved_at = format_iso_datetime(datetime.now(timezone.utc))
    plate_dir = root / storage_date / plate
    plate_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[Dict[str, Any]] = []
    for photo in payload.photos:
        role = safe_folder_name(photo.type, fallback="PHOTO")
        stem = f"{role}_{device_id}"
        image_path = plate_dir / f"{stem}.jpg"
        metadata_path = plate_dir / f"{stem}.json"
        image_data = decode_submit_image(photo.image)
        tmp_image_path = image_path.with_suffix(image_path.suffix + ".tmp")
        tmp_image_path.write_bytes(image_data)
        os.replace(tmp_image_path, image_path)

        metadata = {
            "plate": plate,
            "storage_date": storage_date,
            "capture_role": role.lower(),
            "capture_cam_id": payload.deviceId,
            "device_id": payload.deviceId,
            "plate_source": payload.plateSource,
            "record_id": payload.recordId,
            "produced_at": produced_at,
            "saved_at": saved_at,
            "source_method": "mobile_submit",
            "file": str(image_path),
            "confidence": 0.99,
            "status": 200,
        }
        plate_image_path = None
        if photo.plate_image:
            plate_image_path = plate_dir / f"{stem}_plate.jpg"
            plate_image_data = decode_submit_image(photo.plate_image)
            tmp_plate_image_path = plate_image_path.with_suffix(plate_image_path.suffix + ".tmp")
            tmp_plate_image_path.write_bytes(plate_image_data)
            os.replace(tmp_plate_image_path, plate_image_path)
            metadata["plate_image_file"] = str(plate_image_path)
        atomic_write_json(metadata_path, metadata)
        saved_file = {
            "type": role,
            "image_path": str(image_path),
            "metadata_path": str(metadata_path),
        }
        if plate_image_path is not None:
            saved_file["plate_image_path"] = str(plate_image_path)
        saved_files.append(saved_file)

    return {
        "ok": True,
        "plate": plate,
        "storage_date": storage_date,
        "saved_at": saved_at,
        "count": len(saved_files),
        "files": saved_files,
    }


def encode_cursor(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(value: str) -> Dict[str, Any]:
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except Exception as exc:
        raise ValueError("invalid cursor") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid cursor")
    return payload


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_relative_path(root: Path, relative: str) -> Path:
    relative = str(relative or "").strip().lstrip("/")
    if not relative:
        raise ValueError("relative path is required")
    path = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("path escapes output root") from exc
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path


def file_entry(root: Path, path: Path, *, base_url: str) -> Dict[str, Any]:
    stat = path.stat()
    rel = relative_path(root, path)
    return {
        "relative_path": rel,
        "size": stat.st_size,
        "mtime": format_iso_datetime(datetime.fromtimestamp(stat.st_mtime, timezone.utc)),
        "download_url": f"{base_url.rstrip('/')}/sync/files/{quote(rel)}",
    }


def path_from_metadata_value(root: Path, value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            return None
        return path
    return root / text


def capture_files(root: Path, metadata_path: Path, metadata: Dict[str, Any], *, base_url: str) -> List[Dict[str, Any]]:
    candidates: List[Path] = [metadata_path]
    image_path = path_from_metadata_value(root, metadata.get("file")) or metadata_path.with_suffix(".jpg")
    candidates.append(image_path)

    plate_image_path = path_from_metadata_value(root, metadata.get("plate_image_file"))
    if plate_image_path is None:
        plate_candidate = metadata_path.with_name(f"{metadata_path.stem}_plate.jpg")
        if plate_candidate.is_file():
            plate_image_path = plate_candidate
    if plate_image_path is not None:
        candidates.append(plate_image_path)

    files: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        files.append(file_entry(root, resolved, base_url=base_url))
    return files


def iter_capture_items(root: Path, *, base_url: str, group_index: CameraGroupIndex) -> Iterable[Dict[str, Any]]:
    for metadata_path in sorted(root.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/*/*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Skipping unreadable capture metadata %s: %s", metadata_path, exc)
            continue
        if not isinstance(metadata, dict):
            continue
        saved_at = parse_iso_datetime(metadata.get("saved_at"))
        if saved_at is None:
            continue

        rel_metadata_path = relative_path(root, metadata_path)
        camera_group_ids = camera_group_ids_for_metadata(metadata, group_index)
        item = {
            "type": "capture",
            "sync_timestamp": format_iso_datetime(saved_at),
            "saved_at": format_iso_datetime(saved_at),
            "relative_metadata_path": rel_metadata_path,
            "plate": metadata.get("plate") or metadata_path.parent.name,
            "storage_date": metadata.get("storage_date") or metadata_path.parent.parent.name,
            "produced_at": metadata.get("produced_at"),
            "group_id": metadata.get("group_id"),
            "camera_group_ids": camera_group_ids,
            "capture_role": metadata.get("capture_role"),
            "capture_cam_id": metadata.get("capture_cam_id"),
            "trigger_cam_id": metadata.get("trigger_cam_id"),
            "confidence": metadata.get("confidence"),
            "tracking_object_id": metadata.get("tracking_object_id"),
            "review_status": metadata.get("review_status"),
            "review_reason": metadata.get("review_reason"),
            "review_group_id": metadata.get("review_group_id"),
            "similar_plates": metadata.get("similar_plates"),
            "canonical_plate_suggestion": metadata.get("canonical_plate_suggestion"),
            "files": capture_files(root, metadata_path, metadata, base_url=base_url),
        }
        yield item


def iter_review_items(root: Path, *, base_url: str, group_index: CameraGroupIndex) -> Iterable[Dict[str, Any]]:
    for review_path in sorted((root / "_review").glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/suspicious_plates.jsonl")):
        try:
            lines = review_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logging.warning("Skipping unreadable review manifest %s: %s", review_path, exc)
            continue
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping invalid review JSONL line %s:%s", review_path, line_no)
                continue
            if not isinstance(record, dict):
                continue
            created_at = parse_iso_datetime(record.get("created_at"))
            if created_at is None:
                continue
            camera_group_ids: Set[str] = set()
            candidates = record.get("candidates") or []
            if isinstance(candidates, list):
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        camera_group_ids.update(camera_group_ids_for_metadata(candidate, group_index))
            rel_path = f"{relative_path(root, review_path)}#{line_no}"
            yield {
                "type": "review",
                "sync_timestamp": format_iso_datetime(created_at),
                "created_at": format_iso_datetime(created_at),
                "relative_metadata_path": rel_path,
                "storage_date": record.get("storage_date") or review_path.parent.name,
                "camera_group_ids": sorted(camera_group_ids),
                "review_group_id": record.get("review_group_id"),
                "review_status": record.get("review_status"),
                "review_reason": record.get("review_reason"),
                "canonical_plate_suggestion": record.get("canonical_plate_suggestion"),
                "plates": record.get("plates"),
                "record": record,
                "files": [file_entry(root, review_path, base_url=base_url)],
            }


def manifest_items(root: Path, *, base_url: str, group_index: CameraGroupIndex) -> List[Dict[str, Any]]:
    items = list(iter_capture_items(root, base_url=base_url, group_index=group_index))
    items.extend(iter_review_items(root, base_url=base_url, group_index=group_index))
    items.sort(key=lambda item: (item["sync_timestamp"], item["relative_metadata_path"]))
    return items


def build_changes_manifest(
    root: Path,
    *,
    since: Optional[str],
    cursor: Optional[str],
    limit: int,
    base_url: str,
    location: Optional[str] = None,
    camera_group: Optional[str] = None,
    group_index: Optional[CameraGroupIndex] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    limit = max(1, min(MAX_LIMIT, int(limit or DEFAULT_LIMIT)))
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    group_index = group_index or CameraGroupIndex(group_names={}, cam_to_group={})
    requested_location = location or camera_group

    if cursor:
        try:
            cursor_payload = decode_cursor(cursor)
            upper_bound = parse_iso_datetime(cursor_payload.get("upper_bound"))
            last_timestamp = str(cursor_payload.get("last_timestamp") or "")
            last_relative_path = str(cursor_payload.get("last_relative_path") or "")
            since_dt = parse_iso_datetime(cursor_payload.get("since"))
            selected_group = str(cursor_payload.get("camera_group") or "") or None
        except ValueError as exc:
            raise ValueError("invalid cursor") from exc
        if upper_bound is None or since_dt is None:
            raise ValueError("invalid cursor")
        requested_group = requested_location
        if requested_group:
            requested_group = group_index.canonical_group_id(requested_group) or requested_group
        if requested_group and requested_group != selected_group:
            raise ValueError("cursor belongs to a different location")
    else:
        since_dt = parse_iso_datetime(since)
        if since_dt is None:
            raise ValueError("since must be an ISO datetime")
        upper_bound = now
        last_timestamp = ""
        last_relative_path = ""
        selected_group = requested_location

    if selected_group:
        canonical_group = group_index.canonical_group_id(selected_group)
        if canonical_group:
            selected_group = canonical_group
        elif group_index.group_names:
            raise ValueError(f"unknown location: {selected_group}")

    selected: List[Dict[str, Any]] = []
    for item in manifest_items(root, base_url=base_url, group_index=group_index):
        item_dt = parse_iso_datetime(item.get("sync_timestamp"))
        if item_dt is None:
            continue
        if item_dt <= since_dt or item_dt > upper_bound:
            continue
        if selected_group and selected_group not in set(item.get("camera_group_ids") or []):
            continue
        item_key = (item["sync_timestamp"], item["relative_metadata_path"])
        if cursor and item_key <= (last_timestamp, last_relative_path):
            continue
        selected.append(item)

    page = selected[:limit]
    has_more = len(selected) > limit
    if has_more and page:
        last = page[-1]
        next_cursor = encode_cursor(
            {
                "since": format_iso_datetime(since_dt),
                "upper_bound": format_iso_datetime(upper_bound),
                "last_timestamp": last["sync_timestamp"],
                "last_relative_path": last["relative_metadata_path"],
                "camera_group": selected_group,
            }
        )
        next_since = None
    else:
        next_cursor = None
        next_since = format_iso_datetime(upper_bound)

    return {
        "server_time": format_iso_datetime(now),
        "since": format_iso_datetime(since_dt),
        "upper_bound": format_iso_datetime(upper_bound),
        "location": selected_group,
        "camera_group": selected_group,
        "next_cursor": next_cursor,
        "next_since": next_since,
        "has_more": has_more,
        "limit": limit,
        "items": page,
    }


def create_app(args: argparse.Namespace) -> FastAPI:
    output_root = Path(args.output_root)
    sync_token = str(args.sync_token or "")
    submit_token = str(getattr(args, "submit_token", "") or "")
    storage_timezone = str(getattr(args, "storage_timezone", DEFAULT_STORAGE_TIMEZONE) or DEFAULT_STORAGE_TIMEZONE)
    config_provider = getattr(args, "camera_config_provider", None) or InspectionCameraConfigProvider()
    reload_interval_sec = max(0.1, float(getattr(args, "config_reload_interval_sec", DEFAULT_CONFIG_RELOAD_INTERVAL_SEC)))
    app = FastAPI(title="Vehicle Capture Sync API")
    app.state.config_provider = config_provider
    app.state.config_version = None
    app.state.last_config_check = 0.0
    try:
        state = config_provider.load(force=True)
        app.state.group_index = load_camera_group_index_from_config(state.config)
        app.state.config_version = state.version
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    def reload_group_index_if_needed(*, force: bool = False) -> None:
        now = datetime.now(timezone.utc).timestamp()
        if not force and now - float(app.state.last_config_check or 0.0) < reload_interval_sec:
            return
        app.state.last_config_check = now
        try:
            state = config_provider.load()
            if not force and state.version == app.state.config_version:
                return
            group_index = load_camera_group_index_from_config(state.config)
        except Exception as exc:
            logging.warning("Failed to reload camera config for sync API: %s", exc)
            return
        app.state.group_index = group_index
        app.state.config_version = state.version
        logging.info("Reloaded sync camera config version=%s cameras=%d", state.version, len(group_index.cam_to_group))

    async def require_auth(authorization: str = Header("")) -> None:
        if not sync_token:
            return
        expected = f"Bearer {sync_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid token")

    async def require_submit_auth(x_api_token: str = Header("")) -> None:
        validate_submit_token(submit_token, x_api_token)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        reload_group_index_if_needed()
        return {
            "ok": output_root.is_dir(),
            "output_root": str(output_root),
            "config_version": app.state.config_version,
        }

    @app.get("/api/admin/camera-groups")
    async def get_camera_groups(_auth: None = Depends(require_auth)) -> Dict[str, Any]:
        state = config_provider.load()
        return {
            "version": state.version,
            "updated_at": state.updated_at,
            "config": state.config,
        }

    @app.get("/sync/changes")
    async def sync_changes(
        request: Request,
        since: Optional[str] = Query(None),
        cursor: Optional[str] = Query(None),
        location: Optional[str] = Query(None),
        camera_group: Optional[str] = Query(None),
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        _auth: None = Depends(require_auth),
    ) -> Dict[str, Any]:
        reload_group_index_if_needed()
        base_url = str(request.base_url).rstrip("/")
        try:
            return build_changes_manifest(
                output_root,
                since=since,
                cursor=cursor,
                limit=limit,
                base_url=base_url,
                location=location,
                camera_group=camera_group,
                group_index=app.state.group_index,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/sync/files/{relative_path:path}", name="sync_file")
    async def sync_file(relative_path: str, _auth: None = Depends(require_auth)) -> FileResponse:
        try:
            path = resolve_relative_path(output_root, relative_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="file not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return FileResponse(path)

    @app.post("/api/submit")
    async def submit_capture(
        payload: SubmitPayload = Body(...),
        _auth: None = Depends(require_submit_auth),
    ) -> Dict[str, Any]:
        try:
            return save_mobile_submit(output_root, payload, storage_timezone=storage_timezone)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve timestamp-based sync manifests for vehicle captures.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--output-root", default=os.environ.get("SYNC_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sync-token", default=os.environ.get("SYNC_TOKEN", ""))
    parser.add_argument("--submit-token", default=os.environ.get("SUBMIT_TOKEN", ""))
    parser.add_argument("--storage-timezone", default=os.environ.get("STORAGE_TIMEZONE", DEFAULT_STORAGE_TIMEZONE))
    parser.add_argument("--config-reload-interval-sec", type=float, default=float(os.environ.get("CAMERA_CONFIG_RELOAD_INTERVAL_SEC", DEFAULT_CONFIG_RELOAD_INTERVAL_SEC)))
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    app = create_app(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=str(args.log_level).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
