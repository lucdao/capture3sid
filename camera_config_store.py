import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


DEFAULT_CAMERA_API_BASE_URL = "http://10.11.12.12:8101"
DEFAULT_CAMERA_API_PATH = "/api/inspections/stations/cameras"
DEFAULT_CAMERA_API_TIMEOUT_SEC = 5.0
DEFAULT_CAMERA_API_PAGE_SIZE = 20
DEFAULT_CONFIG_RELOAD_INTERVAL_SEC = 30.0

# The inspection API now emits ``camera_frontback``. Keep the legacy
# ``camera_inout`` values as compatibility aliases for older deployments.
CAMERA_DIRECTION_TO_ROLES = {
    "front": ("front",),
    "back": ("back",),
    "frontback": ("front", "back"),
    "in": ("front",),
    "out": ("back",),
    "inout": ("back",),
}


@dataclass(frozen=True)
class ConfigState:
    version: str
    updated_at: str
    config: Dict[str, Any]


def format_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def inspection_camera_api_payload_to_config(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Any] = {}
    seen_camera_codes: Dict[str, str] = {}
    seen_iapp_ids: Dict[str, str] = {}

    for payload in pages:
        stations = payload.get("stations")
        if not isinstance(stations, list):
            raise ValueError("inspection camera API response stations must be a list")
        for station in stations:
            if not isinstance(station, dict):
                raise ValueError("inspection camera API station must be an object")
            station_code = str(station.get("station_code") or "").strip()
            if not station_code:
                raise ValueError("inspection camera API station_code must not be empty")
            if station_code in groups:
                raise ValueError(f"duplicate station_code {station_code!r}")

            role_codes: Dict[str, List[str]] = {"front": [], "back": []}
            raw_iapp_ids = station.get("iapp_ids")
            if raw_iapp_ids is None:
                raw_iapp_ids = []
            elif not isinstance(raw_iapp_ids, list):
                raise ValueError(f"inspection camera API station {station_code!r} iapp_ids must be a list")
            iapp_ids: List[str] = []
            for raw_iapp_id in raw_iapp_ids:
                iapp_id = str(raw_iapp_id or "").strip()
                if not iapp_id:
                    raise ValueError(f"inspection camera API station {station_code!r} contains an empty iapp_id")
                owner = seen_iapp_ids.get(iapp_id)
                if owner is not None and owner != station_code:
                    raise ValueError(
                        f"duplicate iapp_id {iapp_id!r} in stations {owner!r} and {station_code!r}"
                    )
                seen_iapp_ids[iapp_id] = station_code
                if iapp_id not in iapp_ids:
                    iapp_ids.append(iapp_id)
            cameras = station.get("cameras")
            if not isinstance(cameras, list):
                    raise ValueError(f"inspection camera API station {station_code!r} cameras must be a list")
            for camera in cameras:
                if not isinstance(camera, dict):
                    raise ValueError(f"inspection camera API station {station_code!r} camera must be an object")
                camera_code = str(camera.get("camera_code") or "").strip()
                if not camera_code:
                    raise ValueError(f"inspection camera API station {station_code!r} has a camera without camera_code")
                owner = seen_camera_codes.get(camera_code)
                if owner is not None:
                    raise ValueError(
                        f"duplicate camera_code {camera_code!r} in inspection camera API stations {owner!r} and {station_code!r}"
                    )
                direction_field = "camera_frontback" if camera.get("camera_frontback") is not None else "camera_inout"
                raw_direction = camera.get(direction_field)
                direction = str(raw_direction or "").strip().lower()
                if not direction:
                    logging.warning(
                        "Inspection camera API camera %r has no camera_inout; skipping unclassified camera",
                        camera_code,
                    )
                    continue
                roles = CAMERA_DIRECTION_TO_ROLES.get(direction)
                if roles is None:
                    raise ValueError(
                        f"inspection camera API camera {camera_code!r} has unknown {direction_field} {direction!r}"
                    )
                seen_camera_codes[camera_code] = station_code
                for role in roles:
                    role_codes[role].append(camera_code)

            camera_roles: Dict[str, Any] = {}
            for role, codes in role_codes.items():
                if codes:
                    camera_roles[role] = {"cam_ids": sorted(codes)}
            if iapp_ids:
                camera_roles["mobile"] = {"cam_ids": sorted(iapp_ids)}
            groups[station_code] = {
                "name": str(station.get("station_name") or station_code).strip() or station_code,
                "cameras": camera_roles,
            }

    if not groups:
        raise ValueError("inspection camera API returned no stations")
    return {"camera_groups": dict(sorted(groups.items()))}


# Compatibility alias for existing imports.
camera_api_payload_to_config = inspection_camera_api_payload_to_config


class InspectionCameraConfigProvider:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        authorization: Optional[str] = None,
        api_path: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        page_size: Optional[int] = None,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = str(base_url or os.environ.get("CAMERA_API_BASE_URL") or DEFAULT_CAMERA_API_BASE_URL).rstrip("/")
        self.authorization = str(authorization if authorization is not None else os.environ.get("CAMERA_API_AUTHORIZATION", ""))
        self.api_path = str(api_path or os.environ.get("CAMERA_API_PATH") or DEFAULT_CAMERA_API_PATH)
        self.timeout_sec = float(timeout_sec or os.environ.get("CAMERA_API_TIMEOUT_SEC", DEFAULT_CAMERA_API_TIMEOUT_SEC))
        self.page_size = int(page_size or os.environ.get("CAMERA_API_PAGE_SIZE", DEFAULT_CAMERA_API_PAGE_SIZE))
        self.session = session or requests.Session()
        self._state: Optional[ConfigState] = None

    def _fetch(self) -> ConfigState:
        headers = {"accept": "application/json"}
        if self.authorization:
            headers["Authorization"] = self.authorization
        pages: List[Dict[str, Any]] = []
        page = 1
        total_pages = 1
        while page <= total_pages:
            response = self.session.get(
                f"{self.base_url}{self.api_path}",
                params={"page": page, "limit": self.page_size},
                headers=headers,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("inspection camera API response must be an object")
            response_page = int(payload.get("page", page))
            if response_page != page:
                raise ValueError(f"inspection camera API returned page {response_page}, expected {page}")
            total_pages = int(payload.get("total_pages", 1))
            if total_pages < 1:
                raise ValueError("inspection camera API total_pages must be positive")
            pages.append(payload)
            page += 1

        config = inspection_camera_api_payload_to_config(pages)
        canonical = json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        version = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return ConfigState(version=version, updated_at=format_utc_now(), config=config)

    def load(self, *, force: bool = False) -> ConfigState:
        try:
            state = self._fetch()
        except Exception:
            if self._state is None:
                raise
            logging.warning("Inspection camera API refresh failed; retaining config version=%s", self._state.version, exc_info=True)
            return self._state
        if force or self._state is None or state.version != self._state.version:
            self._state = state
        return self._state

    def current_version(self) -> Optional[str]:
        return self._state.version if self._state is not None else None


# Compatibility alias for existing service wiring and external callers.
CameraConfigProvider = InspectionCameraConfigProvider
