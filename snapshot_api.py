#!/usr/bin/env python3
import argparse
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888
DEFAULT_RELAY_BASE_URL = "rtsp://ivista-go2rtc:8554"
DEFAULT_STREAM_PROFILE = "main"
DEFAULT_RECONNECT_DELAY_SEC = 1.0
DEFAULT_FRAME_STALE_SEC = 5.0
DEFAULT_JPEG_QUALITY = 90


@dataclass(frozen=True)
class CameraStreamConfig:
    cam_id: str
    role: str
    group_id: str
    stream_url: str
    stream_profile: str


@dataclass(frozen=True)
class PixelCrop:
    left: int
    top: int
    width: int
    height: int


def coerce_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def normalized_crop_to_pixels(
    *,
    frame_width: int,
    frame_height: int,
    left_norm: Any,
    top_norm: Any,
    width_norm: Any,
    height_norm: Any,
) -> PixelCrop:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")

    left = min(1.0, max(0.0, coerce_float(left_norm, "crop_left_norm")))
    top = min(1.0, max(0.0, coerce_float(top_norm, "crop_top_norm")))
    width = coerce_float(width_norm, "crop_width_norm")
    height = coerce_float(height_norm, "crop_height_norm")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("crop width and height must be positive")

    width = min(width, 1.0 - left)
    height = min(height, 1.0 - top)

    x1 = int(round(left * frame_width))
    y1 = int(round(top * frame_height))
    x2 = int(round((left + width) * frame_width))
    y2 = int(round((top + height) * frame_height))

    x1 = min(max(0, x1), frame_width)
    y1 = min(max(0, y1), frame_height)
    x2 = min(max(0, x2), frame_width)
    y2 = min(max(0, y2), frame_height)

    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width < 2 or crop_height < 2:
        raise ValueError("crop is smaller than 2 pixels after scaling")

    return PixelCrop(left=x1, top=y1, width=crop_width, height=crop_height)


def _url_join(base_url: str, path: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    path = str(path or "").strip().lstrip("/")
    if not base:
        return path
    if path.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return path
    return f"{base}/{path}"


def _default_stream_path(cam_id: str, profile: str) -> str:
    return "/".join(
        quote(part, safe="")
        for part in (str(cam_id).strip(), str(profile).strip())
        if part
    )


def config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_camera_stream_configs(
    path: str,
    *,
    relay_base_url: Optional[str] = None,
    default_stream_profile: Optional[str] = None,
) -> Dict[str, CameraStreamConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    relay_base = str(
        relay_base_url
        or raw.get("relay_base_url")
        or os.environ.get("RELAY_BASE_URL")
        or DEFAULT_RELAY_BASE_URL
    ).strip()
    default_profile = str(
        default_stream_profile
        or raw.get("default_stream_profile")
        or os.environ.get("DEFAULT_STREAM_PROFILE")
        or DEFAULT_STREAM_PROFILE
    ).strip() or DEFAULT_STREAM_PROFILE

    raw_groups = raw.get("camera_groups") or {}
    if not isinstance(raw_groups, dict):
        raise ValueError("config camera_groups must be a mapping")

    configs: Dict[str, CameraStreamConfig] = {}
    for group_id, group_cfg in raw_groups.items():
        if not isinstance(group_cfg, dict):
            continue
        cameras = group_cfg.get("cameras") or {}
        if not isinstance(cameras, dict):
            continue
        has_ai_markers = any(
            isinstance(cam_cfg, dict) and config_bool(cam_cfg.get("ai"))
            for cam_cfg in cameras.values()
        )
        for role, cam_cfg in cameras.items():
            cam_id = ""
            stream_path = ""
            profile = default_profile
            if isinstance(cam_cfg, dict):
                if has_ai_markers and not config_bool(cam_cfg.get("ai")):
                    continue
                cam_id = str(cam_cfg.get("cam_id") or "").strip()
                stream_path = str(cam_cfg.get("stream_path") or "").strip()
                profile = str(cam_cfg.get("stream_profile") or profile).strip() or default_profile
            elif cam_cfg is not None:
                cam_id = str(cam_cfg).strip()
            if not cam_id:
                continue
            path_part = stream_path or _default_stream_path(cam_id, profile)
            configs[cam_id] = CameraStreamConfig(
                cam_id=cam_id,
                role=str(role),
                group_id=str(group_id),
                stream_url=_url_join(relay_base, path_part),
                stream_profile=profile,
            )

    if not configs:
        raise ValueError("no camera streams configured")
    return configs


class StreamReader:
    def __init__(
        self,
        config: CameraStreamConfig,
        *,
        reconnect_delay_sec: float = DEFAULT_RECONNECT_DELAY_SEC,
    ):
        self.config = config
        self.reconnect_delay_sec = max(0.1, float(reconnect_delay_sec))
        self._lock = threading.Lock()
        self._frame = None
        self._frame_monotonic: Optional[float] = None
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
        self._reconnects = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name=f"stream-{self.config.cam_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def snapshot(self):
        with self._lock:
            if self._frame is None:
                return None, None, None
            return self._frame.copy(), self._frame_monotonic, self._frame_shape

    def status(self) -> Dict[str, Any]:
        with self._lock:
            frame_age_ms = None
            if self._frame_monotonic is not None:
                frame_age_ms = (time.monotonic() - self._frame_monotonic) * 1000.0
            return {
                "cam_id": self.config.cam_id,
                "role": self.config.role,
                "group_id": self.config.group_id,
                "stream_url": self.config.stream_url,
                "stream_profile": self.config.stream_profile,
                "has_frame": self._frame is not None,
                "frame_age_ms": frame_age_ms,
                "frame_shape": self._frame_shape,
                "last_error": self._last_error,
                "reconnects": self._reconnects,
            }

    def _run(self) -> None:
        try:
            import cv2
        except Exception as exc:
            with self._lock:
                self._last_error = f"opencv import failed: {exc}"
            logging.exception("OpenCV import failed for stream reader")
            return

        while not self._stop_event.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.config.stream_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    raise RuntimeError("VideoCapture did not open")

                logging.info("Stream reader connected cam_id=%s url=%s", self.config.cam_id, self.config.stream_url)
                with self._lock:
                    self._last_error = None

                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("failed to read frame")
                    height, width = frame.shape[:2]
                    with self._lock:
                        self._frame = frame
                        self._frame_monotonic = time.monotonic()
                        self._frame_shape = (width, height)

            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                    self._reconnects += 1
                logging.warning(
                    "Stream reader failed cam_id=%s stream_url=%s: %s",
                    self.config.cam_id,
                    self.config.stream_url,
                    exc,
                )
                self._stop_event.wait(self.reconnect_delay_sec)
            finally:
                if cap is not None:
                    cap.release()


class SnapshotService:
    def __init__(
        self,
        camera_configs: Dict[str, CameraStreamConfig],
        *,
        token: str = "",
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        frame_stale_sec: float = DEFAULT_FRAME_STALE_SEC,
        reconnect_delay_sec: float = DEFAULT_RECONNECT_DELAY_SEC,
    ):
        self.token = str(token or "")
        self.jpeg_quality = min(100, max(1, int(jpeg_quality)))
        self.frame_stale_sec = max(0.1, float(frame_stale_sec))
        self.readers = {
            cam_id: StreamReader(config, reconnect_delay_sec=reconnect_delay_sec)
            for cam_id, config in camera_configs.items()
        }

    def start(self) -> None:
        for reader in self.readers.values():
            reader.start()

    def stop(self) -> None:
        for reader in self.readers.values():
            reader.stop()

    def snapshot(
        self,
        *,
        device_id: str,
        token: str,
        crop_left_norm: Any,
        crop_top_norm: Any,
        crop_width_norm: Any,
        crop_height_norm: Any,
    ) -> Tuple[bytes, Dict[str, str]]:
        if self.token and token != self.token:
            raise HTTPException(status_code=401, detail="invalid token")

        reader = self.readers.get(str(device_id))
        if reader is None:
            raise HTTPException(status_code=404, detail="unknown device_id")

        request_started = time.monotonic()
        frame, frame_time, frame_shape = reader.snapshot()
        if frame is None or frame_time is None or frame_shape is None:
            raise HTTPException(status_code=503, detail="stream has no decoded frame yet")

        frame_age_ms = (time.monotonic() - frame_time) * 1000.0
        if frame_age_ms > self.frame_stale_sec * 1000.0:
            raise HTTPException(status_code=503, detail="latest frame is stale")

        frame_width, frame_height = frame_shape
        try:
            crop = normalized_crop_to_pixels(
                frame_width=frame_width,
                frame_height=frame_height,
                left_norm=crop_left_norm,
                top_norm=crop_top_norm,
                width_norm=crop_width_norm,
                height_norm=crop_height_norm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        crop_started = time.monotonic()
        crop_frame = frame[crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]

        import cv2

        ok, encoded = cv2.imencode(
            ".jpg",
            crop_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise HTTPException(status_code=500, detail="jpeg encode failed")

        crop_duration_ms = (time.monotonic() - crop_started) * 1000.0
        total_ms = (time.monotonic() - request_started) * 1000.0
        headers = {
            "Cache-Control": "no-cache",
            "X-Frame-Age-Ms": f"{frame_age_ms:.3f}",
            "X-Crop-Duration-Ms": f"{crop_duration_ms:.3f}",
            "X-Snapshot-Total-Ms": f"{total_ms:.3f}",
            "X-Frame-Width": str(frame_width),
            "X-Frame-Height": str(frame_height),
            "X-Crop-Left": str(crop.left),
            "X-Crop-Top": str(crop.top),
            "X-Crop-Width": str(crop.width),
            "X-Crop-Height": str(crop.height),
        }
        return encoded.tobytes(), headers

    def health(self) -> Dict[str, Any]:
        statuses = {cam_id: reader.status() for cam_id, reader in self.readers.items()}
        healthy = any(status["has_frame"] for status in statuses.values())
        return {"status": "ok" if healthy else "warming", "streams": statuses}


def create_app(args: argparse.Namespace) -> FastAPI:
    configs = load_camera_stream_configs(
        args.config,
        relay_base_url=args.relay_base_url,
        default_stream_profile=args.default_stream_profile,
    )
    service = SnapshotService(
        configs,
        token=args.snapshot_token,
        jpeg_quality=args.jpeg_quality,
        frame_stale_sec=args.frame_stale_sec,
        reconnect_delay_sec=args.reconnect_delay_sec,
    )

    app = FastAPI(title="Low-latency snapshot API")
    app.state.snapshot_service = service

    @app.on_event("startup")
    def _startup() -> None:
        service.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        service.stop()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return service.health()

    @app.get("/streams")
    def streams() -> Dict[str, Any]:
        return service.health()["streams"]

    @app.get("/snapshot_rtsp")
    def snapshot_rtsp(
        device_id: str = Query(...),
        token: str = Query(""),
        crop_left_norm: float = Query(...),
        crop_top_norm: float = Query(...),
        crop_width_norm: float = Query(...),
        crop_height_norm: float = Query(...),
    ) -> Response:
        data, headers = service.snapshot(
            device_id=device_id,
            token=token,
            crop_left_norm=crop_left_norm,
            crop_top_norm=crop_top_norm,
            crop_width_norm=crop_width_norm,
            crop_height_norm=crop_height_norm,
        )
        return Response(content=data, media_type="image/jpeg", headers=headers)

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm relay streams and serve low-latency cropped snapshots.")
    parser.add_argument("--config", default="camera_groups.yaml")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--relay-base-url", default=os.environ.get("RELAY_BASE_URL", ""))
    parser.add_argument("--default-stream-profile", default=os.environ.get("DEFAULT_STREAM_PROFILE", ""))
    parser.add_argument("--snapshot-token", default=os.environ.get("SNAPSHOT_TOKEN", ""))
    parser.add_argument("--jpeg-quality", type=int, default=int(os.environ.get("JPEG_QUALITY", DEFAULT_JPEG_QUALITY)))
    parser.add_argument("--frame-stale-sec", type=float, default=float(os.environ.get("FRAME_STALE_SEC", DEFAULT_FRAME_STALE_SEC)))
    parser.add_argument("--reconnect-delay-sec", type=float, default=float(os.environ.get("RECONNECT_DELAY_SEC", DEFAULT_RECONNECT_DELAY_SEC)))
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
