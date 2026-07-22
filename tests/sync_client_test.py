#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests


DEFAULT_STATE_FILE = ".capture_sync_state.json"
DEFAULT_SINCE = "1970-01-01T00:00:00Z"


def auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def load_state(path: Path) -> Dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def write_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def safe_output_path(output_root: Path, relative_path: str) -> Path:
    relative_path = str(relative_path or "").strip().lstrip("/")
    if not relative_path:
        raise ValueError("empty relative path")
    path = (output_root / relative_path).resolve()
    root = output_root.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"server returned unsafe relative path: {relative_path}") from exc
    return path


def iter_files(items: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    seen: set[str] = set()
    for item in items:
        files = item.get("files")
        if not isinstance(files, list):
            continue
        for file_info in files:
            if not isinstance(file_info, dict):
                continue
            relative_path = str(file_info.get("relative_path") or "")
            download_url = str(file_info.get("download_url") or "")
            if not relative_path or not download_url or relative_path in seen:
                continue
            seen.add(relative_path)
            yield file_info


def download_file(
    session: requests.Session,
    *,
    file_info: Dict[str, Any],
    output_root: Path,
    token: str,
    timeout_sec: float,
    dry_run: bool,
) -> bool:
    relative_path = str(file_info["relative_path"])
    output_path = safe_output_path(output_root, relative_path)
    expected_size = file_info.get("size")
    if output_path.is_file() and expected_size is not None:
        try:
            if output_path.stat().st_size == int(expected_size):
                print(f"skip existing {relative_path}")
                return False
        except (OSError, TypeError, ValueError):
            pass

    if dry_run:
        print(f"would download {relative_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp", dir=str(output_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with session.get(
            str(file_info["download_url"]),
            headers=auth_headers(token),
            timeout=timeout_sec,
            stream=True,
        ) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    print(f"downloaded {relative_path}")
    return True


def sync_once(args: argparse.Namespace) -> Dict[str, int]:
    output_root = Path(args.output_root)
    state_path = Path(args.state_file)
    state = load_state(state_path)
    mode = str(getattr(args, "mode", "capture") or "capture")
    requested_location = str(args.location or args.camera_group or "")
    stream_key = f"{mode}:{requested_location}"
    streams = dict(state.get("streams")) if isinstance(state.get("streams"), dict) else {}
    legacy_since = state.get("since")
    legacy_mode = str(state.get("mode") or "capture")
    legacy_location = str(state.get("location") or state.get("camera_group") or "")
    legacy_key = f"{legacy_mode}:{legacy_location}"
    if legacy_since and legacy_key not in streams:
        streams[legacy_key] = {
            "since": legacy_since,
            "last_sync_unix": state.get("last_sync_unix"),
            "base_url": state.get("base_url"),
            "location": legacy_location,
            "camera_group": legacy_location,
            "mode": legacy_mode,
        }
    stream_state = streams.get(stream_key) if isinstance(streams.get(stream_key), dict) else {}
    state_since = stream_state.get("since")
    since = args.since or state_since or DEFAULT_SINCE
    cursor: Optional[str] = None
    pages = 0
    items = 0
    downloads = 0

    base_url = str(args.base_url).rstrip("/")
    session = requests.Session()

    while True:
        params: Dict[str, Any] = {"limit": args.limit, "mode": mode}
        if args.camera_group:
            params["camera_group"] = args.camera_group
        if args.location:
            params["location"] = args.location
        if cursor:
            params["cursor"] = cursor
        else:
            params["since"] = since

        response = session.get(
            f"{base_url}/sync/changes",
            params=params,
            headers=auth_headers(args.token),
            timeout=args.timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        page_items = payload.get("items") or []
        if not isinstance(page_items, list):
            raise RuntimeError("sync response items must be a list")

        pages += 1
        items += len(page_items)
        print(
            "page=%d items=%d has_more=%s upper_bound=%s"
            % (pages, len(page_items), payload.get("has_more"), payload.get("upper_bound"))
        )

        for file_info in iter_files(page_items):
            if download_file(
                session,
                file_info=file_info,
                output_root=output_root,
                token=args.token,
                timeout_sec=args.timeout_sec,
                dry_run=args.dry_run,
            ):
                downloads += 1

        cursor = payload.get("next_cursor")
        if not payload.get("has_more"):
            next_since = payload.get("next_since")
            if not args.dry_run and next_since:
                next_streams = dict(streams)
                next_streams[stream_key] = {
                    "since": next_since,
                    "last_sync_unix": time.time(),
                    "base_url": base_url,
                    "location": requested_location,
                    "camera_group": requested_location,
                    "mode": mode,
                }
                write_state(
                    state_path,
                    {
                        "version": 2,
                        "streams": next_streams,
                        "last_sync_unix": time.time(),
                        "base_url": base_url,
                    },
                )
            return {"pages": pages, "items": items, "downloads": downloads}

        if not cursor:
            raise RuntimeError("server returned has_more=true without next_cursor")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test client for the capture and document sync API.")
     
    parser.add_argument("--base-url",default="https://imagesync.ivistatech.vn")
    parser.add_argument("--token", default=os.environ.get("SYNC_TOKEN", ""))
    parser.add_argument("--output-root", default="synced_vehicle_captures")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    parser.add_argument("--since", help="Override saved_at cursor, for example 2026-07-02T00:00:00Z")
    parser.add_argument("--mode", choices=("capture", "document"), default="capture")
    parser.add_argument("--location", help="Only sync items from one location id or location name.")
    parser.add_argument("--camera-group", help="Deprecated alias for --location.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        result = sync_once(args)
    except Exception as exc:
        print(f"sync failed: {exc}", file=sys.stderr)
        return 1
    print(
        "sync complete pages=%d items=%d downloads=%d"
        % (result["pages"], result["items"], result["downloads"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
