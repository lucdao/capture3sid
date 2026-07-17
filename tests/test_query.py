import argparse
import os
from urllib.parse import urljoin

import requests

BASE_URL = "http://192.168.1.199:8011"
TARGET_FILE_PATH = (
    "v3sstorage/snapshot/2910D-01/2026/07/01/"
    "2910D-01_2026_07_01_14_35_23_785570.jpg"
)
TARGET_DEVICE_ID = "2910D-01"

def _headers(token):
    return {"X-Service-Token": token} if token else {}


def find_assets_by_file_path(file_path, device_id=None, base_url=BASE_URL, token=""):
    filename = os.path.basename(file_path)
    params = {
        "asset_type": "snapshot",
        "search": filename,
        "page": 1,
        "page_size": 20,
    }
    if device_id:
        params["device_id"] = device_id

    r = requests.get(
        f"{base_url}/api/admin/assets",
        params=params,
        headers=_headers(token),
        timeout=10,
    )
    if r.status_code == 401:
        raise RuntimeError(
            "Storage API returned 401 Unauthorized. Set STORAGE_MANAGE_TOKEN "
            "or pass --token with a valid service token."
        )
    r.raise_for_status()

    assets = r.json().get("assets", [])
    exact_matches = [
        asset for asset in assets
        if asset.get("file_path") == file_path or asset.get("filename") == filename
    ]
    return exact_matches or assets


def get_preview_url(asset_id, base_url=BASE_URL, token=""):
    r = requests.get(
        f"{base_url}/api/admin/assets/{asset_id}/preview-url",
        headers=_headers(token),
        timeout=10,
    )
    if r.status_code == 401:
        raise RuntimeError(
            "Storage API returned 401 Unauthorized while requesting preview URL. "
            "Set STORAGE_MANAGE_TOKEN or pass --token."
        )
    r.raise_for_status()
    return urljoin(f"{base_url.rstrip('/')}/", r.json()["access_url"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a saved V3S full-frame asset by storage file_path.")
    parser.add_argument("--base-url", default=os.getenv("STORAGE_MANAGE_BASE_URL", BASE_URL))
    parser.add_argument("--token", default=os.getenv("STORAGE_MANAGE_TOKEN", "st_OIA4rTxbC5hL30neTSkjj7x6kUmh7gvh"))
    parser.add_argument("--file-path", default=TARGET_FILE_PATH)
    parser.add_argument("--device-id", default=TARGET_DEVICE_ID)
    args = parser.parse_args()

    assets = find_assets_by_file_path(
        args.file_path,
        device_id=args.device_id,
        base_url=args.base_url,
        token=args.token,
    )
    if not assets:
        raise SystemExit(f"No matching saved full frame found for {args.file_path}")

    asset = assets[0]
    print("asset_id:", asset["id"])
    print("registered file_path:", asset["file_path"])
    print("filename:", asset["filename"])
    print("device_id:", asset["device_id"])
    print("timestamp:", asset["timestamp"])

    print("preview URL:", get_preview_url(asset["id"], base_url=args.base_url, token=args.token))
