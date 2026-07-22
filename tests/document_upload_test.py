#!/usr/bin/env python3
"""Upload randomly selected local images to the document API."""

import argparse
import base64
import mimetypes
import os
import random
from datetime import datetime
from pathlib import Path

import requests


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def encoded_image(path: Path) -> str:
    image_data = base64.b64encode(path.read_bytes()).decode("ascii")
    content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{content_type};base64,{image_data}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the document upload API with random images.")
    parser.add_argument("--base-url", default=os.environ.get("SYNC_API_BASE_URL", "http://localhost:8890"))
    parser.add_argument("--token", default=os.environ.get("SUBMIT_TOKEN", ""))
    parser.add_argument("--image-dir", default=".", help="Directory to search recursively for images")
    parser.add_argument("--device-id", default="test-device")
    parser.add_argument("--count", type=int, choices=range(1, 21), default=2)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    candidates = [
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(candidates) < args.count:
        parser.error(f"found {len(candidates)} images under {image_dir}, but --count is {args.count}")

    image_paths = random.sample(candidates, args.count)
    payload = {
        "deviceId": args.device_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "documents": [
            {
                "image": encoded_image(image_path),
                "recordId": f"test-{index}-{image_path.stem}",
            }
            for index, image_path in enumerate(image_paths, 1)
        ],
    }

    response = requests.post(
        f"{args.base_url.rstrip('/')}/api/documents",
        headers={"X-API-Token": args.token},
        json=payload,
        timeout=30,
    )
    print("selected images:")
    for image_path in image_paths:
        print(f"  {image_path}")
    print(f"status: {response.status_code}")
    print(response.text)
    response.raise_for_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
