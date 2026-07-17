import base64
import json
import argparse
import sys
import tempfile
import unittest
from unittest.mock import Mock
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture_sync_api import (
    SubmitPayload,
    build_changes_manifest,
    create_app,
    load_camera_group_index,
    resolve_relative_path,
    save_mobile_submit,
    validate_submit_token,
)


def write_capture(root, storage_date, plate, stem, *, saved_at, with_plate=False, cam_id="2927D-02", group_id=None):
    folder = Path(root) / storage_date / plate
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / f"{stem}.jpg"
    metadata_path = folder / f"{stem}.json"
    image_path.write_bytes(b"\xff\xd8\xfftest")
    metadata = {
        "plate": plate,
        "storage_date": storage_date,
        "saved_at": saved_at,
        "produced_at": saved_at,
        "capture_role": stem.split("_", 1)[0].lower(),
        "capture_cam_id": cam_id,
        "file": str(image_path),
        "confidence": 1.0,
    }
    if group_id is not None:
        metadata["group_id"] = group_id
    if with_plate:
        plate_path = folder / f"{stem}_plate.jpg"
        plate_path.write_bytes(b"\xff\xd8\xffplate")
        metadata["plate_image_file"] = str(plate_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def write_review(root, storage_date, records):
    path = Path(root) / "_review" / storage_date / "suspicious_plates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def write_camera_group_config(root):
    path = Path(root) / "camera_groups.yaml"
    path.write_text(
        """
camera_groups:
  2927D:
    name: 2927D
    cameras:
      front:
        - cam_id: 2927D-03
        - cam_id: 2927D-04
      back:
        cam_ids:
          - 2927D-02
          - 2927D-05
      mobile:
        cam_ids:
          - DKV3_2927D
          - DKV3_2927D_02
  2903V:
    name: 2903V
    cameras:
      front:
        cam_id: CAM_1_DK_TEST
      back:
        cam_id: CAM_2_DK_TEST
      mobile:
        cam_ids:
          - DKV3_2903V
          - DKV3_2903V_02
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_image_data_url(content=b"\xff\xd8\xffmobile"):
    return "data:image/jpeg;base64," + base64.b64encode(content).decode("ascii")


def submit_payload(**overrides):
    payload = {
        "plate": "30A-12345",
        "plateSource": "mobile",
        "deviceId": "device-01",
        "timestamp": "2026-06-29T12:00:00+07:00",
        "recordId": "external-id",
        "photos": [
            {
                "type": "FRONT",
                "image": test_image_data_url(),
            }
        ],
    }
    payload.update(overrides)
    return payload


class CaptureSyncApiTests(unittest.TestCase):
    def test_mobile_submit_route_is_registered(self):
        provider = Mock()
        provider.load.return_value = Mock(
            version="test",
            config={"camera_groups": {"2927D": {"name": "2927D", "cameras": {"front": {"cam_ids": ["2927D-02"]}}}}},
        )
        app = create_app(
            argparse.Namespace(
                output_root="/tmp/vehicle_captures",
                sync_token="",
                submit_token="secret",
                storage_timezone="Asia/Ho_Chi_Minh",
                camera_config_provider=provider,
            )
        )

        submit_routes = [
            route
            for route in app.routes
            if getattr(route, "path", None) == "/api/submit" and "POST" in getattr(route, "methods", set())
        ]
        routes = {
            (getattr(route, "path", ""), method)
            for route in app.routes
            for method in getattr(route, "methods", set())
        }

        self.assertEqual(len(submit_routes), 1)
        self.assertIn(("/api/admin/camera-groups", "GET"), routes)
        self.assertNotIn(("/api/admin/camera-groups", "PUT"), routes)
        self.assertNotIn(("/admin/camera-groups", "GET"), routes)

    def test_mobile_submit_saves_image_plate_crop_and_metadata_by_payload_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = submit_payload(
                photos=[
                    {
                        "type": "BACK",
                        "image": test_image_data_url(b"\xff\xd8\xffmobile"),
                        "plate_image": test_image_data_url(b"\xff\xd8\xffplate"),
                    }
                ]
            )
            result = save_mobile_submit(
                Path(tmp),
                SubmitPayload(**payload),
                storage_timezone="Asia/Ho_Chi_Minh",
            )

            self.assertEqual(result["plate"], "30A12345")
            self.assertEqual(result["storage_date"], "2026-06-29")
            self.assertEqual(result["count"], 1)

            image_path = Path(tmp) / "2026-06-29" / "30A12345" / "BACK_DEVICE-01.jpg"
            plate_image_path = Path(tmp) / "2026-06-29" / "30A12345" / "BACK_DEVICE-01_plate.jpg"
            metadata_path = image_path.with_suffix(".json")
            self.assertEqual(image_path.read_bytes(), b"\xff\xd8\xffmobile")
            self.assertEqual(plate_image_path.read_bytes(), b"\xff\xd8\xffplate")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["plate"], "30A12345")
            self.assertEqual(metadata["storage_date"], "2026-06-29")
            self.assertEqual(metadata["capture_role"], "back")
            self.assertEqual(metadata["capture_cam_id"], "device-01")
            self.assertEqual(metadata["device_id"], "device-01")
            self.assertEqual(metadata["plate_source"], "mobile")
            self.assertEqual(metadata["record_id"], "external-id")
            self.assertEqual(metadata["produced_at"], "2026-06-29T05:00:00Z")
            self.assertEqual(metadata["source_method"], "mobile_submit")
            self.assertEqual(metadata["file"], str(image_path))
            self.assertEqual(metadata["plate_image_file"], str(plate_image_path))
            self.assertEqual(metadata["confidence"], 0.99)
            self.assertEqual(metadata["status"], 200)
            self.assertIn("saved_at", metadata)
            self.assertEqual(result["files"][0]["plate_image_path"], str(plate_image_path))

    def test_mobile_submit_overwrites_duplicate_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = submit_payload(photos=[{"type": "FRONT", "image": test_image_data_url(b"\xff\xd8\xfffirst")}])
            second = submit_payload(photos=[{"type": "FRONT", "image": test_image_data_url(b"\xff\xd8\xffsecond")}])

            save_mobile_submit(Path(tmp), SubmitPayload(**first), storage_timezone="Asia/Ho_Chi_Minh")
            save_mobile_submit(Path(tmp), SubmitPayload(**second), storage_timezone="Asia/Ho_Chi_Minh")

            image_path = Path(tmp) / "2026-06-29" / "30A12345" / "FRONT_DEVICE-01.jpg"
            self.assertEqual(image_path.read_bytes(), b"\xff\xd8\xffsecond")

    def test_mobile_submit_rejects_missing_device_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = submit_payload(deviceId="")

            with self.assertRaises(ValueError):
                save_mobile_submit(Path(tmp), SubmitPayload(**payload), storage_timezone="Asia/Ho_Chi_Minh")

    def test_mobile_submit_rejects_missing_or_invalid_token(self):
        for supplied in ("", "wrong"):
            with self.assertRaises(HTTPException) as ctx:
                validate_submit_token("secret", supplied)

            self.assertEqual(ctx.exception.status_code, 401)

    def test_mobile_submit_rejects_when_submit_token_not_configured(self):
        with self.assertRaises(HTTPException) as ctx:
            validate_submit_token("", "secret")

        self.assertEqual(ctx.exception.status_code, 401)

    def test_mobile_submit_rejects_invalid_base64_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = submit_payload(photos=[{"type": "FRONT", "image": "not-base64"}])

            with self.assertRaises(ValueError):
                save_mobile_submit(Path(tmp), SubmitPayload(**payload), storage_timezone="Asia/Ho_Chi_Minh")

    def test_mobile_submit_normalizes_unknown_photo_type_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = submit_payload(photos=[{"type": "left side", "image": test_image_data_url()}])

            result = save_mobile_submit(Path(tmp), SubmitPayload(**payload), storage_timezone="Asia/Ho_Chi_Minh")

            self.assertEqual(result["count"], 1)
            image_path = Path(tmp) / "2026-06-29" / "30A12345" / "LEFT_SIDE_DEVICE-01.jpg"
            self.assertTrue(image_path.is_file())

    def test_mobile_submit_allows_photo_without_plate_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = submit_payload(photos=[{"type": "SIDE", "image": test_image_data_url()}])

            result = save_mobile_submit(Path(tmp), SubmitPayload(**payload), storage_timezone="Asia/Ho_Chi_Minh")

            image_path = Path(tmp) / "2026-06-29" / "30A12345" / "SIDE_DEVICE-01.jpg"
            plate_image_path = Path(tmp) / "2026-06-29" / "30A12345" / "SIDE_DEVICE-01_plate.jpg"
            metadata_path = image_path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(result["count"], 1)
            self.assertTrue(image_path.is_file())
            self.assertFalse(plate_image_path.exists())
            self.assertNotIn("plate_image_file", metadata)

    def test_manifest_paginates_same_saved_at_with_cursor_tiebreaker(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_capture(tmp, "2026-07-02", "30A00001", "BACK_2927D02", saved_at="2026-07-02T01:00:00Z")
            write_capture(tmp, "2026-07-02", "30A00002", "BACK_2927D02", saved_at="2026-07-02T01:00:00Z")

            first = build_changes_manifest(
                Path(tmp),
                since="2026-07-02T00:00:00Z",
                cursor=None,
                limit=1,
                base_url="http://sync.local",
                now=datetime(2026, 7, 2, 2, 0, tzinfo=timezone.utc),
            )
            second = build_changes_manifest(
                Path(tmp),
                since=None,
                cursor=first["next_cursor"],
                limit=1,
                base_url="http://sync.local",
            )

            self.assertTrue(first["has_more"])
            self.assertEqual(len(first["items"]), 1)
            self.assertFalse(second["has_more"])
            self.assertEqual(len(second["items"]), 1)
            self.assertNotEqual(
                first["items"][0]["relative_metadata_path"],
                second["items"][0]["relative_metadata_path"],
            )

    def test_cursor_upper_bound_excludes_new_files_during_paging(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_capture(tmp, "2026-07-02", "30A00001", "BACK_2927D02", saved_at="2026-07-02T01:00:00Z")
            write_capture(tmp, "2026-07-02", "30A00002", "BACK_2927D02", saved_at="2026-07-02T01:01:00Z")
            first = build_changes_manifest(
                Path(tmp),
                since="2026-07-02T00:00:00Z",
                cursor=None,
                limit=1,
                base_url="http://sync.local",
                now=datetime(2026, 7, 2, 1, 1, 30, tzinfo=timezone.utc),
            )
            write_capture(tmp, "2026-07-02", "30A00003", "BACK_2927D02", saved_at="2026-07-02T01:02:00Z")

            second = build_changes_manifest(
                Path(tmp),
                since=None,
                cursor=first["next_cursor"],
                limit=10,
                base_url="http://sync.local",
            )

            plates = [item["plate"] for item in second["items"]]
            self.assertEqual(plates, ["30A00002"])
            self.assertEqual(second["next_since"], "2026-07-02T01:01:30Z")

    def test_manifest_filters_captures_by_camera_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            group_index = load_camera_group_index(str(write_camera_group_config(tmp)))
            write_capture(
                tmp,
                "2026-07-02",
                "30A00001",
                "BACK_2927D02",
                saved_at="2026-07-02T01:00:00Z",
                cam_id="2927D-02",
            )
            write_capture(
                tmp,
                "2026-07-02",
                "30A00002",
                "BACK_CAM2DKTEST",
                saved_at="2026-07-02T01:01:00Z",
                cam_id="CAM_2_DK_TEST",
            )

            result = build_changes_manifest(
                Path(tmp),
                since="2026-07-02T00:00:00Z",
                cursor=None,
                limit=10,
                base_url="http://sync.local",
                location="2927D",
                group_index=group_index,
                now=datetime(2026, 7, 2, 2, 0, tzinfo=timezone.utc),
            )

            self.assertEqual([item["plate"] for item in result["items"]], ["30A00001"])
            self.assertEqual(result["location"], "2927D")
            self.assertEqual(result["items"][0]["camera_group_ids"], ["2927D"])

    def test_manifest_filters_mobile_capture_by_group_device_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            group_index = load_camera_group_index(str(write_camera_group_config(tmp)))
            payload = submit_payload(deviceId="DKV3_2903V")
            save_mobile_submit(Path(tmp), SubmitPayload(**payload), storage_timezone="Asia/Ho_Chi_Minh")

            result = build_changes_manifest(
                Path(tmp),
                since="2026-06-29T00:00:00Z",
                cursor=None,
                limit=10,
                base_url="http://sync.local",
                camera_group="2903V",
                group_index=group_index,
                now=datetime(2026, 7, 14, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(len(result["items"]), 1)
            self.assertEqual(result["location"], "2903V")
            self.assertEqual(result["camera_group"], "2903V")
            self.assertEqual(result["items"][0]["camera_group_ids"], ["2903V"])

    def test_group_filter_is_preserved_in_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            group_index = load_camera_group_index(str(write_camera_group_config(tmp)))
            write_capture(
                tmp,
                "2026-07-02",
                "30A00001",
                "BACK_2927D02",
                saved_at="2026-07-02T01:00:00Z",
                cam_id="2927D-02",
            )
            write_capture(
                tmp,
                "2026-07-02",
                "30A00002",
                "FRONT_2927D03",
                saved_at="2026-07-02T01:01:00Z",
                cam_id="2927D-03",
            )
            write_capture(
                tmp,
                "2026-07-02",
                "30A00003",
                "BACK_CAM2DKTEST",
                saved_at="2026-07-02T01:02:00Z",
                cam_id="CAM_2_DK_TEST",
            )

            first = build_changes_manifest(
                Path(tmp),
                since="2026-07-02T00:00:00Z",
                cursor=None,
                limit=1,
                base_url="http://sync.local",
                location="2927D",
                group_index=group_index,
                now=datetime(2026, 7, 2, 3, 0, tzinfo=timezone.utc),
            )
            second = build_changes_manifest(
                Path(tmp),
                since=None,
                cursor=first["next_cursor"],
                limit=10,
                base_url="http://sync.local",
                group_index=group_index,
            )

            self.assertEqual(first["location"], "2927D")
            self.assertEqual([item["plate"] for item in second["items"]], ["30A00002"])

    def test_manifest_filters_review_records_by_candidate_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            group_index = load_camera_group_index(str(write_camera_group_config(tmp)))
            write_review(
                tmp,
                "2026-07-02",
                [
                    {
                        "created_at": "2026-07-02T01:00:00Z",
                        "review_group_id": "lane-01-review",
                        "candidates": [{"capture_cam_id": "2927D-02"}],
                    },
                    {
                        "created_at": "2026-07-02T01:01:00Z",
                        "review_group_id": "lane-02-review",
                        "candidates": [{"capture_cam_id": "CAM_2_DK_TEST"}],
                    },
                ],
            )

            result = build_changes_manifest(
                Path(tmp),
                since="2026-07-02T00:00:00Z",
                cursor=None,
                limit=10,
                base_url="http://sync.local",
                location="2903V",
                group_index=group_index,
                now=datetime(2026, 7, 2, 2, 0, tzinfo=timezone.utc),
            )

            self.assertEqual([item["review_group_id"] for item in result["items"]], ["lane-02-review"])

    def test_location_index_supports_multiple_cameras_per_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            group_index = load_camera_group_index(str(write_camera_group_config(tmp)))

            self.assertEqual(group_index.cam_to_group["2927D-03"], "2927D")
            self.assertEqual(group_index.cam_to_group["2927D-04"], "2927D")
            self.assertEqual(group_index.cam_to_group["2927D-02"], "2927D")
            self.assertEqual(group_index.cam_to_group["2927D-05"], "2927D")
            self.assertEqual(group_index.cam_to_group["DKV3_2927D"], "2927D")
            self.assertEqual(group_index.cam_to_group["DKV3_2927D_02"], "2927D")

    def test_capture_files_include_optional_plate_image_only_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_capture(
                tmp,
                "2026-07-02",
                "30A00001",
                "BACK_2927D02",
                saved_at="2026-07-02T01:00:00Z",
                with_plate=True,
            )

            result = build_changes_manifest(
                Path(tmp),
                since="2026-07-02T00:00:00Z",
                cursor=None,
                limit=10,
                base_url="http://sync.local",
                now=datetime(2026, 7, 2, 2, 0, tzinfo=timezone.utc),
            )

            files = result["items"][0]["files"]
            relative_paths = {file["relative_path"] for file in files}
            self.assertIn("2026-07-02/30A00001/BACK_2927D02.json", relative_paths)
            self.assertIn("2026-07-02/30A00001/BACK_2927D02.jpg", relative_paths)
            self.assertIn("2026-07-02/30A00001/BACK_2927D02_plate.jpg", relative_paths)

    def test_review_jsonl_records_are_included_by_created_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_review(
                tmp,
                "2026-07-02",
                [
                    {
                        "created_at": "2026-07-02T01:00:00Z",
                        "review_group_id": "old",
                        "storage_date": "2026-07-02",
                    },
                    {
                        "created_at": "2026-07-02T01:10:00Z",
                        "review_group_id": "new",
                        "storage_date": "2026-07-02",
                    },
                ],
            )

            result = build_changes_manifest(
                Path(tmp),
                since="2026-07-02T01:05:00Z",
                cursor=None,
                limit=10,
                base_url="http://sync.local",
                now=datetime(2026, 7, 2, 2, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(len(result["items"]), 1)
            self.assertEqual(result["items"][0]["type"], "review")
            self.assertEqual(result["items"][0]["review_group_id"], "new")
            self.assertEqual(
                result["items"][0]["files"][0]["relative_path"],
                "_review/2026-07-02/suspicious_plates.jsonl",
            )

    def test_resolve_relative_path_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                resolve_relative_path(Path(tmp), "../secret.txt")


if __name__ == "__main__":
    unittest.main()
