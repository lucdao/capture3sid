import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snapshot import (
    BBox,
    CameraGroup,
    CameraTarget,
    CaptureEvent,
    ServiceConfig,
    SnapshotClient,
    StorageAssetClient,
    VehicleCaptureProcessor,
    iter_capture_events,
)


def make_response(*, status=200, json_data=None, content=b"", content_type=""):
    response = Mock()
    response.status_code = status
    response.content = content
    response.headers = {"Content-Type": content_type} if content_type else {}
    response.json.return_value = json_data or {}
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"status={status}")
    return response


def make_event(*, event_type="object_update", asset_id="asset-123", file_name="legacy/path.jpg"):
    return CaptureEvent(
        plate="30A12345",
        trigger_cam_id="CAM-1",
        capture_cam_id="CAM-1",
        capture_role="front",
        group_id="group-1",
        message_id="message-1",
        frame_num=1,
        ntp_timestamp=None,
        produced_at="2026-07-22T08:00:00Z",
        snapshot_at="2026-07-22T08:00:00Z",
        tracking_object_id=7,
        confidence=1.0,
        event_type=event_type,
        bbox=BBox(left=10, top=10, width=100, height=100),
        crop_bbox=BBox(left=0, top=0, width=120, height=120),
        plate_bbox=None,
        image_width=1920,
        image_height=1080,
        received_monotonic=None,
        source_asset_id=asset_id,
        source_file_name=file_name,
        ai_result={},
        placeholder={},
    )


class CaptureEventAssetIdTests(unittest.TestCase):
    def test_extracts_image_asset_id_and_preserves_it_in_group_event(self):
        payload = {
            "cam_id": "CAM-1",
            "message_id": "message-1",
            "produced_at": "2026-07-22T08:00:00Z",
            "image": {
                "width": 1920,
                "height": 1080,
                "asset_id": " asset-123 ",
                "image_path": "legacy/path.jpg",
            },
            "ai_results": [{
                "meta_type": "car",
                "event_type": "object_update",
                "confidence": 1.0,
                "detected_object_ids": "30A12345",
                "bbox": {"left": 10, "top": 10, "width": 100, "height": 100},
            }],
        }

        event = list(iter_capture_events(payload))[0]
        self.assertEqual(event.source_asset_id, "asset-123")
        self.assertEqual(event.source_file_name, "legacy/path.jpg")

        processor = VehicleCaptureProcessor(
            Mock(),
            service_config=ServiceConfig(camera_groups=(
                CameraGroup(
                    group_id="group-1",
                    cameras=(CameraTarget(role="front", cam_id="CAM-1"),),
                ),
            )),
        )
        expanded = processor.expand_group_events(event)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].source_asset_id, "asset-123")


class StorageAssetClientTests(unittest.TestCase):
    def test_downloads_via_service_access_url_by_asset_id(self):
        session = Mock()
        session.post.return_value = make_response(json_data={"access_url": "/minio/image.jpg"})
        session.get.return_value = make_response(content=b"image-bytes", content_type="image/jpeg")
        client = StorageAssetClient(
            base_url="http://storage/api/storage",
            token="secret",
            session=session,
            retries=0,
        )

        content, asset, preview_url = client.download_by_asset_id("asset/one")

        self.assertEqual(content, b"image-bytes")
        self.assertEqual(asset, {"id": "asset/one"})
        self.assertEqual(preview_url, "http://storage/minio/image.jpg")
        self.assertEqual(session.post.call_args.args[0], "http://storage/api/storage/get-access-url")
        self.assertEqual(session.post.call_args.kwargs["json"], {
            "asset_id": "asset/one",
            "access_scope": "file",
            "duration": 300,
            "as_attachment": False,
        })

    def test_retries_preview_lookup_and_download(self):
        session = Mock()
        session.post.side_effect = [
            requests.ConnectionError("temporary"),
            make_response(json_data={"access_url": "http://minio/image.jpg"}),
        ]
        session.get.return_value = make_response(content=b"image-bytes", content_type="image/jpeg")
        client = StorageAssetClient(session=session, retries=1, retry_delay_sec=0)

        content, _, _ = client.download_by_asset_id("asset-123")

        self.assertEqual(content, b"image-bytes")
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(session.get.call_count, 1)

    def test_retries_after_image_download_failure(self):
        preview_response = make_response(json_data={"access_url": "http://minio/image.jpg"})
        session = Mock()
        session.post.side_effect = [preview_response, preview_response]
        session.get.side_effect = [
            requests.ConnectionError("minio unavailable"),
            make_response(content=b"image-bytes", content_type="image/jpeg"),
        ]
        client = StorageAssetClient(session=session, retries=1, retry_delay_sec=0)

        content, _, _ = client.download_by_asset_id("asset-123")

        self.assertEqual(content, b"image-bytes")
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(session.get.call_count, 2)

    def test_rejects_missing_asset_id(self):
        client = StorageAssetClient(session=Mock())
        with self.assertRaisesRegex(ValueError, "asset_id is required"):
            client.download_by_asset_id("  ")

    def test_reports_unauthorized_access_url_request(self):
        session = Mock()
        session.post.return_value = make_response(status=401)
        client = StorageAssetClient(session=session, retries=0)

        with self.assertRaisesRegex(RuntimeError, "401 unauthorized"):
            client.download_by_asset_id("asset-123")

    def test_reports_forbidden_access_url_request(self):
        session = Mock()
        session.post.return_value = make_response(status=403)
        client = StorageAssetClient(session=session, retries=0)

        with self.assertRaisesRegex(RuntimeError, "403 forbidden"):
            client.download_by_asset_id("asset-123")
        self.assertEqual(session.post.call_count, 1)

    def test_rejects_non_image_and_empty_downloads(self):
        cases = [
            (b"not-an-image", "application/json", "not an image"),
            (b"", "image/jpeg", "empty body"),
        ]
        for content, content_type, error in cases:
            with self.subTest(error=error):
                session = Mock()
                session.post.return_value = make_response(json_data={"access_url": "http://minio/image.jpg"})
                session.get.return_value = make_response(content=content, content_type=content_type)
                client = StorageAssetClient(session=session, retries=0)
                with self.assertRaisesRegex(RuntimeError, error):
                    client.download_by_asset_id("asset-123")


class SnapshotSourceRoutingTests(unittest.TestCase):
    def make_client(self):
        client = object.__new__(SnapshotClient)
        client.capture_from_storage_source = Mock(return_value={"source": "storage"})
        client.capture_from_camera = Mock(return_value={"source": "camera"})
        return client

    def test_storage_event_with_asset_id_uses_storage(self):
        client = self.make_client()
        result = client.capture(make_event(), storage_date="2026-07-22")
        self.assertEqual(result, {"source": "storage"})
        client.capture_from_camera.assert_not_called()

    def test_missing_asset_id_falls_back_to_camera_despite_legacy_path(self):
        client = self.make_client()
        event = make_event(asset_id=None, file_name="legacy/path.jpg")
        result = client.capture(event, storage_date="2026-07-22")
        self.assertEqual(result, {"source": "camera"})
        client.capture_from_storage_source.assert_not_called()

    def test_object_exist_uses_camera_even_when_asset_id_exists(self):
        client = self.make_client()
        result = client.capture(make_event(event_type="object_exist"), storage_date="2026-07-22")
        self.assertEqual(result, {"source": "camera"})
        client.capture_from_storage_source.assert_not_called()


class SnapshotStorageMetadataTests(unittest.TestCase):
    def test_capture_records_asset_id_source_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage_client = Mock()
            storage_client.download_by_asset_id.return_value = (
                b"image-bytes",
                {"id": "asset-123"},
                "http://minio/image.jpg",
            )
            client = SnapshotClient(output_root=tmp, storage_asset_client=storage_client)
            client.crop_storage_image_to_file = Mock(return_value=False)
            client.write_capture_metadata = Mock()
            client.cleanup_replaced_record = Mock()
            client.cleanup_view_duplicates = Mock()
            client.flag_similar_plates_for_review = Mock()

            client.capture_from_storage_source(make_event(), storage_date="2026-07-22")

            storage_client.download_by_asset_id.assert_called_once_with("asset-123")
            metadata_kwargs = client.write_capture_metadata.call_args.kwargs
            self.assertEqual(metadata_kwargs["source_method"], "storage_asset_id")
            self.assertEqual(metadata_kwargs["storage_asset"], {"id": "asset-123"})


if __name__ == "__main__":
    unittest.main()
