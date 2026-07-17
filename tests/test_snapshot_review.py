import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snapshot import (
    BBox,
    CaptureEvent,
    VehicleCaptureProcessor,
    SnapshotClient,
    expand_bbox_to_region,
    is_one_missing_character_variant,
    load_service_config,
)


def make_event(plate, *, track=1, role="back", cam="2927D-02", produced_at="2026-07-02T01:00:00Z", confidence=1.0):
    return CaptureEvent(
        plate=plate,
        trigger_cam_id=cam,
        capture_cam_id=cam,
        capture_role=role,
        group_id="",
        message_id=f"msg-{plate}",
        frame_num=1,
        ntp_timestamp=None,
        produced_at=produced_at,
        snapshot_at=produced_at,
        tracking_object_id=track,
        confidence=confidence,
        event_type="object_update",
        bbox=BBox(left=0, top=0, width=100, height=100),
        crop_bbox=BBox(left=0, top=0, width=100, height=100),
        plate_bbox=None,
        image_width=1920,
        image_height=1080,
        received_monotonic=None,
        source_file_name=None,
        ai_result={},
        placeholder={},
    )


def write_metadata(root, storage_date, plate, filename, **overrides):
    path = Path(root) / storage_date / plate / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "plate": plate,
        "storage_date": storage_date,
        "capture_cam_id": "2927D-02",
        "capture_role": "back",
        "tracking_object_id": 1,
        "produced_at": "2026-07-02T01:00:00Z",
        "confidence": 1.0,
        "file": str(path.with_suffix(".jpg")),
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_service_config(root):
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


class SnapshotReviewTests(unittest.TestCase):
    def test_service_config_loads_without_ai_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_service_config(str(write_service_config(tmp)))

            self.assertEqual([group.group_id for group in config.camera_groups], ["2927D", "2903V"])
            self.assertEqual(
                [(target.role, target.cam_id) for target in config.camera_groups[0].cameras],
                [
                    ("front", "2927D-03"),
                    ("front", "2927D-04"),
                    ("back", "2927D-02"),
                    ("back", "2927D-05"),
                    ("mobile", "DKV3_2927D"),
                    ("mobile", "DKV3_2927D_02"),
                ],
            )

    def test_service_config_rejects_duplicate_cam_id_across_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera_groups.yaml"
            path.write_text(
                """
camera_groups:
  2927D:
    cameras:
      front:
        cam_id: CAM-01
  2903V:
    cameras:
      front:
        cam_id: CAM-01
""".lstrip(),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_service_config(str(path))

    def test_expand_group_events_uses_detected_camera_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_service_config(str(write_service_config(tmp)))
            processor = VehicleCaptureProcessor(
                SnapshotClient(output_root=tmp, dry_run=True),
                service_config=config,
            )

            expanded = processor.expand_group_events(make_event("30A12345", role="front", cam="2927D-03"))

            self.assertEqual(len(expanded), 1)
            self.assertEqual(expanded[0].group_id, "2927D")
            self.assertEqual(expanded[0].capture_role, "front")
            self.assertEqual(expanded[0].capture_cam_id, "2927D-03")

    def test_expand_group_events_maps_mobile_device_to_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_service_config(str(write_service_config(tmp)))
            processor = VehicleCaptureProcessor(
                SnapshotClient(output_root=tmp, dry_run=True),
                service_config=config,
            )

            expanded = processor.expand_group_events(make_event("30A12345", role="mobile", cam="DKV3_2927D"))

            self.assertEqual(len(expanded), 1)
            self.assertEqual(expanded[0].group_id, "2927D")
            self.assertEqual(expanded[0].capture_role, "mobile")
            self.assertEqual(expanded[0].capture_cam_id, "DKV3_2927D")

            expanded_second = processor.expand_group_events(make_event("30A12345", role="mobile", cam="DKV3_2927D_02"))
            self.assertEqual(len(expanded_second), 1)
            self.assertEqual(expanded_second[0].group_id, "2927D")
            self.assertEqual(expanded_second[0].capture_role, "mobile")
            self.assertEqual(expanded_second[0].capture_cam_id, "DKV3_2927D_02")

    def test_expand_group_events_skips_unknown_camera_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_service_config(str(write_service_config(tmp)))
            processor = VehicleCaptureProcessor(
                SnapshotClient(output_root=tmp, dry_run=True),
                service_config=config,
            )

            self.assertEqual(processor.expand_group_events(make_event("30A12345", cam="UNKNOWN")), [])

    def test_expands_plate_bbox_inside_region(self):
        expanded = expand_bbox_to_region(
            BBox(left=40, top=50, width=20, height=10),
            region=BBox(left=0, top=0, width=100, height=100),
            expand_ratio=0.5,
        )

        self.assertEqual(expanded, BBox(left=35, top=47.5, width=30, height=15))

    def test_expanded_plate_bbox_clamps_to_region(self):
        expanded = expand_bbox_to_region(
            BBox(left=2, top=4, width=20, height=10),
            region=BBox(left=0, top=0, width=100, height=100),
            expand_ratio=1.0,
        )

        self.assertEqual(expanded, BBox(left=0, top=0, width=32, height=19))

    def test_date_plate_path_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = SnapshotClient(output_root=tmp, dry_run=True)
            event = make_event("30M89587")

            self.assertEqual(
                client.capture_path(event, storage_date="2026-07-02"),
                Path(tmp) / "2026-07-02" / "30M89587" / "BACK_2927D02.jpg",
            )
            self.assertEqual(
                client.plate_capture_path(event, storage_date="2026-07-02"),
                Path(tmp) / "2026-07-02" / "30M89587" / "BACK_2927D02_plate.jpg",
            )

    def test_flags_missing_suffix_same_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = SnapshotClient(output_root=tmp, dry_run=True)
            storage_date = "2026-07-02"
            current = write_metadata(
                tmp,
                storage_date,
                "30K36766",
                "BACK_2927D02.json",
                tracking_object_id=3762,
                confidence=1.0,
            )
            other = write_metadata(
                tmp,
                storage_date,
                "30K3674",
                "BACK_2927D02.json",
                tracking_object_id=3762,
                confidence=0.991,
            )

            client.flag_similar_plates_for_review(
                storage_date=storage_date,
                event=make_event("30K36766", track=3762),
                metadata_path=current,
            )

            current_data = json.loads(current.read_text(encoding="utf-8"))
            other_data = json.loads(other.read_text(encoding="utf-8"))
            self.assertEqual(current_data["review_status"], "candidate")
            self.assertEqual(other_data["review_status"], "candidate")
            self.assertEqual(current_data["canonical_plate_suggestion"], "30K36766")
            self.assertIn("30K3674", current_data["similar_plates"])
            self.assertTrue(client.review_manifest_path(storage_date).is_file())

    def test_flags_shorter_plate_when_confidence_is_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = SnapshotClient(output_root=tmp, dry_run=True)
            storage_date = "2026-07-02"
            current = write_metadata(
                tmp,
                storage_date,
                "30F03813",
                "BACK_2927D02.json",
                tracking_object_id=3234,
                confidence=0.9975,
            )
            write_metadata(
                tmp,
                storage_date,
                "30F0381",
                "BACK_2927D02.json",
                tracking_object_id=3234,
                confidence=0.9997,
            )

            client.flag_similar_plates_for_review(
                storage_date=storage_date,
                event=make_event("30F03813", track=3234),
                metadata_path=current,
            )

            current_data = json.loads(current.read_text(encoding="utf-8"))
            self.assertEqual(current_data["canonical_plate_suggestion"], "30F03813")

    def test_flags_one_char_mismatch_same_track_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = SnapshotClient(output_root=tmp, dry_run=True)
            storage_date = "2026-07-02"
            current = write_metadata(
                tmp,
                storage_date,
                "30B64810",
                "FRONT_2927D03.json",
                capture_cam_id="2927D-03",
                capture_role="front",
                tracking_object_id=2912,
                produced_at="2026-07-02T01:00:00Z",
            )
            write_metadata(
                tmp,
                storage_date,
                "30D64810",
                "FRONT_2927D03.json",
                capture_cam_id="2927D-03",
                capture_role="front",
                tracking_object_id=2912,
                produced_at="2026-07-02T01:00:01Z",
            )

            client.flag_similar_plates_for_review(
                storage_date=storage_date,
                event=make_event("30B64810", track=2912, role="front", cam="2927D-03"),
                metadata_path=current,
            )

            current_data = json.loads(current.read_text(encoding="utf-8"))
            self.assertIn("30D64810", current_data["similar_plates"])

    def test_missing_character_helper_requires_single_deletion(self):
        self.assertTrue(is_one_missing_character_variant("30K4184", "30K41841"))
        self.assertTrue(is_one_missing_character_variant("15A4602", "15A44602"))
        self.assertFalse(is_one_missing_character_variant("29K03266", "29K03076"))
        self.assertFalse(is_one_missing_character_variant("30C60857", "30L60857"))

    def test_flags_missing_one_character_close_time_even_when_track_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = SnapshotClient(output_root=tmp, dry_run=True)
            storage_date = "2026-07-02"
            current = write_metadata(
                tmp,
                storage_date,
                "15A4602",
                "FRONT_2927D03.json",
                capture_cam_id="2927D-03",
                capture_role="front",
                tracking_object_id=9002,
                produced_at="2026-07-02T04:40:00Z",
                confidence=0.995,
            )
            write_metadata(
                tmp,
                storage_date,
                "15A44602",
                "FRONT_2927D03.json",
                capture_cam_id="2927D-03",
                capture_role="front",
                tracking_object_id=9001,
                produced_at="2026-07-02T04:39:00Z",
                confidence=0.999,
            )

            client.flag_similar_plates_for_review(
                storage_date=storage_date,
                event=make_event("15A4602", track=9002, role="front", cam="2927D-03"),
                metadata_path=current,
            )

            current_data = json.loads(current.read_text(encoding="utf-8"))
            self.assertEqual(current_data["review_status"], "candidate")
            self.assertEqual(current_data["review_reason"], "plate_distance_one")
            self.assertIn("15A44602", current_data["similar_plates"])

    def test_does_not_flag_different_tracks_even_when_time_is_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = SnapshotClient(output_root=tmp, dry_run=True)
            storage_date = "2026-07-02"
            current = write_metadata(
                tmp,
                storage_date,
                "29K03266",
                "FRONT_2927D03.json",
                capture_cam_id="2927D-03",
                capture_role="front",
                tracking_object_id=5228,
                produced_at="2026-07-02T02:10:35Z",
                confidence=0.999,
            )
            write_metadata(
                tmp,
                storage_date,
                "29K03076",
                "FRONT_2927D03.json",
                capture_cam_id="2927D-03",
                capture_role="front",
                tracking_object_id=5174,
                produced_at="2026-07-02T02:08:40Z",
                confidence=1.0,
            )

            client.flag_similar_plates_for_review(
                storage_date=storage_date,
                event=make_event("29K03266", track=5228, role="front", cam="2927D-03"),
                metadata_path=current,
            )

            current_data = json.loads(current.read_text(encoding="utf-8"))
            self.assertNotIn("review_status", current_data)
            self.assertFalse(client.review_manifest_path(storage_date).exists())

    def test_does_not_flag_distance_two_when_tracks_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = SnapshotClient(output_root=tmp, dry_run=True)
            storage_date = "2026-07-02"
            current = write_metadata(
                tmp,
                storage_date,
                "29K03266",
                "FRONT_2927D03.json",
                capture_cam_id="2927D-03",
                capture_role="front",
                tracking_object_id=5228,
                produced_at="2026-07-02T02:10:35Z",
                confidence=0.999,
            )
            write_metadata(
                tmp,
                storage_date,
                "29K03076",
                "FRONT_2927D03.json",
                capture_cam_id="2927D-03",
                capture_role="front",
                tracking_object_id=5174,
                produced_at="2026-07-02T02:08:40Z",
                confidence=1.0,
            )

            client.flag_similar_plates_for_review(
                storage_date=storage_date,
                event=make_event("29K03266", track=5228, role="front", cam="2927D-03"),
                metadata_path=current,
            )

            current_data = json.loads(current.read_text(encoding="utf-8"))
            self.assertNotIn("review_status", current_data)
            self.assertFalse(client.review_manifest_path(storage_date).exists())

    def test_flags_distance_one_even_when_tracks_and_time_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = SnapshotClient(output_root=tmp, dry_run=True)
            storage_date = "2026-07-02"
            current = write_metadata(
                tmp,
                storage_date,
                "30H46320",
                "FRONT_2927D03.json",
                capture_cam_id="2927D-03",
                capture_role="front",
                tracking_object_id=1729,
                produced_at="2026-07-02T01:00:00Z",
            )
            write_metadata(
                tmp,
                storage_date,
                "30M46320",
                "FRONT_2927D03.json",
                capture_cam_id="2927D-03",
                capture_role="front",
                tracking_object_id=1668,
                produced_at="2026-07-02T01:10:00Z",
            )

            client.flag_similar_plates_for_review(
                storage_date=storage_date,
                event=make_event("30H46320", track=1729, role="front", cam="2927D-03"),
                metadata_path=current,
            )

            current_data = json.loads(current.read_text(encoding="utf-8"))
            self.assertEqual(current_data["review_status"], "candidate")
            self.assertEqual(current_data["review_reason"], "plate_distance_one")
            self.assertIn("30M46320", current_data["similar_plates"])

    def test_does_not_append_duplicate_review_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = SnapshotClient(output_root=tmp, dry_run=True)
            storage_date = "2026-07-02"
            current = write_metadata(
                tmp,
                storage_date,
                "30K36766",
                "BACK_2927D02.json",
                tracking_object_id=3762,
                confidence=1.0,
            )
            write_metadata(
                tmp,
                storage_date,
                "30K3674",
                "BACK_2927D02.json",
                tracking_object_id=3762,
                confidence=0.991,
            )

            client.flag_similar_plates_for_review(
                storage_date=storage_date,
                event=make_event("30K36766", track=3762),
                metadata_path=current,
            )
            client.flag_similar_plates_for_review(
                storage_date=storage_date,
                event=make_event("30K36766", track=3762),
                metadata_path=current,
            )

            manifest_lines = client.review_manifest_path(storage_date).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(manifest_lines), 1)


if __name__ == "__main__":
    unittest.main()
