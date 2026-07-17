import unittest
from pathlib import Path
from unittest.mock import Mock

import requests
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_config_store import (
    CameraConfigProvider,
    camera_api_payload_to_config,
    inspection_camera_api_payload_to_config,
)


def payload(*, page=1, total_pages=1, station_code="2927D", cameras=None):
    return {
        "page": page,
        "total_pages": total_pages,
        "stations": [{
            "station_id": "ignored-station-uuid",
            "station_code": station_code,
            "station_name": station_code,
            "iapp_ids": [station_code],
            "lines": [{"camera_in_id": "ignored-camera-uuid"}],
            "cameras": cameras if cameras is not None else [
                {"camera_id": "ignored-in-uuid", "camera_code": f"{station_code}-02", "camera_inout": "in"},
                {"camera_id": "ignored-out-uuid", "camera_code": f"{station_code}-03", "camera_inout": "out"},
            ],
        }],
    }


def updated_payload(*, page=1, total_pages=1, station_code="2910D", cameras=None):
    data = payload(page=page, total_pages=total_pages, station_code=station_code, cameras=[])
    data["stations"][0]["iapp_ids"] = None
    data["stations"][0]["cameras"] = cameras if cameras is not None else [
        {"camera_id": "front-uuid", "camera_code": f"{station_code}-01", "camera_frontback": "front"},
        {"camera_id": "back-uuid", "camera_code": f"{station_code}-02", "camera_frontback": "back"},
    ]
    data["page_size"] = 20
    data["total_count"] = len(data["stations"])
    return data


def response(data):
    result = Mock()
    result.json.return_value = data
    result.raise_for_status.return_value = None
    return result


class CameraApiConfigTests(unittest.TestCase):
    def test_uses_codes_and_maps_in_front_out_back(self):
        config = camera_api_payload_to_config([payload()])
        cameras = config["camera_groups"]["2927D"]["cameras"]
        self.assertEqual(cameras["front"]["cam_ids"], ["2927D-02"])
        self.assertEqual(cameras["back"]["cam_ids"], ["2927D-03"])
        self.assertEqual(cameras["mobile"]["cam_ids"], ["2927D"])
        self.assertNotIn("ignored-in-uuid", str(config))

    def test_does_not_synthesize_legacy_mobile_id(self):
        config = camera_api_payload_to_config([payload()])
        self.assertNotIn("DKV3_2927D", str(config))

    def test_invalid_iapp_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "iapp_ids must be a list"):
            camera_api_payload_to_config([{**payload(), "stations": [{**payload()["stations"][0], "iapp_ids": "2927D"}]}])
        with self.assertRaisesRegex(ValueError, "empty iapp_id"):
            camera_api_payload_to_config([{**payload(), "stations": [{**payload()["stations"][0], "iapp_ids": [" "]}]}])

    def test_duplicate_iapp_id_is_rejected(self):
        first = payload(station_code="A")
        second = payload(station_code="B")
        second["stations"][0]["iapp_ids"] = ["A"]
        with self.assertRaisesRegex(ValueError, "duplicate iapp_id"):
            camera_api_payload_to_config([first, second])

    def test_inout_is_compatible_with_back_role(self):
        config = camera_api_payload_to_config([
            payload(cameras=[{"camera_code": "2927D-03", "camera_inout": "inout"}])
        ])
        cameras = config["camera_groups"]["2927D"]["cameras"]
        self.assertEqual(cameras["back"]["cam_ids"], ["2927D-03"])
        self.assertNotIn("front", cameras)

    def test_updated_frontback_field_maps_front_back_and_both(self):
        config = inspection_camera_api_payload_to_config([
            updated_payload(cameras=[
                {"camera_code": "2910D-01", "camera_frontback": "front"},
                {"camera_code": "2910D-02", "camera_frontback": "back"},
                {"camera_code": "2910D-03", "camera_frontback": "frontback"},
            ])
        ])
        cameras = config["camera_groups"]["2910D"]["cameras"]
        self.assertEqual(cameras["front"]["cam_ids"], ["2910D-01", "2910D-03"])
        self.assertEqual(cameras["back"]["cam_ids"], ["2910D-02", "2910D-03"])

    def test_missing_direction_skips_unclassified_camera(self):
        config = inspection_camera_api_payload_to_config([
            payload(cameras=[
                {"camera_code": "2910D-01", "camera_inout": ""},
                {"camera_code": "2910D-02", "camera_inout": "in"},
            ])
        ])
        cameras = config["camera_groups"]["2927D"]["cameras"]
        self.assertEqual(cameras["front"]["cam_ids"], ["2910D-02"])
        self.assertNotIn("2910D-01", str(config))

    def test_fetches_all_pages(self):
        session = Mock()
        session.get.side_effect = [
            response(payload(page=1, total_pages=2, station_code="A")),
            response(payload(page=2, total_pages=2, station_code="B")),
        ]
        state = CameraConfigProvider(base_url="http://camera", session=session).load(force=True)
        self.assertEqual(set(state.config["camera_groups"]), {"A", "B"})
        self.assertEqual(session.get.call_count, 2)
        self.assertEqual(session.get.call_args_list[0].args[0], "http://camera/api/inspections/stations/cameras")
        self.assertEqual(session.get.call_args_list[0].kwargs["params"], {"page": 1, "limit": 20})
        self.assertEqual(session.get.call_args_list[0].kwargs["headers"], {"accept": "application/json"})

    def test_sends_raw_authorization_header(self):
        session = Mock()
        session.get.return_value = response(payload())
        provider = CameraConfigProvider(
            base_url="http://camera",
            authorization="YGDweJ9Ucx58ZG8iW4fyTfx5KmSPAz4u",
            session=session,
        )
        provider.load(force=True)
        self.assertEqual(
            session.get.call_args.kwargs["headers"],
            {
                "accept": "application/json",
                "Authorization": "YGDweJ9Ucx58ZG8iW4fyTfx5KmSPAz4u",
            },
        )

    def test_duplicate_camera_code_is_rejected(self):
        camera = [{"camera_id": "different-each-time", "camera_code": "CAM-1", "camera_inout": "in"}]
        with self.assertRaisesRegex(ValueError, "duplicate camera_code"):
            camera_api_payload_to_config([
                payload(station_code="A", cameras=camera),
                payload(station_code="B", cameras=camera),
            ])

    def test_unknown_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown camera_inout"):
            camera_api_payload_to_config([
                payload(cameras=[{"camera_code": "CAM-1", "camera_inout": "side"}])
            ])

    def test_initial_failure_is_fatal_then_last_good_is_retained(self):
        session = Mock()
        session.get.side_effect = requests.ConnectionError("offline")
        provider = CameraConfigProvider(base_url="http://camera", session=session)
        with self.assertRaises(requests.ConnectionError):
            provider.load(force=True)

        session.get.side_effect = [response(payload()), requests.ConnectionError("offline")]
        first = provider.load(force=True)
        second = provider.load()
        self.assertIs(first, second)

    def test_uuid_changes_do_not_change_version(self):
        session = Mock()
        first_payload = payload()
        second_payload = payload()
        second_payload["stations"][0]["station_id"] = "another-uuid"
        second_payload["stations"][0]["cameras"][0]["camera_id"] = "another-camera-uuid"
        session.get.side_effect = [response(first_payload), response(second_payload)]
        provider = CameraConfigProvider(base_url="http://camera", session=session)
        first = provider.load(force=True)
        second = provider.load(force=True)
        self.assertEqual(first.version, second.version)


if __name__ == "__main__":
    unittest.main()
