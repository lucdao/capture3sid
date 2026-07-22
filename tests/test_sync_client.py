import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture3sid.tests.sync_client_test import DEFAULT_SINCE, sync_once


def args_for(tmp, *, mode):
    return argparse.Namespace(
        base_url="http://sync.local",
        token="secret",
        output_root=str(Path(tmp) / "downloads"),
        state_file=str(Path(tmp) / "state.json"),
        since=None,
        mode=mode,
        location="",
        camera_group="",
        limit=100,
        timeout_sec=5.0,
        dry_run=False,
    )


def changes_response(next_since):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "items": [],
        "has_more": False,
        "next_cursor": None,
        "next_since": next_since,
        "upper_bound": next_since,
    }
    return response


class SyncClientModeTests(unittest.TestCase):
    def test_modes_use_independent_checkpoints_and_preserve_legacy_capture_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            state_path.write_text(
                json.dumps({
                    "since": "2026-07-01T01:00:00Z",
                    "location": "",
                    "camera_group": "",
                    "base_url": "http://sync.local",
                }),
                encoding="utf-8",
            )
            session = Mock()
            session.get.side_effect = [
                changes_response("2026-07-02T01:00:00Z"),
                changes_response("2026-07-03T01:00:00Z"),
            ]

            with patch("sync_client_test.requests.Session", return_value=session):
                sync_once(args_for(tmp, mode="document"))
                sync_once(args_for(tmp, mode="capture"))

            first_params = session.get.call_args_list[0].kwargs["params"]
            second_params = session.get.call_args_list[1].kwargs["params"]
            self.assertEqual(first_params["mode"], "document")
            self.assertEqual(first_params["since"], DEFAULT_SINCE)
            self.assertEqual(second_params["mode"], "capture")
            self.assertEqual(second_params["since"], "2026-07-01T01:00:00Z")

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["version"], 2)
            self.assertEqual(state["streams"]["document:"]["since"], "2026-07-02T01:00:00Z")
            self.assertEqual(state["streams"]["capture:"]["since"], "2026-07-03T01:00:00Z")

    def test_location_is_part_of_stream_checkpoint_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Mock()
            session.get.side_effect = [
                changes_response("2026-07-02T01:00:00Z"),
                changes_response("2026-07-02T02:00:00Z"),
            ]
            first_args = args_for(tmp, mode="document")
            first_args.location = "2927D"
            second_args = args_for(tmp, mode="document")
            second_args.location = "2903V"

            with patch("sync_client_test.requests.Session", return_value=session):
                sync_once(first_args)
                sync_once(second_args)

            state = json.loads((Path(tmp) / "state.json").read_text(encoding="utf-8"))
            self.assertIn("document:2927D", state["streams"])
            self.assertIn("document:2903V", state["streams"])
            self.assertEqual(session.get.call_args_list[1].kwargs["params"]["since"], DEFAULT_SINCE)


if __name__ == "__main__":
    unittest.main()
