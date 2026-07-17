import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snapshot_api import load_camera_stream_configs


class SnapshotApiConfigTests(unittest.TestCase):
    def test_default_stream_path_uses_live_cam_id_profile_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "camera_groups.yaml"
            config_path.write_text(
                """
relay_base_url: rtsp://relay:8554
default_stream_profile: main
camera_groups:
  2927D:
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
""".lstrip(),
                encoding="utf-8",
            )

            configs = load_camera_stream_configs(str(config_path))

            self.assertEqual(configs["2927D-03"].stream_url, "rtsp://relay:8554/live/2927D-03/main")
            self.assertEqual(configs["2927D-04"].stream_url, "rtsp://relay:8554/live/2927D-04/main")
            self.assertEqual(configs["2927D-02"].stream_url, "rtsp://relay:8554/live/2927D-02/main")
            self.assertEqual(configs["2927D-05"].stream_url, "rtsp://relay:8554/live/2927D-05/main")
            self.assertNotIn("DKV3_2927D", configs)
            self.assertNotIn("DKV3_2927D_02", configs)


if __name__ == "__main__":
    unittest.main()
