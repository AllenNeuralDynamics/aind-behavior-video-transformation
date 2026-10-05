"""Tests methods in filesystem module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from aind_behavior_video_transformation.filesystem import (
    create_symlinks,
    transform_directory,
)


class TestModule(unittest.TestCase):
    """Tests methods in module"""

    @classmethod
    def setUpClass(cls):
        """Set up class with some mocked os.walk outputs."""

        cls.example_ffmpeg_args = (
            "",
            '-vf "scale=out_color_matrix=bt709:out_range=full:sws_dither=none,'
            "format=yuv420p10le,colorspace=ispace=bt709:all=bt709:dither=none,"
            'scale=out_range=tv:sws_dither=none,format=yuv420p" -c:v libx264 '
            "-preset veryslow -crf 18 -pix_fmt yuv420p -metadata "
            'author="Allen Institute for Neural Dynamics" '
            "-movflags +faststart+write_colr",
        )
        cls.example_src = Path("behavior-videos")
        cls.example_directory_walk_vr_foraging = [
            (
                "behavior-videos",
                ["FaceCamera", "FrontCamera", "SideCamera"],
                [],
            ),
            (
                str(Path("behavior-videos") / "FaceCamera"),
                [],
                ["metadata.csv", "video.mp4"],
            ),
            (
                str(Path("behavior-videos") / "FrontCamera"),
                [],
                ["metadata.csv", "video.mp4"],
            ),
            (
                str(Path("behavior-videos") / "SideCamera"),
                [],
                ["metadata.csv", "video.mp4"],
            ),
        ]
        cls.example_directory_walk_chronic_ephys = [
            ("behavior-videos", ["TopCamera"], []),
            (
                str(Path("behavior-videos") / "TopCamera"),
                [],
                [
                    "TopCamera_2025-05-13T19-00-00.csv",
                    "TopCamera_2025-05-13T19-00-00.mp4",
                    "TopCamera_2025-05-13T20-00-00.csv",
                    "TopCamera_2025-05-13T20-00-00.mp4",
                ],
            ),
        ]

    @patch("aind_behavior_video_transformation.filesystem.walk")
    @patch("pathlib.Path.mkdir")
    @patch("aind_behavior_video_transformation.filesystem.symlink")
    def test_transform_directory(
        self,
        mock_symlink: MagicMock,
        mock_mkdir: MagicMock,
        mock_walk: MagicMock,
    ):
        """Tests transform_directory"""

        mock_walk.return_value = self.example_directory_walk_vr_foraging
        job_in_dir_path = self.example_src
        overrides = dict()
        ffmpeg_arg_set = self.example_ffmpeg_args
        file_filter = None
        job_out_dir_path = Path("output_directory")
        expected_convert_video_args = [
            (
                Path("behavior-videos") / "FaceCamera" / "video.mp4",
                Path("output_directory") / "FaceCamera",
                self.example_ffmpeg_args,
            ),
            (
                Path("behavior-videos") / "FrontCamera" / "video.mp4",
                Path("output_directory") / "FrontCamera",
                self.example_ffmpeg_args,
            ),
            (
                Path("behavior-videos") / "SideCamera" / "video.mp4",
                Path("output_directory") / "SideCamera",
                self.example_ffmpeg_args,
            ),
        ]
        expected_symlink_args = [
            (
                Path("behavior-videos") / "FaceCamera" / "metadata.csv",
                Path("output_directory") / "FaceCamera" / "metadata.csv",
            ),
            (
                Path("behavior-videos") / "FrontCamera" / "metadata.csv",
                Path("output_directory") / "FrontCamera" / "metadata.csv",
            ),
            (
                Path("behavior-videos") / "SideCamera" / "metadata.csv",
                Path("output_directory") / "SideCamera" / "metadata.csv",
            ),
        ]
        actual_convert_video_args, actual_symlink_args = transform_directory(
            job_in_dir_path,
            job_out_dir_path,
            ffmpeg_arg_set,
            overrides,
            file_filter,
        )
        self.assertEqual(
            expected_convert_video_args, actual_convert_video_args
        )
        self.assertEqual(expected_symlink_args, actual_symlink_args)
        # Discovery must not touch the filesystem: every node of a
        # partitioned job runs it, so creating links here would have them
        # all racing to create the same ones.
        mock_symlink.assert_not_called()
        mock_mkdir.assert_has_calls(
            [
                call(parents=True, exist_ok=True),
                call(parents=True, exist_ok=True),
                call(parents=True, exist_ok=True),
            ]
        )

    @patch("aind_behavior_video_transformation.filesystem.walk")
    @patch("pathlib.Path.mkdir")
    @patch("aind_behavior_video_transformation.filesystem.symlink")
    def test_transform_directory_with_file_filter(
        self,
        mock_symlink: MagicMock,
        mock_mkdir: MagicMock,
        mock_walk: MagicMock,
    ):
        """Tests transform_directory with filter_filter set"""

        mock_walk.return_value = self.example_directory_walk_chronic_ephys
        job_in_dir_path = self.example_src
        overrides = dict()
        ffmpeg_arg_set = self.example_ffmpeg_args
        file_filter = ".*2025-05-13T20-00-00.*"
        job_out_dir_path = Path("output_directory")
        expected_convert_video_args = [
            (
                Path("behavior-videos")
                / "TopCamera"
                / "TopCamera_2025-05-13T20-00-00.mp4",
                Path("output_directory") / "TopCamera",
                self.example_ffmpeg_args,
            )
        ]
        expected_symlink_args = [
            (
                Path("behavior-videos")
                / "TopCamera"
                / "TopCamera_2025-05-13T20-00-00.csv",
                Path("output_directory")
                / "TopCamera"
                / "TopCamera_2025-05-13T20-00-00.csv",
            )
        ]
        actual_convert_video_args, actual_symlink_args = transform_directory(
            job_in_dir_path,
            job_out_dir_path,
            ffmpeg_arg_set,
            overrides,
            file_filter,
        )
        self.assertEqual(
            expected_convert_video_args, actual_convert_video_args
        )
        self.assertEqual(expected_symlink_args, actual_symlink_args)
        mock_symlink.assert_not_called()
        mock_mkdir.assert_has_calls([call(parents=True, exist_ok=True)])

    def test_create_symlinks(self):
        """Tests create_symlinks makes the requested links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            src_path = temp_path / "metadata.csv"
            src_path.write_text("data")
            out_path = temp_path / "out" / "metadata.csv"

            create_symlinks([(src_path, out_path)])

            self.assertTrue(out_path.is_symlink())
            self.assertEqual("data", out_path.read_text())

    def test_create_symlinks_skips_existing(self):
        """Tests an existing destination is skipped, not an error.

        Reruns of a job, and dangling links left by an earlier run, must
        not abort the whole partition.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            src_path = temp_path / "metadata.csv"
            src_path.write_text("data")
            out_path = temp_path / "metadata_link.csv"

            create_symlinks([(src_path, out_path)])
            # Second call must be a no-op rather than FileExistsError
            create_symlinks([(src_path, out_path)])
            self.assertTrue(out_path.is_symlink())

            # A dangling link reads as absent to exists(), so it is
            # checked with is_symlink() first
            src_path.unlink()
            self.assertFalse(out_path.exists())
            create_symlinks([(src_path, out_path)])
            self.assertTrue(out_path.is_symlink())


if __name__ == "__main__":
    unittest.main()
