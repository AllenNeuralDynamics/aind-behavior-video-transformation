"""Tests transform_videos module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aind_data_transformation.core import JobResponse

from aind_behavior_video_transformation.transform_videos import (
    BehaviorVideoJob,
    CompressionRequest,
    CompressionSettings,
)


class TestJobSettings(unittest.TestCase):
    """Tests methods in JobSettings class"""

    def test_class_constructor(self):
        """Tests basic class constructor from init args"""
        job_settings = CompressionSettings(
            input_source=Path("some_path"),
            output_directory=Path("some_other_path"),
        )
        self.assertEqual(Path("some_path"), job_settings.input_source)
        self.assertEqual(
            Path("some_other_path"), job_settings.output_directory
        )


class TestBehaviorVideoJob(unittest.TestCase):
    """Test methods in BehaviorVideoJob class."""

    # NOTE:
    # Test suite does not run yet.
    # Resolving lint errors first.
    def helper_run_compression_job(self, job_settings, mock_time):
        """Helper function to run compression job."""
        mock_time.side_effect = [0, 1]
        etl_job = BehaviorVideoJob(job_settings=job_settings)
        response = etl_job.run_job()
        expected_response = JobResponse(
            status_code=200,
            message="Job finished in: 1",
            data=None,
        )
        self.assertEqual(expected_response, response)

    @patch("aind_behavior_video_transformation.transform_videos.time")
    def test_run_job(self, mock_time: MagicMock):
        """Tests run_job method."""
        INPUT_SOURCE = Path("tests/test_video_in_dir")
        for compression_setting in [
            CompressionRequest.DEFAULT,
            CompressionRequest.GAMMA_ENCODING,
            CompressionRequest.NO_GAMMA_ENCODING,
            CompressionRequest.NO_COMPRESSION,
        ]:
            with tempfile.TemporaryDirectory() as temp_dir:
                job_settings = CompressionSettings(
                    input_source=INPUT_SOURCE,
                    output_directory=temp_dir,
                    compression_requested=compression_setting,
                )
                self.helper_run_compression_job(job_settings, mock_time)

        # User Defined
        with tempfile.TemporaryDirectory() as temp_dir:
            job_settings = CompressionSettings(
                input_source=INPUT_SOURCE,
                output_directory=temp_dir,
                compression_requested=CompressionRequest.USER_DEFINED,
                user_ffmpeg_input_options="",
                user_ffmpeg_output_options="-libx264 -preset veryfast -crf 40",
            )
            self.helper_run_compression_job(job_settings, mock_time)


if __name__ == "__main__":
    unittest.main()
