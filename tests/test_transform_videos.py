"""Tests transform_videos module."""

import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aind_data_transformation.core import JobResponse

from aind_behavior_video_transformation.transform_videos import (
    BehaviorVideoJob,
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

    @patch("aind_behavior_video_transformation.transform_videos.time")
    def test_run_job(self, mock_time: MagicMock):
        """Tests run_job method."""

        job_settings = CompressionSettings(
            input_source=Path("some_path"),
            output_directory=Path("some_other_path"),
        )

        etl_job = BehaviorVideoJob(job_settings=job_settings)

        # Okay, all I need here is an example video
        # and an example ffmpeg string. 

        start_time = time.time()
        response = etl_job.run_job()
        end_time = time.time()

        expected_response = JobResponse(
            status_code=200,
            message=f"Job finished in: {end_time - start_time}",
            data=None,
        )
        self.assertEqual(expected_response, response)


if __name__ == "__main__":
    unittest.main()
