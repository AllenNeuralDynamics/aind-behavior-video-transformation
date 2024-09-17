"""Tests transform_videos module."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aind_data_transformation.core import JobResponse

from aind_behavior_video_transformation.transform_videos import (
    BehaviorVideoJob,
    JobSettings,
)


class TestJobSettings(unittest.TestCase):
    """Tests methods in JobSettings class"""

    def test_class_constructor(self):
        """Tests basic class constructor from init args"""
        job_settings = JobSettings(
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

        job_settings = JobSettings(
            input_source=Path("some_path"),
            output_directory=Path("some_other_path"),
        )
        etl_job = BehaviorVideoJob(job_settings=job_settings)
        t0 = 1726602472.6364267
        t1 = 1726602475.3568988
        dt = t1 - t0
        mock_time.side_effect = [t0, t1]
        response = etl_job.run_job()

        expected_response = JobResponse(
            status_code=200,
            message=f"Job finished in: {dt}",
            data=None,
        )
        self.assertEqual(expected_response, response)


if __name__ == "__main__":
    unittest.main()
