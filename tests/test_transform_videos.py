"""Tests transform_videos module."""

import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aind_data_transformation.core import JobResponse

from aind_behavior_video_transformation.transform_videos import (
    BehaviorVideoJob,
    CompressionSettings,
    CompressionRequest
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

        INPUT_SOURCE = Path("some_path")
        OUTPUT_DIR = Path("some_other_path")

        # Test 1: No Compression
        job_settings = CompressionSettings(
            input_source=INPUT_SOURCE,
            output_directory=OUTPUT_DIR,
            compression_requested=CompressionRequest.NO_COMPRESSION
        )
        etl_job = BehaviorVideoJob(job_settings=job_settings)
        start_time = time.time()
        response = etl_job.run_job()
        end_time = time.time()
        expected_response = JobResponse(
            status_code=200,
            message=f"Job finished in: {end_time - start_time}",
            data=None,
        )
        self.assertEqual(expected_response, response)

        # Test 2: Default Compression (Gamma Encoding)
        job_settings = CompressionSettings(
            input_source=INPUT_SOURCE,
            output_directory=OUTPUT_DIR,
        )
        etl_job = BehaviorVideoJob(job_settings=job_settings)
        start_time = time.time()
        response = etl_job.run_job()
        end_time = time.time()
        expected_response = JobResponse(
            status_code=200,
            message=f"Job finished in: {end_time - start_time}",
            data=None,
        )
        self.assertEqual(expected_response, response)

        # Test 3: User Defined
        job_settings = CompressionSettings(
            input_source=INPUT_SOURCE,
            output_directory=OUTPUT_DIR,
            compression_requested=CompressionRequest.USER_DEFINED,
            user_ffmpeg_input_options="",
            user_ffmpeg_output_options=""
        )
        etl_job = BehaviorVideoJob(job_settings=job_settings)
        start_time = time.time()
        response = etl_job.run_job()
        end_time = time.time()
        expected_response = JobResponse(
            status_code=200,
            message=f"Job finished in: {end_time - start_time}",
            data=None,
        )
        self.assertEqual(expected_response, response)

        # Test 4: Custom Preset-- No Gamma Encoding
        job_settings = CompressionSettings(
            input_source=INPUT_SOURCE,
            output_directory=OUTPUT_DIR,
            compression_requested=CompressionRequest.NO_GAMMA_ENCODING,
        )
        etl_job = BehaviorVideoJob(job_settings=job_settings)
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
