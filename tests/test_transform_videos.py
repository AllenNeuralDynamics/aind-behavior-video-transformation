"""Tests transform_videos module."""

import logging
import shlex
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from os import symlink, unlink
from pathlib import Path
from unittest.mock import MagicMock, patch

from aind_data_transformation.core import JobResponse

from aind_behavior_video_transformation import (
    BehaviorVideoJob,
    BehaviorVideoJobSettings,
    CompressionEnum,
    CompressionRequest,
)


class TestJobSettings(unittest.TestCase):
    """Tests methods in JobSettings class"""

    def test_class_constructor(self):
        """Tests basic class constructor from init args"""
        job_settings = BehaviorVideoJobSettings(
            input_source=Path("some_path"),
            output_directory=Path("some_other_path"),
        )
        self.assertEqual(Path("some_path"), job_settings.input_source)
        self.assertEqual(
            Path("some_other_path"), job_settings.output_directory
        )


def helper_run_compression_job(job_settings, mock_time):
    """Helper function to run compression job."""
    mock_time.side_effect = [0, 1]
    etl_job = BehaviorVideoJob(job_settings=job_settings)
    return etl_job.run_job()


class TestBehaviorVideoJob(unittest.TestCase):
    """Test methods in BehaviorVideoJob class."""

    test_data_path = Path("tests/test_video_in_dir").resolve()
    dummy_response = JobResponse(
        status_code=200,
        message="Job finished in: 1",
        data=None,
    )
    test_vid_name = "clip.mp4"
    test_vid_path = test_data_path / test_vid_name

    @patch("aind_behavior_video_transformation.etl.time")
    def test_run_job(self, mock_time: MagicMock):
        """Tests run_job method."""
        input_dir = self.test_data_path
        expected_response = self.dummy_response
        test_vid_name = self.test_vid_name
        for compression_enum in [
            CompressionEnum.DEFAULT,
            CompressionEnum.GAMMA_ENCODING,
            CompressionEnum.GAMMA_ENCODING_FIX_COLORSPACE,
            CompressionEnum.NO_GAMMA_ENCODING,
            CompressionEnum.NO_COMPRESSION,
        ]:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                job_settings = BehaviorVideoJobSettings(
                    input_source=input_dir,
                    output_directory=temp_dir,
                    compression_requested=CompressionRequest(
                        compression_enum=compression_enum
                    ),
                    parallel_compression=False,
                )
                response = helper_run_compression_job(job_settings, mock_time)
                self.assertEqual(expected_response, response)
                self.assertTrue(temp_path.joinpath(test_vid_name).exists())

        # User Defined
        with tempfile.TemporaryDirectory() as temp_dir:
            job_settings = BehaviorVideoJobSettings(
                input_source=input_dir,
                output_directory=temp_dir,
                compression_requested=CompressionRequest(
                    compression_enum=CompressionEnum.USER_DEFINED,
                    user_ffmpeg_input_options="-color_trc linear",
                    user_ffmpeg_output_options=(
                        "-c:v libx264 -preset veryfast -crf 40"
                    ),
                ),
            )
            response = helper_run_compression_job(job_settings, mock_time)
            self.assertEqual(expected_response, response)

    @patch("aind_behavior_video_transformation.etl.time")
    def test_run_job_with_data_structure(self, mock_time: MagicMock):
        # Test that data file structure is maintained
        test_vid_path = self.test_vid_path
        test_vid_name = self.test_vid_name
        dummy_response = self.dummy_response
        metadata_file = "metadata.csv"
        if not test_vid_path.is_file():
            raise FileNotFoundError(f"File not found: {test_vid_path}")
        camera_subdirs = [f"camera{i}" for i in range(1, 3)]
        with tempfile.TemporaryDirectory() as in_temp_dir:
            # Prepare input data
            in_path = Path(in_temp_dir)
            camera_in_paths = [in_path / d for d in camera_subdirs]
            for camera_path in camera_in_paths:
                camera_path.mkdir()
                symlink(test_vid_path, camera_path / test_vid_name)
                open(camera_path / metadata_file, "w").close()

            with tempfile.TemporaryDirectory() as out_temp_dir:
                out_path = Path(out_temp_dir)
                job_settings = BehaviorVideoJobSettings(
                    input_source=in_path,
                    output_directory=out_path,
                    compression_requested=CompressionRequest(
                        compression_enum=CompressionEnum.DEFAULT,
                    ),
                    parallel_compression=True,
                    ffmpeg_thread_cnt=4,
                )
                response = helper_run_compression_job(job_settings, mock_time)
                self.assertEqual(dummy_response, response)

                for d in camera_subdirs:
                    self.assertTrue(out_path.joinpath(d).exists())
                    self.assertTrue(
                        out_path.joinpath(d, test_vid_name).exists()
                    )
                    self.assertTrue(
                        out_path.joinpath(d, metadata_file).exists()
                    )

            override_dir = camera_subdirs[0]
            override_specifications = [
                override_dir,
                in_path / override_dir,
                in_path / override_dir / test_vid_name,
            ]
            for override_spec in override_specifications:
                with tempfile.TemporaryDirectory() as out_temp_dir:
                    out_path = Path(out_temp_dir)
                    job_settings = BehaviorVideoJobSettings(
                        input_source=in_path,
                        output_directory=out_path,
                        compression_requested=CompressionRequest(
                            compression_enum=CompressionEnum.DEFAULT,
                        ),
                        video_specific_compression_requests=[
                            (
                                override_spec,
                                CompressionRequest(
                                    compression_enum=CompressionEnum.NO_COMPRESSION  # noqa E501
                                ),
                            )
                        ],
                    )
                    response = helper_run_compression_job(
                        job_settings, mock_time
                    )
                    self.assertEqual(dummy_response, response)

                    for d in camera_subdirs:
                        self.assertTrue(out_path.joinpath(d).exists())
                        self.assertTrue(
                            out_path.joinpath(d, test_vid_name).exists()
                        )
                    overriden_out = out_path / override_dir / test_vid_name
                    self.assertTrue(overriden_out.is_symlink())

    @patch("aind_behavior_video_transformation.etl.time")
    def test_run_job_missing_colorspace(self, mock_time: MagicMock):
        """Tests run_job method with missing colorspace."""
        expected_response = self.dummy_response
        test_vid_name = self.test_vid_name
        test_vid_path = self.test_vid_path
        with tempfile.TemporaryDirectory() as temp_in_dir:
            temp_in_path = Path(temp_in_dir)
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-v",
                "warning",
                "-i",
            ]
            # Strip colorspace metadata from test video
            temp_vid_path = temp_in_path / test_vid_name
            ffmpeg_cmd.append(str(test_vid_path))
            ffmpeg_cmd.extend(
                shlex.split(
                    "-c:v copy -bsf:v h264_metadata=colour_primaries=2:"
                    "transfer_characteristics=2:matrix_coefficients=2"
                )
            )
            ffmpeg_cmd.append(str(temp_vid_path))
            subprocess.run(ffmpeg_cmd, check=True)
            compression_enum = CompressionEnum.GAMMA_ENCODING_FIX_COLORSPACE
            with tempfile.TemporaryDirectory() as temp_out_dir:
                temp_out_path = Path(temp_out_dir)
                job_settings = BehaviorVideoJobSettings(
                    input_source=temp_in_path,
                    output_directory=temp_out_dir,
                    compression_requested=CompressionRequest(
                        compression_enum=compression_enum
                    ),
                )
                response = helper_run_compression_job(job_settings, mock_time)
                self.assertEqual(expected_response, response)
                self.assertTrue(temp_out_path.joinpath(test_vid_name).exists())

    @contextmanager
    def capture_logs(self):
        """
        Context manager that creates a temporary log file
        and configures logging to use it.
        """
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_log:
            # Configure logging to write to our temporary file
            file_handler = logging.FileHandler(temp_log.name)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logging.getLogger().addHandler(file_handler)
            logging.getLogger().setLevel(logging.DEBUG)

            try:
                yield temp_log.name
            finally:
                # Clean up: restore original handlers and remove temp file
                file_handler.close()
                unlink(temp_log.name)

    @patch("aind_behavior_video_transformation.etl.time")
    def test_serial_log(self, mock_time: MagicMock):
        """
        Test errors in parallel execution are logged
        sequentially. The length of the error message
        is known to be 9 lines. So non-overlap is tested
        by checking 'ERROR' does not occur within 9 lines
        of each instance of 'ERROR'.
        """

        faulty_req = CompressionRequest(
            compression_enum=CompressionEnum.USER_DEFINED,
            user_ffmpeg_input_options="invalid input args",
            user_ffmpeg_output_options="invalid output args",
        )

        if not self.test_vid_path.is_file():
            raise FileNotFoundError(f"File not found: {self.test_vid_path}")
        camera_subdirs = [f"camera{i}" for i in range(1, 3)]
        with tempfile.TemporaryDirectory() as in_temp_dir:
            # Prepare input data
            in_path = Path(in_temp_dir)
            camera_in_paths = [in_path / d for d in camera_subdirs]
            for camera_path in camera_in_paths:
                camera_path.mkdir()
                symlink(self.test_vid_path, camera_path / self.test_vid_name)

            with tempfile.TemporaryDirectory() as out_temp_dir:
                out_path = Path(out_temp_dir)
                job_settings = BehaviorVideoJobSettings(
                    input_source=in_path,
                    output_directory=out_path,
                    compression_requested=faulty_req,
                    parallel_compression=True,
                )
                with self.capture_logs() as log_file:
                    self.assertRaises(
                        RuntimeError,
                        helper_run_compression_job,
                        job_settings,
                        mock_time,
                    )
                    error_blocks = []
                    with open(log_file, "r") as f:
                        for line_num, line in enumerate(f, 1):
                            if "ERROR" in line:
                                error_blocks.append(line_num)

                    for i in range(len(error_blocks) - 1):
                        error_diff = error_blocks[i + 1] - error_blocks[i]
                        self.assertTrue(error_diff >= 9)


if __name__ == "__main__":
    unittest.main()
