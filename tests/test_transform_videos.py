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

import pyarrow.parquet as pq
from aind_data_transformation.core import JobResponse

from aind_behavior_video_transformation import (
    BehaviorVideoJob,
    BehaviorVideoJobSettings,
    CompressionEnum,
    CompressionRequest,
)
from aind_behavior_video_transformation import transform_videos
from aind_behavior_video_transformation.etl import _format_conversion_error
from aind_behavior_video_transformation.transform_videos import convert_video


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


def _nb_frames(video: Path) -> int:
    """Count the frames in a video."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(out.stdout)


class TestPreviews(unittest.TestCase):
    """Tests the preview, preview metadata and poster beside each video."""

    test_vid_path = Path("tests/test_video_in_dir/clip.mp4").resolve()
    clip_frames = 501
    derivative_names = {
        "preview.mp4",
        "preview_metadata.parquet",
        "poster.jpg",
    }

    def setUp(self):
        """Make temporary input and output directories."""
        in_dir = tempfile.TemporaryDirectory()
        out_dir = tempfile.TemporaryDirectory()
        self.addCleanup(in_dir.cleanup)
        self.addCleanup(out_dir.cleanup)
        self.in_path = Path(in_dir.name)
        self.out_path = Path(out_dir.name)

    def _camera(self, video_name="video.mp4", metadata_rows=None):
        """Make a camera folder holding the clip and maybe a metadata.csv."""
        camera = self.in_path / "BodyCamera"
        camera.mkdir()
        symlink(self.test_vid_path, camera / video_name)
        if metadata_rows is not None:
            (camera / "metadata.csv").write_text(
                "ReferenceTime,CameraFrameNumber\n"
                + "".join(f"{n / 500},{n}\n" for n in range(metadata_rows))
            )
        return self.out_path / "BodyCamera"

    def _run(self, compression_enum=CompressionEnum.DEFAULT, **settings):
        """Run a job over the input directory."""
        job_settings = BehaviorVideoJobSettings(
            input_source=self.in_path,
            output_directory=self.out_path,
            compression_requested=CompressionRequest(
                compression_enum=compression_enum
            ),
            **settings,
        )
        with patch("aind_behavior_video_transformation.etl.time") as mock:
            return helper_run_compression_job(job_settings, mock)

    def _names(self, folder: Path) -> set:
        """Names of the files in a folder."""
        return {p.name for p in folder.iterdir()}

    def test_spec_layout_gets_every_derivative(self):
        """video.mp4 with a metadata.csv gets the spec's three files."""
        out = self._camera(metadata_rows=self.clip_frames)
        self._run()
        self.assertTrue(self.derivative_names <= self._names(out))
        rows = pq.read_table(out / "preview_metadata.parquet").num_rows
        self.assertEqual(_nb_frames(out / "preview.mp4"), rows)
        self.assertLess(rows, self.clip_frames)

    def test_other_names_take_the_video_stem(self):
        """A video not named video.* names its derivatives after itself.

        metadata.csv belongs to video.* alone, so no preview metadata.
        """
        out = self._camera("clip.mp4", metadata_rows=self.clip_frames)
        self._run()
        names = self._names(out)
        self.assertTrue({"clip_preview.mp4", "clip_poster.jpg"} <= names)
        self.assertFalse(any(n.endswith(".parquet") for n in names))

    def test_no_metadata_csv_still_gets_preview_and_poster(self):
        """Only the preview metadata needs metadata.csv."""
        out = self._camera()
        self._run()
        self.assertEqual(
            {"video.mp4", "preview.mp4", "poster.jpg"}, self._names(out)
        )

    def test_write_previews_false_writes_only_the_video(self):
        """write_previews=False turns every derivative off."""
        out = self._camera(metadata_rows=self.clip_frames)
        self._run(write_previews=False)
        self.assertEqual({"video.mp4", "metadata.csv"}, self._names(out))

    def test_no_gamma_encoding_gets_no_poster(self):
        """Its chain needs no known transfer, which the poster does."""
        out = self._camera(metadata_rows=self.clip_frames)
        self._run(CompressionEnum.NO_GAMMA_ENCODING)
        names = self._names(out)
        self.assertIn("preview.mp4", names)
        self.assertNotIn("poster.jpg", names)

    def test_no_compression_gets_no_derivatives(self):
        """A symlinked video is not encoded, so it gets no derivatives."""
        out = self._camera(metadata_rows=self.clip_frames)
        self._run(CompressionEnum.NO_COMPRESSION)
        self.assertFalse(self.derivative_names & self._names(out))

    def test_metadata_row_mismatch_fails_the_job(self):
        """Preview metadata rows must line up with the preview's frames."""
        self._camera(metadata_rows=self.clip_frames - 1)
        with self.assertRaisesRegex(RuntimeError, "rows"):
            self._run()

    def test_metadata_without_reference_time_fails_the_job(self):
        """Its producer has to hear that the metadata is unusable."""
        self._camera()
        (self.in_path / "BodyCamera" / "metadata.csv").write_text(
            "".join(f"{n}\n" for n in range(self.clip_frames + 1))
        )
        with self.assertRaisesRegex(RuntimeError, "ReferenceTime"):
            self._run()

    def test_thread_count_reaches_every_preset_codec(self):
        """aind-video-utils caps the decoder, filters and each encoder."""
        self._camera()
        with patch.object(
            transform_videos,
            "transcode_video",
            wraps=transform_videos.transcode_video,
        ) as spy:
            self._run(ffmpeg_thread_cnt=2)
        self.assertEqual(2, spy.call_args.kwargs["threads"])


class TestUserDefinedThreads(unittest.TestCase):
    """Tests ffmpeg_thread_cnt on user-defined encodes."""

    @patch("aind_behavior_video_transformation.transform_videos.subprocess")
    def test_threads_bind_the_decoder_and_encoder(self, mock_subprocess):
        """A count in the user's own options still wins."""
        convert_video(
            Path("in.avi"),
            Path("out"),
            ("", "-c:v libx264 -threads 1"),
            ffmpeg_thread_cnt=4,
        )
        cmd = mock_subprocess.run.call_args.args[0]
        before_input, after_input = (
            cmd[: cmd.index("-i")],
            cmd[cmd.index("-i") + 2:],
        )
        self.assertEqual(
            ["-filter_threads", "4", "-threads", "4"], before_input[-4:]
        )
        self.assertEqual(
            ["-threads", "4", "-c:v", "libx264", "-threads", "1"],
            after_input[:-1],
        )


class TestFormatConversionError(unittest.TestCase):
    """Tests _format_conversion_error"""

    def test_decodes_bytes_stderr(self):
        """aind-video-utils raises with stderr as bytes."""
        exc = subprocess.CalledProcessError(
            1, ["ffmpeg"], stderr=b"[error] bad\n"
        )
        message = _format_conversion_error(Path("video.mp4"), exc)
        self.assertIn("[error] bad\n--- end stderr ---", message)

    def test_formats_other_errors(self):
        """Frame-count and metadata failures carry no ffmpeg command."""
        exc = RuntimeError("lost frames")
        message = _format_conversion_error(Path("video.mp4"), exc)
        self.assertEqual(
            "Conversion failed for video.mp4\nlost frames", message
        )


if __name__ == "__main__":
    unittest.main()
