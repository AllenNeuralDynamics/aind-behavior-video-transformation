"""Module to handle transforming behavior videos"""

import logging
import shlex
import subprocess
import sys
from enum import Enum
from os import symlink
from pathlib import Path
from time import time
from typing import Optional

from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)
from pydantic import Field


class CompressionRequest(Enum):
    """
    Enum class to define different types of compression requests.
    Details of requests found in FfmpegParamSets.
    """

    DEFAULT = "gamma"
    GAMMA_ENCODING = "gamma"
    NO_GAMMA_ENCODING = "no gamma"
    USER_DEFINED = "user defined"
    NO_COMPRESSION = "no compression"


class InputFfmpegParams(Enum):
    """
    Input parameter set referenced inside FfmpegParamSets
    """

    NONE = ""


class OutputFfmpegParams(Enum):
    """
    Output parameter set referenced inside FfmpegParamSets
    """

    GAMMA_ENCODING = (
        "-vf "
        '"scale=out_color_matrix=bt709:out_range=full:sws_dither=none,'
        "colorspace=ispace=bt709:all=bt709:dither=none,"
        'scale=out_range=tv:sws_dither=none,format=yuv420p" -c:v libx264 '
        "-preset veryslow -crf 18 -pix_fmt yuv420p "
        '-metadata author="Allen Institute for Neural Dyamics" '
        "-movflags +faststart+write_colr"
    )
    NO_GAMMA_ENCODING = (
        "-vf "
        '"scale=out_range=tv:sws_dither=none,format=yuv420p" -c:v libx264 '
        "-preset veryslow -crf 18 -pix_fmt yuv420p "
        '-metadata author="Allen Institute for Neural Dyamics" '
        "-movflags +faststart+write_colr"
    )
    NONE = ""


class FfmpegParamSets(Enum):
    """
    Define different ffmpeg params to be used for video compression
    Two-tuple with first element as input params and second element as output
    params.

    Default takes 10 bit input and converts to 8 bit output after doing gamma
    correction.

    Assumes input has linear transfer characteristic, and pixel format
    yuv420p10le. Output is yuv420p standard range
    """

    GAMMA_ENCODING = (
        InputFfmpegParams.NONE,
        OutputFfmpegParams.GAMMA_ENCODING,
    )
    NO_GAMMA_ENCODING = (
        InputFfmpegParams.NONE,
        OutputFfmpegParams.NO_GAMMA_ENCODING,
    )


class CompressionSettings(BasicJobSettings):
    """BehaviorJob settings. Inherits both fields input_source and
    output_directory from BasicJobSettings."""

    compression_requested: CompressionRequest = Field(
        default=CompressionRequest.DEFAULT,
        description="Params to pass to ffmpeg command",
    )  # Choose among FfmegParams Enum or provide your own string.
    user_ffmpeg_input_options: Optional[str] = Field(
        default=None, description="User defined ffmpeg input options"
    )
    user_ffmpeg_output_options: Optional[str] = Field(
        default=None, description="User defined ffmpeg output options"
    )


class BehaviorVideoJob(GenericEtl[CompressionSettings]):
    """Main class to handle behavior video transformations"""

    def convert_video(self, video_path: Path) -> None:
        """
        Convert video to a different format
        Parameters
        ----------
        video_path : Path
            Path to the video file to be converted
        """

        out_path = (
            self.job_settings.output_directory / f"{video_path.stem}.mp4"
        )  # noqa: E501
        # Pydantic validation ensures this is a 'CompressionRequest' value.
        compression_requested = self.job_settings.compression_requested

        # Trivial Case, do nothing
        if compression_requested == CompressionRequest.NO_COMPRESSION:
            symlink(str(video_path), str(out_path))
            return

        # Compression Cases corresponding to each CompressionRequest.
        # Each case sets input/output args to pass into ffmpeg command.
        # If user defined, use the user defined options.
        if compression_requested == CompressionRequest.USER_DEFINED:
            input_args = self.job_settings.user_ffmpeg_input_options
            output_args = self.job_settings.user_ffmpeg_output_options
        # In all other cases, the options are defined in FfmpegParamSets.
        else:
            # If default, set compression to gamma
            if compression_requested == CompressionRequest.DEFAULT:
                compression_preset = CompressionRequest.GAMMA_ENCODING
            else:
                compression_preset = compression_requested
            param_set = FfmpegParamSets[compression_preset.name].value
            input_args = param_set[0].value
            output_args = param_set[1].value

        logging.info(f"{input_args=}")
        logging.info(f"{output_args=}")

        ffmpeg_command = ["ffmpeg", "-y", "-v", "info"]
        if input_args:
            ffmpeg_command.extend(shlex.split(input_args))
        ffmpeg_command.extend(["-i", str(video_path)])
        if output_args:
            ffmpeg_command.extend(shlex.split(output_args))
        ffmpeg_command.append(str(out_path))

        # Run command in subprocess
        try:
            result = subprocess.run(  # noqa: F841
                ffmpeg_command,
                check=True,
                stderr=subprocess.PIPE,  # Capture stderr
                text=True,  # Get output as string, not bytes
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running FFmpeg: {e.stderr}")

        return

    def run_job(self) -> JobResponse:
        """
        Main public method to run the compression job
        Returns
        -------
        JobResponse
            Information about the job that can be used for metadata downstream.

        """
        job_start_time = time()
        input_dir = self.job_settings.input_source
        video_files = [
            f
            for f in input_dir.iterdir()
            if f.suffix in (".mp4", ".avi", ".mov", ".mkv")
        ]
        for video_file in video_files:
            self.convert_video(video_file)

        job_end_time = time()
        return JobResponse(
            status_code=200,
            message=f"Job finished in: {job_end_time-job_start_time}",
            data=None,
        )


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    parser = get_parser()
    cli_args = parser.parse_args(sys_args)
    if cli_args.job_settings is not None:
        job_settings = CompressionSettings.model_validate_json(
            cli_args.job_settings
        )
    elif cli_args.config_file is not None:
        job_settings = CompressionSettings.from_config_file(
            cli_args.config_file
        )
    else:
        # Construct settings from env vars
        job_settings = CompressionSettings(
            input_source=Path("tests/test_video_in_dir"),
            output_directory=Path("tests/test_video_out_dir"),
            compression_requested=CompressionRequest.NO_GAMMA_ENCODING,
        )

    job = BehaviorVideoJob(job_settings=job_settings)
    job_response = job.run_job()
    print(job_response.status_code)

    logging.info(job_response.model_dump_json())
