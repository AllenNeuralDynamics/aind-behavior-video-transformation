"""Module to handle transforming behavior videos"""

from enum import Enum
import logging
import subprocess
import sys
from time import time
from pathlib import Path

from pydantic import Field

from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)


class FfmpegParams(str, Enum):
    DEFAULT = "add ffmpeg parameters here"
    USE_CASE_1 = "add ffmpeg parameters here"
    USE_CASE_2 = "add ffmpeg parameters here"


class CompressionSettings(BasicJobSettings):
    """BehaviorJob settings. Inherits both fields input_source and
    output_directory from BasicJobSettings."""

    ffmpeg_params: FfmpegParams = Field(
        default=FfmpegParams.DEFAULT,
        description="Parameters added to ffmpeg compression"
    )  # Choose among FfmegParams Enum or provide your own string.


class BehaviorVideoJob(GenericEtl[CompressionSettings]):
    """Main class to handle behavior video transformations"""

    def run_job(self) -> JobResponse:
        """
        Main public method to run the compression job
        Returns
        -------
        JobResponse
          Information about the job that can be used for metadata downstream.

        """
        job_start_time = time()

        for video in self.job_settings.input_source.glob("*.mp4"):
            # Build command
            ffmpeg_command = ['ffmpeg']
            ffmpeg_command.extend(['i', self.job_settings.input_source / video.name])
            ffmpeg_command.extend(self.job_settings.ffmpeg_params.split(' '))
            ffmpeg_command.append(self.job_settings.output_directory / video.name)

            # Run command in subprocess
            process = subprocess.Popen(ffmpeg_command, stderr=subprocess.PIPE)
            for line in process.stderr:
                print(line.decode('utf-8').strip())
            process.wait()
            if process.returncode == 0:
                print("FFmpeg process completed successfully")
            else:
                print(f"Error in FFmpeg process")

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
        job_settings = CompressionSettings.model_validate_json(cli_args.job_settings)
    elif cli_args.config_file is not None:
        job_settings = CompressionSettings.from_config_file(cli_args.config_file)
    else:
        # Construct settings from env vars
        job_settings = CompressionSettings(
            input_source=Path("some_path"),
            output_directory=Path("some_other_path"),
        )
    job = BehaviorVideoJob(job_settings=job_settings)
    job_response = job.run_job()
    logging.info(job_response.model_dump_json())
