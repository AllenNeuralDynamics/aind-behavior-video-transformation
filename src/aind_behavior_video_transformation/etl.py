"""Module that defines the ETL class for behavior video transformations."""

import logging
import sys
from pathlib import Path
from time import time
from typing import List, Optional, Tuple, Union

from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)
from pydantic import Field

from aind_behavior_video_transformation.filesystem import (
    build_overrides_dict,
    transform_directory,
)
from aind_behavior_video_transformation.transform_videos import (
    CompressionRequest,
)


class BehaviorVideoJobSettings(BasicJobSettings):
    """
    BehaviorJob settings. Inherits both fields input_source and
    output_directory from BasicJobSettings.
    """

    compression_requested: CompressionRequest = Field(
        default=CompressionRequest(),
        description="Compression requested for video files",
    )
    video_specific_compression_requests: Optional[
        List[Tuple[Union[Path, str], CompressionRequest]]
    ] = Field(
        default=None,
        description=(
            "Pairs of video files or directories containing videos, and "
            "compression requests that differ from the global compression "
            "request"
        ),
    )


class BehaviorVideoJob(GenericEtl[BehaviorVideoJobSettings]):
    """
    Main class to handle behavior video transformations.

    This class is responsible for running the compression job on behavior
    videos.  It processes the input videos based on the provided settings and
    generates the transformed videos in the specified output directory.

    Attributes
    ----------
    job_settings : BehaviorVideoJobSettings
        Settings specific to the behavior video job, including input source,
        output directory, and compression requests.

    Methods
    -------
    run_job() -> JobResponse
    """

    def run_job(self) -> JobResponse:
        """
        Main public method to run the compression job.

        Run the compression job for behavior videos.

        This method processes the input videos based on the provided settings,
        applies the necessary compression transformations, and saves the output
        videos to the specified directory. It also handles any specific
        compression requests for individual videos or directories.

        Returns
        -------
        JobResponse
            Contains the status code, a message indicating the job duration,
            and any additional data.
        """
        job_start_time = time()

        video_comp_pairs = (
            self.job_settings.video_specific_compression_requests
        )
        job_out_dir_path = self.job_settings.output_directory.resolve()
        Path(job_out_dir_path).mkdir(exist_ok=True)
        job_in_dir_path = self.job_settings.input_source.resolve()
        overrides = build_overrides_dict(video_comp_pairs, job_in_dir_path)

        ffmpeg_arg_set = (
            self.job_settings.compression_requested.determine_ffmpeg_arg_set()
        )
        transform_directory(
            job_in_dir_path, job_out_dir_path, ffmpeg_arg_set, overrides
        )
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
        job_settings = BehaviorVideoJobSettings.model_validate_json(
            cli_args.job_settings
        )
    elif cli_args.config_file is not None:
        job_settings = BehaviorVideoJobSettings.from_config_file(
            cli_args.config_file
        )
    else:
        # Default settings
        job_settings = BehaviorVideoJobSettings(
            input_source=Path("tests/test_video_in_dir"),
            output_directory=Path("tests/test_video_out_dir"),
        )

    job = BehaviorVideoJob(job_settings=job_settings)
    job_response = job.run_job()
    print(job_response.status_code)

    logging.info(job_response.model_dump_json())
