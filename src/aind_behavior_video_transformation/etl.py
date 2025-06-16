"""Module that defines the ETL class for behavior video transformations."""

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import time
from typing import List, Optional, Tuple, Union

from pydantic import Field

from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)
from aind_behavior_video_transformation.filesystem import (
    build_overrides_dict,
    transform_directory,
)
from aind_behavior_video_transformation.transform_videos import (
    CompressionRequest,
    convert_video,
)


class BehaviorVideoJobSettings(BasicJobSettings):
    """
    BehaviorJob settings. Inherits both fields input_source and
    output_directory from BasicJobSettings.
    """

    compression_requested: CompressionRequest = Field(
        default_factory=CompressionRequest,
        description="Compression requested for video files.",
    )
    video_specific_compression_requests: Optional[
        List[Tuple[Union[Path, str], CompressionRequest]]
    ] = Field(
        default=None,
        description=(
            "Pairs of video files or directories containing videos, and "
            "compression requests that differ from the global compression request."
        ),
    )
    parallel_compression: bool = Field(
        default=True,
        description="Run compression in parallel or sequentially.",
    )
    ffmpeg_thread_cnt: int = Field(
        default=0,
        description="Number of threads per ffmpeg compression job.",
    )


class BehaviorVideoJob(GenericEtl[BehaviorVideoJobSettings]):
    """
    Main class to handle behavior video transformations.

    Processes input videos based on settings and generates the transformed
    videos in the output directory.
    """

    def _run_compression(
        self,
        convert_video_args: list[tuple[Path, Path, tuple[str, str] | None]],
    ) -> None:
        """
        Runs CompressionRequests at the specified paths.
        """
        if not convert_video_args:
            return

        error_traces = []

        if self.job_settings.parallel_compression:
            with ProcessPoolExecutor(max_workers=len(convert_video_args)) as executor:
                jobs = [
                    executor.submit(
                        convert_video,
                        *params,
                        self.job_settings.ffmpeg_thread_cnt,
                    )
                    for params in convert_video_args
                ]
                for job in as_completed(jobs):
                    result = job.result()
                    if isinstance(result, tuple):
                        error_traces.append(result[1])
                    else:
                        logging.info(f"FFmpeg job completed: {result}")
        else:
            for params in convert_video_args:
                result = convert_video(*params, self.job_settings.ffmpeg_thread_cnt)
                if isinstance(result, tuple):
                    error_traces.append(result[1])
                else:
                    logging.info(f"FFmpeg job completed: {result}")

        if error_traces:
            for err in error_traces:
                logging.error(err)
            raise RuntimeError("One or more FFmpeg jobs failed. See error logs.")

    def run_job(self) -> JobResponse:
        """
        Main public method to run the compression job.

        Applies compression transformations and saves outputs.

        Returns
        -------
        JobResponse
            Status, duration message, and any extra data.
        """
        job_start_time = time()

        overrides = build_overrides_dict(
            self.job_settings.video_specific_compression_requests,
            self.job_settings.input_source.resolve(),
        )
        ffmpeg_arg_set = self.job_settings.compression_requested.determine_ffmpeg_arg_set()

        convert_video_args = transform_directory(
            self.job_settings.input_source.resolve(),
            self.job_settings.output_directory.resolve(),
            ffmpeg_arg_set,
            overrides,
        )

        Path(self.job_settings.output_directory).mkdir(exist_ok=True)
        self._run_compression(convert_video_args)

        job_end_time = time()
        return JobResponse(
            status_code=200,
            message=f"Job finished in: {job_end_time - job_start_time:.2f} seconds.",
            data=None,
        )


if __name__ == "__main__":
    parser = get_parser()
    cli_args = parser.parse_args(sys.argv[1:])

    if cli_args.job_settings is not None:
        job_settings = BehaviorVideoJobSettings.model_validate_json(cli_args.job_settings)
    elif cli_args.config_file is not None:
        job_settings = BehaviorVideoJobSettings.from_config_file(cli_args.config_file)
    else:
        job_settings = BehaviorVideoJobSettings(
            input_source=Path("tests/test_video_in_dir"),
            output_directory=Path("tests/test_video_out_dir"),
        )

    job = BehaviorVideoJob(job_settings=job_settings)
    job_response = job.run_job()
    print(job_response.status_code)
    logging.info(job_response.model_dump_json())
