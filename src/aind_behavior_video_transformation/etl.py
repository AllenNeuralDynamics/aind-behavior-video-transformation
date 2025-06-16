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
    CompressionEnum,
    CompressionRequest,
    convert_video,
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
    """

    def _run_compression(  # noqa: C901
        self,
        convert_video_args: list[tuple[Path, Path, tuple[str, str] | None]],
    ) -> None:
        """
        Runs CompressionRequests at the specified paths.
        If a compression job fails, retries with no compression setting.
        """
        if not convert_video_args:
            return

        no_compression_request = CompressionRequest(
            compression_enum=CompressionEnum.NO_COMPRESSION
        )
        no_compression_args = no_compression_request.determine_ffmpeg_arg_set()

        if self.job_settings.parallel_compression:
            num_jobs = len(convert_video_args)
            with ProcessPoolExecutor(max_workers=num_jobs) as executor:
                jobs = [
                    executor.submit(
                        convert_video,
                        *params,
                        self.job_settings.ffmpeg_thread_cnt,
                    )
                    for params in convert_video_args
                ]

                failed_jobs = []
                for job in as_completed(jobs):
                    result = job.result()
                    if isinstance(result, tuple):
                        job_index = jobs.index(job)
                        original_params = convert_video_args[job_index]
                        failed_jobs.append(original_params)
                        logging.warning(
                            f"FFmpeg job failed for {original_params[0]}, "
                            f"will retry with no compression. Error: {result[1]}"
                        )
                    else:
                        logging.info(f"FFmpeg job completed: {result}")

                if failed_jobs:
                    logging.info(
                        f"Retrying {len(failed_jobs)} failed jobs with no compression"
                    )
                    retry_jobs = [
                        executor.submit(
                            convert_video,
                            params[0],
                            params[1],
                            no_compression_args,
                            self.job_settings.ffmpeg_thread_cnt,
                        )
                        for params in failed_jobs
                    ]

                    for job in as_completed(retry_jobs):
                        result = job.result()
                        if isinstance(result, tuple):
                            logging.error(
                                f"No compression fallback also failed: {result[1]}"
                            )
                            raise RuntimeError(
                                "Both original compression and no compression fallback failed for job. "
                                f"Error: {result[1]}"
                            )
                        else:
                            logging.info(f"Fallback FFmpeg job completed: {result}")
        else:
            for params in convert_video_args:
                result = convert_video(
                    *params, self.job_settings.ffmpeg_thread_cnt
                )
                if isinstance(result, tuple):
                    logging.warning(
                        f"FFmpeg job failed for {params[0]}, "
                        f"retrying with no compression. Error: {result[1]}"
                    )
                    retry_result = convert_video(
                        params[0],
                        params[1],
                        no_compression_args,
                        self.job_settings.ffmpeg_thread_cnt,
                    )
                    if isinstance(retry_result, tuple):
                        logging.error(
                            f"No compression fallback also failed: {retry_result[1]}"
                        )
                        raise RuntimeError(
                            f"Both original compression and no compression fallback failed for {params[0]}. "
                            f"Error: {retry_result[1]}"
                        )
                    else:
                        logging.info(f"Fallback FFmpeg job completed: {retry_result}")
                else:
                    logging.info(f"FFmpeg job completed: {result}")

    def run_job(self) -> JobResponse:
        """
        Main public method to run the compression job.

        Returns
        -------
        JobResponse
        """
        job_start_time = time()

        video_comp_pairs = self.job_settings.video_specific_compression_requests
        job_out_dir_path = self.job_settings.output_directory.resolve()
        job_out_dir_path.mkdir(exist_ok=True)
        job_in_dir_path = self.job_settings.input_source.resolve()

        overrides = build_overrides_dict(video_comp_pairs, job_in_dir_path)
        ffmpeg_arg_set = self.job_settings.compression_requested.determine_ffmpeg_arg_set()

        convert_video_args = transform_directory(
            job_in_dir_path, job_out_dir_path, ffmpeg_arg_set, overrides
        )
        self._run_compression(convert_video_args)

        job_end_time = time()
        return JobResponse(
            status_code=200,
            message=f"Job finished in: {job_end_time - job_start_time:.2f}s",
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
        job_settings = BehaviorVideoJobSettings(
            input_source=Path("tests/test_video_in_dir"),
            output_directory=Path("tests/test_video_out_dir"),
        )

    job = BehaviorVideoJob(job_settings=job_settings)
    job_response = job.run_job()
    print(job_response.status_code)
    logging.info(job_response.model_dump_json())
