"""Module that defines the ETL class for behavior video transformations."""

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from aind_behavior_video_transformation.transform_videos import (
    CompressionRequest,
    convert_video,
)

PathLike = Union[Path, str]


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
        List[Tuple[PathLike, CompressionRequest]]
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
    video_extensions: List[str] = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".flv",
        ".wmv",
        ".webm",
    ]


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

    def _format_output_directory(self) -> None:
        """
        Recurisively copies (symlink) non-video files
        from input directory to output directory
        perserving filesysem structure.
        """
        input_dir = Path(self.job_settings.input_source.resolve())
        output_dir = Path(self.job_settings.output_directory.resolve())
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in input_dir.rglob("*"):
            if file.is_file() and not (
                file.suffix.lower() in self.job_settings.video_extensions
            ):
                relative_path = file.relative_to(input_dir)
                target_link = output_dir / relative_path
                target_link.parent.mkdir(parents=True, exist_ok=True)
                target_link.symlink_to(file)

    def _resolve_compression_requests(
        self,
    ) -> List[Tuple[PathLike, CompressionRequest]]:
        """
        Recursively traverses input directory and
        resolves CompressionRequest of all videos.

        Sets 'compression_requested' as the default, unless
        overrided in 'video_specific_compression_requests'.
        """
        input_dir = Path(self.job_settings.input_source.resolve())

        # Set all videos to global compression setting
        path_req_pairs: List[Tuple[Path, CompressionRequest]] = [
            (file.resolve(), self.job_settings.compression_requested)
            for file in input_dir.rglob("*")
            if (
                file.is_file()
                and (file.suffix.lower() in self.job_settings.video_extensions)
            )
        ]

        # Apply overrides if present.
        comp_reqs = self.job_settings.video_specific_compression_requests
        if comp_reqs:
            # Define map: override abs_vid_path -> override CompressionRequest
            overrides: dict[Path, CompressionRequest] = {}
            for override_path, override_req in comp_reqs:
                override_path = Path(override_path)

                # Case 1: Override Path is a file.
                if override_path.is_file():
                    override_path = override_path.resolve()
                    overrides[override_path] = override_req

                # Case 2: Override Path is a subdirectory (relative/absolute)
                else:
                    if not override_path.is_absolute():
                        override_path = input_dir / override_path
                    for override_file in override_path.rglob("*"):
                        if override_file.is_file() and (
                            override_file.suffix.lower()
                            in self.job_settings.video_extensions
                        ):
                            override_file = override_file.resolve()
                            overrides[override_file] = override_req

            # Filter path_req_pairs by overrides.
            output_pairs: List[Tuple[Path, CompressionRequest]] = []
            for vid_path, default_req in path_req_pairs:
                for override_path, override_req in overrides.items():
                    if vid_path.samefile(override_path):
                        output_pairs.append((override_path, override_req))
                    else:
                        output_pairs.append((vid_path, default_req))
            path_req_pairs = output_pairs

        return path_req_pairs

    def _run_compression(
        self,
        path_comp_req_pairs: List[Tuple[PathLike, CompressionRequest]],
    ) -> None:
        """
        Runs CompressionRequests at the specified paths.
        """
        input_dir = Path(self.job_settings.input_source.resolve())
        output_dir = Path(self.job_settings.output_directory.resolve())

        convert_video_params = []
        for vid_path, comp_req in path_comp_req_pairs:
            # Resolve destination
            relative_path = vid_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_vid_dir = output_path.parent

            # Resolve compression params
            arg_set = comp_req.determine_ffmpeg_arg_set()

            # Add to job buffer
            convert_video_params.append((vid_path, output_vid_dir, arg_set))
            logging.info(
                f"Compressing {str(vid_path)} \
                         w/ {comp_req.compression_enum}"
            )

        if self.job_settings.parallel_compression:
            # ProcessPool implementation
            num_jobs = len(convert_video_params)
            with ProcessPoolExecutor(max_workers=num_jobs) as executor:
                jobs = [
                    executor.submit(convert_video, *params)
                    for params in convert_video_params
                ]
                for job in as_completed(jobs):
                    try:
                        result = job.result()
                        logging.info("FFmpeg job completed:", result)
                    except Exception as e:
                        logging.info("Error:", e)

        else:
            # Execute Serially
            for params in convert_video_params:
                convert_video(params)

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

        self._format_output_directory()
        path_comp_req_pairs = self._resolve_compression_requests()
        self._run_compression(path_comp_req_pairs)

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
