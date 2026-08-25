"""Module that defines the ETL class for behavior video transformations."""

import logging
import shlex
import subprocess
import sys
from pathlib import Path
from subprocess import CalledProcessError
from time import time

from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)
from pydantic import Field

from aind_behavior_video_transformation.filesystem import (
    VIDEO_EXTENSIONS,
    build_overrides_dict,
    transform_directory,
)
from aind_behavior_video_transformation.transform_videos import (
    CompressionRequest,
    convert_video,
)

logger = logging.getLogger(__name__)


def _format_ffmpeg_error(video_path: Path, exc: CalledProcessError) -> str:
    """Format an ffmpeg ``CalledProcessError`` as a single log record body.

    The format keeps each piece on its own line so ffmpeg stderr is
    clearly visible to readers and to log aggregators.
    """
    stderr = (exc.stderr or "(no stderr captured)").rstrip("\n")
    return (
        f"FFmpeg conversion failed for {video_path}\n"
        f"Command: {shlex.join(exc.cmd)}\n"
        f"Return code: {exc.returncode}\n"
        f"--- ffmpeg stderr ---\n"
        f"{stderr}\n"
        f"--- end stderr ---"
    )


def encode_cost(path: Path) -> int:
    """Return total video pixels as a proxy for encode cost.

    Uses container metadata when available, falling back to scanning the
    video stream only when the container does not provide a frame count.
    """
    path = Path(path)

    out = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,width,height",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stream = json.loads(out.stdout)["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])

    frame_count = stream.get("nb_frames")
    if frame_count not in (None, "N/A"):
        return int(frame_count) * width * height

    # Some containers do not expose nb_frames, so count them explicitly.
    out = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    frame_count = out.stdout.strip()
    if not frame_count or frame_count == "N/A":
        raise ValueError(f"Could not determine frame count for {path}")

    return int(frame_count) * width * height


class BehaviorVideoJobSettings(BasicJobSettings):
    """
    BehaviorJob settings. Inherits both fields input_source and
    output_directory from BasicJobSettings.
    """

    compression_requested: CompressionRequest = Field(
        default=CompressionRequest(),
        description="Compression requested for video files",
    )
    video_specific_compression_requests: (
        list[tuple[Path | str, CompressionRequest]] | None
    ) = Field(
        default=None,
        description=(
            "Pairs of video files or directories containing videos, and "
            "compression requests that differ from the global compression "
            "request"
        ),
    )

    ffmpeg_thread_cnt: int = Field(
        default=0, description="Number of threads per ffmpeg compression job."
    )
    file_filter: str | None = Field(
        default=None,
        description="If set, filter file paths based on regex pattern.",
    )

    partition_number: int = Field(
        default=1,
        ge=1,
        description=(
            "1-based partition index for this node. Pass "
            "$SLURM_ARRAY_TASK_ID here with --array=1-N (NOT 0-based)."
        ),
    )

    num_partitions: int = Field(
        default=1,
        ge=1,
        description=(
            "Total number of partitions in the array. Must be identical on "
            "every node. Default 1 means this node processes all videos."
        ),
    )

    video_extensions: set[str] = Field(
        default_factory=lambda: set(VIDEO_EXTENSIONS),
        description=(
            "Lowercase suffixes treated as videos when balancing partitions. "
            "Entries not matching are assumed to be symlinked at negligible "
            "cost and are spread by count rather than by weight."
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

    def _run_serial(
        self,
        convert_video_args: list[tuple[Path, Path, tuple[str, str] | None]],
    ) -> list[tuple[Path, CalledProcessError]]:
        """Run conversions one at a time, collecting failures."""
        errors: list[tuple[Path, CalledProcessError]] = []
        thread_cnt = self.job_settings.ffmpeg_thread_cnt
        for params in convert_video_args:
            video_path = params[0]
            try:
                result = convert_video(*params, thread_cnt)
            except CalledProcessError as exc:
                errors.append((video_path, exc))
            else:
                logger.info("FFmpeg job completed: %s", result)
        return errors

    def _run_compression(
        self,
        convert_video_args: list[tuple[Path, Path, tuple[str, str] | None]],
    ) -> None:
        """
        Runs CompressionRequests at the specified paths, sequentially within
        this partition.
        """
        errors = self._run_serial(convert_video_args)

        if not errors:
            return

        formatted = [
            _format_ffmpeg_error(video_path, exc) for video_path, exc in errors
        ]
        for block in formatted:
            logger.error(block)
        raise RuntimeError(
            f"{len(errors)} ffmpeg job(s) failed:\n\n" + "\n\n".join(formatted)
        )

    @staticmethod
    def _partition_args(
        convert_video_args: list[tuple[Path, Path, tuple[str, str] | None]],
        num_partitions: int,
        video_extensions: set[str],
    ) -> list[list[tuple[Path, Path, tuple[str, str] | None]]]:
        """Split the work list into partitions of comparable cost.

        The input source holds a mix of video files, which are transcoded by
        ffmpeg, and non-video files, which are only symlinked. Those costs
        differ by orders of magnitude, so slicing the combined list balances
        entry counts while leaving the real work lopsided. In the worst case
        the stride aligns with the directory layout and a single partition
        receives every video.

        Videos are therefore assigned first, largest to smallest, each going
        to whichever partition is currently lightest (greedy
        longest-processing-time). File size stands in for encode cost, which
        holds when videos share a duration and recording setup, as different
        camera angles from one session do. The symlink-only entries are then
        spread round-robin, since their cost is negligible.

        Ties break on the input path so every node in the array derives the
        same assignment independently, without coordinating.

        Returns one list per partition, in partition order.
        """
        videos = []
        others = []
        for params in convert_video_args:
            if params[0].suffix.lower() in video_extensions:
                videos.append(params)
            else:
                others.append(params)

        partitions: list[list] = [[] for _ in range(num_partitions)]

        loads = [0] * num_partitions
        weighted = sorted(
            ((frame_count(params[0]), params) for params in videos),
            key=lambda pair: (pair[0], str(pair[1][0])),
            reverse=True,
        )

        for size, params in weighted:
            lightest = min(
                range(num_partitions),
                key=lambda i: (loads[i], len(partitions[i]), i),
            )
            partitions[lightest].append(params)
            loads[lightest] += size

        for index, params in enumerate(
            sorted(others, key=lambda params: str(params[0]))
        ):
            partitions[index % num_partitions].append(params)

        return partitions

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
        Path(job_out_dir_path).mkdir(parents=True, exist_ok=True)
        job_in_dir_path = self.job_settings.input_source.resolve()
        overrides = build_overrides_dict(video_comp_pairs, job_in_dir_path)

        ffmpeg_arg_set = (
            self.job_settings.compression_requested.determine_ffmpeg_arg_set()
        )
        file_filter = self.job_settings.file_filter
        convert_video_args = transform_directory(
            job_in_dir_path,
            job_out_dir_path,
            ffmpeg_arg_set,
            overrides,
            file_filter,
        )

        convert_video_args.sort(key=lambda x: str(x[0]))

        total_entries = len(convert_video_args)
        partition_number = self.job_settings.partition_number
        num_partitions = self.job_settings.num_partitions
        video_extensions = self.job_settings.video_extensions

        if partition_number > num_partitions:
            raise ValueError(
                f"partition_number ({partition_number}) exceeds "
                f"num_partitions ({num_partitions})"
            )

        # This script runs once per SLURM array task; every task executes
        # this same code independently with a different partition_number
        # (from $SLURM_ARRAY_TASK_ID). _partition_args deterministically
        # computes the full split of all jobs into num_partitions groups,
        # and each task selects only the group it is responsible for. The
        # split is derived identically on every node, so no coordination
        # between tasks is needed.
        all_partitions = self._partition_args(
            convert_video_args,
            num_partitions,
            video_extensions,
        )
        this_partition = all_partitions[partition_number - 1]

        num_videos = sum(
            1
            for params in this_partition
            if params[0].suffix.lower() in video_extensions
        )
        logger.info(
            "Partition %d/%d: processing %d of %d entries (%d videos)",
            partition_number,
            num_partitions,
            len(this_partition),
            total_entries,
            num_videos,
        )

        self._run_compression(this_partition)

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
        job_settings = BehaviorVideoJobSettings(
            input_source=Path("tests/test_video_in_dir"),
            output_directory=Path("tests/test_video_out_dir"),
        )

    job = BehaviorVideoJob(job_settings=job_settings)

    job_response = job.run_job()
    print(job_response.status_code)

    logger.info(job_response.model_dump_json())
