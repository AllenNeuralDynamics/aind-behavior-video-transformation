"""Module to handle transforming behavior videos"""

import logging
import shlex
import subprocess
import sys
from enum import Enum
from os import symlink, walk
from os.path import relpath
from pathlib import Path
from time import time
from typing import List, Optional, Tuple, Union

from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)
from pydantic import BaseModel, Field


class CompressionEnum(Enum):
    """
    Enum class to define different types of compression requests.
    Details of requests found in FfmpegArgSet.
    """

    DEFAULT = "default"
    GAMMA_ENCODING = "gamma"
    GAMMA_ENCODING_FIX_COLORSPACE = (
        "gamma after repairing missing input colorspace"
    )
    NO_GAMMA_ENCODING = "no gamma"
    USER_DEFINED = "user defined"
    NO_COMPRESSION = "no compression"


class CompressionRequest(BaseModel):
    """
    A model representing a request for video compression settings.

    Attributes
    ----------
    compression_enum : CompressionEnum
        Enum specifying the compression type.
    user_ffmpeg_input_options : Optional[str]
        User-defined ffmpeg input options.
    user_ffmpeg_output_options : Optional[str]
        User-defined ffmpeg output options.

    Methods
    -------
    determine_ffmpeg_arg_set() -> Optional[Tuple[str, str]]
    """

    compression_enum: CompressionEnum = Field(
        default=CompressionEnum.DEFAULT,
        description="Params to pass to ffmpeg command",
    )  # Choose among FfmegParams Enum or provide your own string.
    user_ffmpeg_input_options: Optional[str] = Field(
        default=None, description="User defined ffmpeg input options"
    )
    user_ffmpeg_output_options: Optional[str] = Field(
        default=None, description="User defined ffmpeg output options"
    )

    def determine_ffmpeg_arg_set(self) -> Optional[Tuple[str, str]]:
        """
        Determines the appropriate set of FFmpeg arguments based on the
        compression requirements.

        Returns
        -------
        Optional[Tuple[str, str]]
            A tuple containing the FFmpeg input and output options if
            compression is required, or None if no compression is needed.

        Notes
        -----
        - If `compression_enum` is `NO_COMPRESSION`, the method returns None.
        - If `compression_enum` is `USER_DEFINED`, the method returns
            user-defined FFmpeg options.
        - For other compression types, the method uses predefined
            FFmpeg argument sets.
        - If `compression_enum` is `DEFAULT`, it defaults to
            `GAMMA_ENCODING`.
        """
        comp_req = self.compression_enum
        # Handle two special cases
        if comp_req == CompressionEnum.NO_COMPRESSION:
            arg_set = None
        elif comp_req == CompressionEnum.USER_DEFINED:
            arg_set = (
                self.user_ffmpeg_input_options,
                self.user_ffmpeg_output_options,
            )
        # If not one of the two special cases, use the enum values
        else:
            # If default, set compression to gamma
            if comp_req == CompressionEnum.DEFAULT:
                compression_preset = CompressionEnum.GAMMA_ENCODING
            else:
                compression_preset = self.compression_enum
            arg_set_enum = FfmpegArgSet[compression_preset.name].value
            arg_set = (arg_set_enum[0].value, arg_set_enum[1].value)
        return arg_set


class FfmpegInputArgs(Enum):
    """
    Input arguments referenced inside FfmpegArgSet
    """

    NONE = ""


class FfmpegOutputArgs(Enum):
    """
    Output arguments referenced inside FfmpegArgSet
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

    # Many video files are missing colorspace metadata, which is necessary for
    # correct gamma conversion. This option applies commonly used values at
    # AIND, which are assumed to be correct for most videos.
    GAMMA_ENCODING_FIX_INPUT_COLOR = (
        "-vf "
        '"setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709,'
        "scale=out_color_matrix=bt709:out_range=full:sws_dither=none,"
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


class FfmpegArgSet(Enum):
    """
    Define different ffmpeg params to be used for video compression.
    Two-tuple with first element as input params and second element as output
    params.
    """

    GAMMA_ENCODING = (
        FfmpegInputArgs.NONE,
        FfmpegOutputArgs.GAMMA_ENCODING,
    )
    GAMMA_ENCODING_FIX_COLORSPACE = (
        FfmpegInputArgs.NONE,
        FfmpegOutputArgs.GAMMA_ENCODING_FIX_INPUT_COLOR,
    )
    NO_GAMMA_ENCODING = (
        FfmpegInputArgs.NONE,
        FfmpegOutputArgs.NO_GAMMA_ENCODING,
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


def likely_video_file(file: Path) -> bool:
    """
    Check if a file is likely a video file based on its suffix.

    Parameters
    ----------
    file : Path
        The file path to check.

    Returns
    -------
    bool
        True if the file suffix indicates it is a video file, False otherwise.
    """
    return file.suffix in set(
        [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".flv",
            ".wmv",
            ".webm",
        ]
    )


def convert_video(video_path: Path, dst: Path, arg_set) -> Path:
    """
    Converts a video to a specified format using ffmpeg.

    Parameters
    ----------
    video_path : Path
        The path to the input video file.
    dst : Path
        The destination directory where the converted video will be saved.
    arg_set : tuple or None
        A tuple containing input and output arguments for ffmpeg. If None, a
        symlink to the original video is created.

    Returns
    -------
    Path
        The path to the converted video file.

    Notes
    -----
    - The function uses ffmpeg for video conversion.
    - If `arg_set` is None, the function creates a symlink to the original
        video file.
    - The function logs the ffmpeg command used for conversion.
    """

    out_path = dst / f"{video_path.stem}.mp4"  # noqa: E501
    # Pydantic validation ensures this is a 'CompressionRequest' value.

    # Trivial Case, do nothing
    if arg_set is None:
        symlink(video_path, out_path)
        return out_path

    input_args = arg_set[0]
    output_args = arg_set[1]

    ffmpeg_command = ["ffmpeg", "-y", "-v", "warning", "-hide_banner"]
    if input_args:
        ffmpeg_command.extend(shlex.split(input_args))
    ffmpeg_command.extend(["-i", str(video_path)])
    if output_args:
        ffmpeg_command.extend(shlex.split(output_args))
    ffmpeg_command.append(str(out_path))

    # For logging I guess
    ffmpeg_str = " ".join(ffmpeg_command)
    logging.info(f"{ffmpeg_str=}")

    subprocess.run(ffmpeg_command, check=True)

    return out_path


def transform_directory(
    input_dir: Path, output_dir: Path, arg_set, overrides=dict()
) -> None:
    """
    Transforms all video files in a directory and its subdirectories,
    and creates symbolic links for non-video files. Subdirectories are
    created as needed.

    Parameters
    ----------
    input_dir : Path
        The directory containing the input files.
    output_dir : Path
        The directory where the transformed files and symbolic links will be
        saved.
    arg_set : Any
        The set of arguments to be used for video transformation.
    overrides : dict, optional
        A dictionary containing overrides for specific directories or files.
        Keys are Paths and values are argument sets. Default is an empty
        dictionary.

    Returns
    -------
    None
    """
    for root, dirs, files in walk(input_dir, followlinks=True):
        root_path = Path(root)
        in_relpath = relpath(root, input_dir)
        dst_dir = output_dir / in_relpath
        for dir_name in dirs:
            out_path = dst_dir / dir_name
            out_path.mkdir(parents=True, exist_ok=True)

        for file_name in files:
            file_path = Path(root) / file_name
            if likely_video_file(file_path):
                # If the parent directory has an override, use that
                this_arg_set = overrides.get(root_path, arg_set)
                # File-level overrides take precedence
                this_arg_set = overrides.get(file_path, this_arg_set)
                convert_video(file_path, dst_dir, this_arg_set)
            else:
                out_path = dst_dir / file_name
                symlink(file_path, out_path)


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
        job_in_dir_path = self.job_settings.input_source.resolve()
        overrides = dict()
        if video_comp_pairs:
            for video_name, comp_req in video_comp_pairs:
                video_path = Path(video_name)
                # Figure out how video path was passed, convert to absolute
                if video_path.is_absolute():
                    in_path = video_path
                elif video_path.exists():
                    in_path = video_path.resolve()
                else:
                    in_path = (job_in_dir_path / video_path).resolve()
                # Set overrides for the video path
                override_arg_set = comp_req.determine_ffmpeg_arg_set()
                # If it is a directory, set overrides for all subdirectories
                if in_path.is_dir():
                    overrides[in_path] = override_arg_set
                    for root, dirs, _ in walk(in_path, followlinks=True):
                        root_path = Path(root)
                        for dir_name in dirs:
                            subdir = root_path / dir_name
                            overrides[subdir] = override_arg_set
                # If it is a file, set override for the file
                else:
                    overrides[in_path] = override_arg_set

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
