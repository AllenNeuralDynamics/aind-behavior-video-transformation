"""
Module to handle transforming behavior videos

To add a new compression preset:
1) Define FfmpegInputArgs/FfmpegOutputArgs.
2) Define a CompressionEnum: 'NEW_PRESET = 'new_preset'
3) Add the CompressionEnum to FfmpegArgSet, and build
   (FfmpegInputArgs, FfmpegOutputArgs) tuple:
   'NEW_PRESET' = (
        FfmpegInputArgs.CUSTOM_INPUT_ARGS,
        FfmpegOutputArgs.CUSTOM_OUTPUT_ARGS,
    )
FfmpegInputArgs / FfmpegOutputArgs can be prexisitng or newly-defined in (1)
"""

import shlex
import subprocess
from enum import Enum
from os import symlink
from pathlib import Path
from subprocess import CalledProcessError
from typing import Optional, Tuple, Union

from pydantic import BaseModel, Field


class CompressionEnum(Enum):
    """
    Enum class to define different types of compression requests.
    Details of requests found in FfmpegArgSet.
    """

    DEFAULT = "default"
    GAMMA_ENCODING = "gamma"
    GAMMA_ENCODING_FIX_COLORSPACE = "gamma fix colorspace"
    NO_GAMMA_ENCODING = "no gamma"
    USER_DEFINED = "user defined"
    NO_COMPRESSION = "no compression"


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
        "format=yuv420p10le,colorspace=ispace=bt709:all=bt709:dither=none,"
        'scale=out_range=tv:sws_dither=none,format=yuv420p" -c:v libx264 '
        "-preset veryslow -crf 18 -pix_fmt yuv420p "
        '-metadata author="Allen Institute for Neural Dynamics" '
        "-movflags +faststart+write_colr"
    )

    # Many video files are missing colorspace metadata, which is necessary for
    # correct gamma conversion. This option applies commonly used values at
    # AIND, which are assumed to be correct for most videos.
    GAMMA_ENCODING_FIX_INPUT_COLOR = (
        "-vf "
        '"setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709,'
        "scale=out_color_matrix=bt709:out_range=full:sws_dither=none,"
        "format=yuv420p10le,colorspace=ispace=bt709:all=bt709:dither=none,"
        'scale=out_range=tv:sws_dither=none,format=yuv420p" -c:v libx264 '
        "-preset veryslow -crf 18 -pix_fmt yuv420p "
        '-metadata author="Allen Institute for Neural Dynamics" '
        "-movflags +faststart+write_colr"
    )
    NO_GAMMA_ENCODING = (
        "-vf "
        '"scale=out_range=tv:sws_dither=none,format=yuv420p" -c:v libx264 '
        "-preset veryslow -crf 18 -pix_fmt yuv420p "
        '-metadata author="Allen Institute for Neural Dynamics" '
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

    def determine_ffmpeg_arg_set(
        self,
    ) -> Optional[Tuple[str, str]]:
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

            # Resolve two levels of indirection here
            # FfmpegArgSet -> (FfmpegInputArgs, FfmpegOutputArgs)
            # (FfmpegInputArgs, FfmpegOutputArgs)
            #      -> (in_args str, out_args str)
            arg_set_enum = FfmpegArgSet[compression_preset.name].value
            arg_set = (arg_set_enum[0].value, arg_set_enum[1].value)

        return arg_set


def convert_video(
    video_path: Path,
    output_dir: Path,
    arg_set: Optional[Tuple[str, str]],
    ffmpeg_thread_cnt: int = 0,
) -> Union[str, Tuple[str, str]]:
    """
    Converts a video to a specified format using ffmpeg.

    Parameters
    ----------
    video_path : Path
        The path to the input video file.
    output_dir : Path
        The destination directory where the converted video will be saved.
    arg_set : tuple or None
        A tuple containing input and output arguments for ffmpeg. If None, a
        symlink to the original video is created.
    ffmpeg_thread_cnt : set number of ffmpeg threads

    Returns
    -------
    Path
        The path to the converted video file.

    Notes
    -----
    - The function uses ffmpeg for video conversion.
    - If `arg_set` is None, the function creates a symlink to the original
        video file.
    """

    out_path = output_dir / f"{video_path.stem}.mp4"  # noqa: E501

    # Trivial Case, do nothing
    if arg_set is None:
        symlink(video_path, out_path)
        return out_path

    input_args = arg_set[0]
    output_args = arg_set[1]

    ffmpeg_command = ["ffmpeg", "-y", "-v", "warning", "-hide_banner"]

    # Set thread count
    if ffmpeg_thread_cnt > 0:
        ffmpeg_command.extend(["-threads", str(ffmpeg_thread_cnt)])

    if input_args:
        ffmpeg_command.extend(shlex.split(input_args))
    ffmpeg_command.extend(["-i", str(video_path)])
    if output_args:
        ffmpeg_command.extend(shlex.split(output_args))
    ffmpeg_command.append(str(out_path))

    # Capture and return error message if it exists
    try:
        subprocess.run(
            ffmpeg_command, check=True, capture_output=True, text=True
        )
        return str(out_path)

    except CalledProcessError as e:
        error_msg = (
            f"FFmpeg conversion failed for {video_path}\n"
            f"Command: {' '.join(ffmpeg_command)}\n"
            f"Return code: {e.returncode}\n"
            f"Error output:\n{e.stderr}\n"
        )
        return (str(out_path), error_msg)
