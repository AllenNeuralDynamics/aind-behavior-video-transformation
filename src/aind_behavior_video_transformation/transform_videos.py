"""Module to handle transforming behavior videos.

Encoding profiles are provided by ``aind-video-utils``.  This module adds
the ETL-layer concepts that don't map to encoding profiles:
``NO_COMPRESSION`` (symlink), ``USER_DEFINED`` (arbitrary ffmpeg strings),
and the ``CompressionRequest`` Pydantic model used by job settings.
"""

import logging
import shlex
import subprocess
from enum import Enum
from os import symlink
from pathlib import Path
from typing import Optional, Tuple

from aind_video_utils.encoding import (
    OFFLINE_8BIT,
    EncodingProfile,
    with_setparams,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CompressionEnum(Enum):
    """
    Enum class to define different types of compression requests.
    """

    DEFAULT = "default"
    GAMMA_ENCODING = "gamma"
    GAMMA_ENCODING_FIX_COLORSPACE = "gamma fix colorspace"
    NO_GAMMA_ENCODING = "no gamma"
    USER_DEFINED = "user defined"
    NO_COMPRESSION = "no compression"


# Map standard compression presets to aind-video-utils encoding profiles.
# ``DEFAULT`` aliases ``GAMMA_ENCODING`` so callers can omit a preset.
_COMPRESSION_PROFILES: dict[CompressionEnum, EncodingProfile] = {
    CompressionEnum.DEFAULT: OFFLINE_8BIT,
    CompressionEnum.GAMMA_ENCODING: OFFLINE_8BIT,
    CompressionEnum.GAMMA_ENCODING_FIX_COLORSPACE: with_setparams(
        OFFLINE_8BIT
    ),
    CompressionEnum.NO_GAMMA_ENCODING: OFFLINE_8BIT.replace(
        video_filters="scale=out_range=tv:sws_dither=none,format=yuv420p",
    ),
}


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
        - For other compression types, the method uses aind-video-utils
            encoding profiles.
        - If `compression_enum` is `DEFAULT`, it defaults to
            `GAMMA_ENCODING`.
        """
        comp_req = self.compression_enum
        if comp_req == CompressionEnum.NO_COMPRESSION:
            return None
        if comp_req == CompressionEnum.USER_DEFINED:
            return (
                self.user_ffmpeg_input_options or "",
                self.user_ffmpeg_output_options or "",
            )

        profile = _COMPRESSION_PROFILES[comp_req]
        return (
            shlex.join(profile.ffmpeg_input_args()),
            shlex.join(profile.ffmpeg_output_args()),
        )


def convert_video(
    video_path: Path,
    output_dir: Path,
    arg_set: Optional[Tuple[str, str]],
    ffmpeg_thread_cnt: int = 0,
) -> str:
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
    str
        The path to the converted video file.

    Raises
    ------
    subprocess.CalledProcessError
        If ffmpeg exits with a non-zero return code.  The exception's
        ``cmd``, ``returncode``, and ``stderr`` attributes carry the
        full failure context for the caller to log.

    Notes
    -----
    - If `arg_set` is None, the function creates a symlink to the original
        video file.
    """

    out_path = output_dir / f"{video_path.stem}.mp4"

    if arg_set is None:
        symlink(video_path, out_path)
        return str(out_path)

    input_args, output_args = arg_set
    ffmpeg_command = ["ffmpeg", "-y", "-v", "warning", "-hide_banner"]
    if ffmpeg_thread_cnt > 0:
        ffmpeg_command.extend(["-threads", str(ffmpeg_thread_cnt)])
    if input_args:
        ffmpeg_command.extend(shlex.split(input_args))
    ffmpeg_command.extend(["-i", str(video_path)])
    if output_args:
        ffmpeg_command.extend(shlex.split(output_args))
    ffmpeg_command.append(str(out_path))

    subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
    return str(out_path)
