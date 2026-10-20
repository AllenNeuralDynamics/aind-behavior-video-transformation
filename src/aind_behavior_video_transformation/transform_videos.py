"""Module to handle transforming behavior videos.

Encoding profiles are provided by ``aind-video-utils``, which also writes
the preview, preview metadata and poster beside each video encoded with
one.  This module adds the ETL-layer concepts that don't map to encoding
profiles: ``NO_COMPRESSION`` (symlink), ``USER_DEFINED`` (arbitrary ffmpeg
strings), and the ``CompressionRequest`` Pydantic model used by job
settings.
"""

import logging
import shlex
import subprocess
from enum import Enum
from os import symlink
from pathlib import Path
from typing import Optional, Tuple, Union

from aind_video_utils.encoding import (
    OFFLINE_8BIT,
    EncodingProfile,
    with_setparams,
)
from aind_video_utils.transcode import transcode_video
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


_NO_GAMMA_PROFILE = OFFLINE_8BIT.replace(
    video_filters="scale=out_range=tv:sws_dither=none,format=yuv420p",
)

# Map standard compression presets to aind-video-utils encoding profiles.
# ``DEFAULT`` aliases ``GAMMA_ENCODING`` so callers can omit a preset.
_COMPRESSION_PROFILES: dict[CompressionEnum, EncodingProfile] = {
    CompressionEnum.DEFAULT: OFFLINE_8BIT,
    CompressionEnum.GAMMA_ENCODING: OFFLINE_8BIT,
    CompressionEnum.GAMMA_ENCODING_FIX_COLORSPACE: with_setparams(
        OFFLINE_8BIT
    ),
    CompressionEnum.NO_GAMMA_ENCODING: _NO_GAMMA_PROFILE,
}

# The poster converts the source's transfer to sRGB, so on a source whose
# transfer ffmpeg cannot resolve it fails, taking the whole encode with it.
# The gamma-encoding presets already need a known transfer; no gamma does not.
_POSTERLESS_PROFILES = (_NO_GAMMA_PROFILE,)

# The spec names a camera folder's video ``video.<ext>``, and its
# ``metadata.csv`` describes that video's frames and no other's.
_SPEC_VIDEO_STEM = "video"
_SPEC_METADATA_FILENAME = "metadata.csv"

PREVIEW_FPS = 30.0

# What to do with one video: encode it with a profile, which also writes
# its derivatives; run ffmpeg with user-defined (input, output) options; or
# symlink it unchanged.
Encoding = Union[EncodingProfile, Tuple[str, str], None]


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
    determine_encoding() -> Encoding
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

    def determine_encoding(self) -> Encoding:
        """
        Determines how to encode a video based on the compression
        requirements.

        Returns
        -------
        Encoding
            The aind-video-utils encoding profile for a preset, the FFmpeg
            input and output options for user-defined compression, or None
            if no compression is needed.

        Notes
        -----
        - If `compression_enum` is `NO_COMPRESSION`, the method returns None.
        - If `compression_enum` is `USER_DEFINED`, the method returns
            user-defined FFmpeg options.
        - For other compression types, the method returns an
            aind-video-utils encoding profile.
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

        return _COMPRESSION_PROFILES[comp_req]


def convert_video(
    video_path: Path,
    output_dir: Path,
    encoding: Encoding,
    ffmpeg_thread_cnt: int = 0,
    write_previews: bool = True,
) -> str:
    """
    Converts a video to a specified format using ffmpeg.

    Parameters
    ----------
    video_path : Path
        The path to the input video file.
    output_dir : Path
        The destination directory where the converted video will be saved.
    encoding : Encoding
        An encoding profile, a tuple of input and output arguments for
        ffmpeg, or None to create a symlink to the original video.
    ffmpeg_thread_cnt : int
        Number of ffmpeg threads; 0 lets ffmpeg choose.
    write_previews : bool
        Whether a video encoded with a profile also gets a preview, its
        preview metadata, and a poster. These take the spec's names
        beside ``video.mp4`` and the video's stem otherwise, as in
        ``BottomCamera_preview.mp4``. The preview metadata is written only
        for ``video.<ext>`` with a ``metadata.csv`` beside it.

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
    RuntimeError
        If a profile's encode loses frames, or ``metadata.csv`` has a row
        count other than the video's frame count.
    ValueError
        If ``metadata.csv`` has no ``ReferenceTime`` column.
    """

    out_path = output_dir / f"{video_path.stem}.mp4"

    if encoding is None:
        symlink(video_path, out_path)
        return str(out_path)

    if isinstance(encoding, EncodingProfile):
        metadata_csv = video_path.with_name(_SPEC_METADATA_FILENAME)
        with_metadata = (
            write_previews
            and video_path.stem == _SPEC_VIDEO_STEM
            and metadata_csv.is_file()
        )
        transcode_video(
            video_path,
            out_path,
            profile=encoding,
            # The presets carry the colour repair they intend; probing for
            # more would change their primary output.
            auto_fix_colorspace=False,
            preview_fps=PREVIEW_FPS if write_previews else None,
            metadata_csv=metadata_csv if with_metadata else None,
            poster=write_previews and encoding not in _POSTERLESS_PROFILES,
            threads=ffmpeg_thread_cnt or None,
        )
        return str(out_path)

    input_args, output_args = encoding
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
