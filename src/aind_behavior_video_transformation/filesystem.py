"""Module for handling file discovery to transform videos."""

import logging
import re
from os import symlink, walk
from os.path import relpath
from pathlib import Path


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


def build_overrides_dict(video_comp_pairs, job_in_dir_path):
    """
    Builds a dictionary of override arguments for video paths.

    Parameters
    ----------
    video_comp_pairs : list of tuple
        A list of tuples where each tuple contains a video file name and a
        corresponding CompressionRequest object.
    job_in_dir_path : Path
        The base directory path where the job input files are located.

    Returns
    -------
    dict
        A dictionary where keys are video paths (either files or directories)
        and values are the override argument sets determined by the
        CompressionRequest objects.
    """
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

    return overrides


def transform_directory(
    input_dir: Path,
    output_dir: Path,
    arg_set,
    overrides=dict(),
    file_filter_pattern: str | None = None,
) -> list[tuple[Path, Path, tuple[str, str] | None]]:
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
    file_filter_pattern : str | None
        If set, will filter file names based on this regex pattern.
        Default is None.

    Returns
    -------
    List of tuples containing convert_video arguments.
    """

    convert_video_args = []
    for root, dirs, files in walk(input_dir, followlinks=True):
        root_path = Path(root)
        in_relpath = relpath(root, input_dir)
        dst_dir = output_dir / in_relpath
        for dir_name in dirs:
            out_path = dst_dir / dir_name
            out_path.mkdir(parents=True, exist_ok=True)

        for file_name in files:
            file_path = Path(root) / file_name
            if file_filter_pattern and not re.search(
                file_filter_pattern, file_name
            ):
                continue
            if likely_video_file(file_path):
                # If the parent directory has an override, use that
                this_arg_set = overrides.get(root_path, arg_set)
                # File-level overrides take precedence
                this_arg_set = overrides.get(file_path, this_arg_set)
                convert_video_args.append((file_path, dst_dir, this_arg_set))

            else:
                out_path = dst_dir / file_name
                if out_path.exists():
                    logging.warning(f"Output path {out_path} already exists!")
                    continue
                symlink(file_path, out_path)

    return convert_video_args
