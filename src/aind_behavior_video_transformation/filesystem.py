"""Module for handling file discovery to transform videos."""

import logging
import re
from os import symlink, walk
from os.path import relpath
from pathlib import Path

from aind_video_utils import VIDEO_EXTENSIONS


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
    return file.suffix.lower() in VIDEO_EXTENSIONS


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
            if video_path.is_absolute() or video_path.exists():
                candidate = video_path
            else:
                candidate = job_in_dir_path / video_path
            in_path = candidate.parent.resolve() / candidate.name
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


def create_symlinks(symlink_args: list[tuple[Path, Path]]) -> None:
    """
    Creates symbolic links for the given source and destination pairs.

    Kept separate from discovery so a partitioned job can have each node
    create only its own share. Every node must run discovery to learn what
    its partition holds, so creating links there means every node races to
    create every link, and the loser raises FileExistsError.

    Parameters
    ----------
    symlink_args : list of tuple
        Pairs of (source path, destination path).

    Returns
    -------
    None
    """
    for src_path, out_path in symlink_args:
        # is_symlink() first: exists() follows the link, so a dangling
        # symlink from an earlier run looks absent and then fails to create.
        if out_path.is_symlink() or out_path.exists():
            logging.warning(f"Output path {out_path} already exists!")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        symlink(src_path, out_path)


def transform_directory(
    input_dir: Path,
    output_dir: Path,
    arg_set,
    overrides=dict(),
    file_filter_pattern: str | None = None,
) -> tuple[
    list[tuple[Path, Path, tuple[str, str] | None]],
    list[tuple[Path, Path]],
]:
    """
    Discovers the work to be done under a directory and its subdirectories.
    Output subdirectories are created as needed.

    This only discovers. No videos are transformed and no symbolic links
    are created, so that every node of a partitioned job can call it and
    then act on its own partition alone. Pass the second return value to
    create_symlinks to make the links.

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
    A tuple of two lists. The first holds convert_video arguments, one per
    video file. The second holds (source, destination) pairs for the
    non-video files, which are to be symlinked rather than transformed.
    """

    convert_video_args = []
    symlink_args = []
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
                symlink_args.append((file_path, dst_dir / file_name))

    return convert_video_args, symlink_args
