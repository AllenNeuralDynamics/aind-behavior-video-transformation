"""Tests partition assignment in the etl module."""

import unittest
from pathlib import PurePosixPath

from aind_behavior_video_transformation.etl import BehaviorVideoJob

VIDEO_EXTENSIONS = {".avi", ".mp4"}


def _entry(name: str) -> tuple:
    """Return a convert_video_args-shaped tuple for a virtual path.

    The simplified partitioner only inspects the path's suffix and string
    form, never the filesystem, so these paths need not exist on disk.
    PurePosixPath keeps string comparisons identical across platforms.
    """
    path = PurePosixPath(name)
    return (path, PurePosixPath("out") / name, None)


def _camera_layout(num_cameras: int = 4) -> list:
    """One video plus three sidecar files per camera."""
    args = []
    for cam in range(1, num_cameras + 1):
        args.append(_entry(f"camera{cam}/video.avi"))
        args.append(_entry(f"camera{cam}/metadata.json"))
        args.append(_entry(f"camera{cam}/timestamps.csv"))
        args.append(_entry(f"camera{cam}/notes.txt"))
    args.sort(key=lambda params: str(params[0]))
    return args


def _video_counts(partitions: list) -> list:
    """Number of video entries in each partition."""
    return [
        sum(
            1
            for params in part
            if params[0].suffix.lower() in VIDEO_EXTENSIONS
        )
        for part in partitions
    ]


class TestPartitionArgs(unittest.TestCase):
    """Tests BehaviorVideoJob._partition_args"""

    def test_every_entry_assigned_exactly_once(self):
        """Nothing is dropped and nothing is converted twice."""
        for num_cameras in (1, 2, 5):
            args = _camera_layout(num_cameras=num_cameras)
            for num_partitions in (1, 2, 3, 4, 8):
                with self.subTest(
                    num_cameras=num_cameras, num_partitions=num_partitions
                ):
                    partitions = BehaviorVideoJob._partition_args(
                        args, num_partitions, VIDEO_EXTENSIONS
                    )
                    assigned = [
                        params[0] for part in partitions for params in part
                    ]
                    self.assertEqual(
                        sorted(assigned),
                        sorted(params[0] for params in args),
                    )
                    self.assertEqual(len(assigned), len(set(assigned)))

    def test_returns_one_list_per_partition(self):
        """Callers index by partition number, so shape must be exact."""
        args = _camera_layout(num_cameras=2)
        for num_partitions in (1, 2, 3, 4, 8):
            with self.subTest(num_partitions=num_partitions):
                partitions = BehaviorVideoJob._partition_args(
                    args, num_partitions, VIDEO_EXTENSIONS
                )
                self.assertEqual(len(partitions), num_partitions)

    def test_assignment_is_deterministic(self):
        """Nodes derive the same split without coordinating."""
        args = _camera_layout(num_cameras=4)
        first = BehaviorVideoJob._partition_args(args, 3, VIDEO_EXTENSIONS)
        second = BehaviorVideoJob._partition_args(
            list(reversed(args)), 3, VIDEO_EXTENSIONS
        )
        self.assertEqual(first, second)

    def test_videos_spread_when_stride_would_clump(self):
        """The layout that defeats a plain stride must not defeat this."""
        args = _camera_layout(num_cameras=4)
        naive = [args[i::4] for i in range(4)]
        self.assertEqual(max(_video_counts(naive)), 4)
        partitions = BehaviorVideoJob._partition_args(
            args, 4, VIDEO_EXTENSIONS
        )
        self.assertEqual(_video_counts(partitions), [1, 1, 1, 1])

    def test_empty_input(self):
        """An empty folder yields empty partitions rather than raising."""
        partitions = BehaviorVideoJob._partition_args(
            [], 4, VIDEO_EXTENSIONS
        )
        self.assertEqual(partitions, [[], [], [], []])

    def test_single_entry(self):
        """One video, several partitions: the surplus come back empty."""
        args = [_entry("only.mp4")]
        partitions = BehaviorVideoJob._partition_args(
            args, 4, VIDEO_EXTENSIONS
        )
        self.assertEqual([len(part) for part in partitions], [1, 0, 0, 0])

    def test_single_partition_takes_everything(self):
        """num_partitions=1 is the unpartitioned default path."""
        args = _camera_layout(num_cameras=3)
        partitions = BehaviorVideoJob._partition_args(
            args, 1, VIDEO_EXTENSIONS
        )
        self.assertEqual(len(partitions), 1)
        self.assertEqual(len(partitions[0]), len(args))

    def test_more_partitions_than_videos(self):
        """Surplus array tasks get no video rather than doubling up."""
        args = _camera_layout(num_cameras=2)
        partitions = BehaviorVideoJob._partition_args(
            args, 6, VIDEO_EXTENSIONS
        )
        self.assertEqual(max(_video_counts(partitions)), 1)

    def test_no_videos_at_all(self):
        """A symlink-only folder still spreads and does not raise."""
        args = [_entry(f"meta_{i}.json") for i in range(6)]
        partitions = BehaviorVideoJob._partition_args(
            args, 4, VIDEO_EXTENSIONS
        )
        self.assertEqual(sum(len(part) for part in partitions), 6)
        self.assertEqual(_video_counts(partitions), [0, 0, 0, 0])

    def test_uppercase_extensions_are_recognised(self):
        """A camera writing VIDEO.MP4 must not be taken for a sidecar."""
        args = [
            _entry("cam1/CLIP.MP4"),
            _entry("cam2/CLIP.MP4"),
            _entry("cam1/notes.CSV"),
        ]
        partitions = BehaviorVideoJob._partition_args(
            args, 2, VIDEO_EXTENSIONS
        )
        self.assertEqual(_video_counts(partitions), [1, 1])

    def test_videos_spread_evenly_by_count(self):
        """Videos are spread round-robin so counts stay balanced."""
        args = [_entry(f"v{i}.mp4") for i in range(8)]
        partitions = BehaviorVideoJob._partition_args(
            args, 4, VIDEO_EXTENSIONS
        )
        self.assertEqual(_video_counts(partitions), [2, 2, 2, 2])

    def test_all_entries_present_with_mixed_files(self):
        """Videos and sidecars are both distributed, none dropped."""
        args = _camera_layout(num_cameras=3)
        partitions = BehaviorVideoJob._partition_args(
            args, 3, VIDEO_EXTENSIONS
        )
        self.assertEqual(sum(len(part) for part in partitions), len(args))


if __name__ == "__main__":
    unittest.main()
