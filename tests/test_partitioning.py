"""Tests partition assignment in the etl module."""

import unittest
from pathlib import PurePosixPath

from aind_behavior_video_transformation.etl import BehaviorVideoJob


def _entry(name: str) -> tuple:
    """Return a convert_video_args-shaped tuple for a virtual path.

    The partitioner only inspects the path's string form, never the
    filesystem, so these paths need not exist on disk. PurePosixPath keeps
    string comparisons identical across platforms.
    """
    path = PurePosixPath(name)
    return (path, PurePosixPath("out") / name, None)


def _videos(num_cameras: int = 4) -> list:
    """The video entries transform_directory would return."""
    args = [_entry(f"camera{cam}/video.avi")
            for cam in range(1, num_cameras + 1)]
    args.sort(key=lambda params: str(params[0]))
    return args


def _symlinks(num_cameras: int = 4) -> list:
    """The symlink pairs transform_directory would return."""
    args = []
    for cam in range(1, num_cameras + 1):
        args.append(_entry(f"camera{cam}/metadata.json"))
        args.append(_entry(f"camera{cam}/timestamps.csv"))
        args.append(_entry(f"camera{cam}/notes.txt"))
    args.sort(key=lambda params: str(params[0]))
    return args


class TestPartitionArgs(unittest.TestCase):
    """Tests BehaviorVideoJob._partition_args"""

    def test_every_entry_assigned_exactly_once(self):
        """Nothing is dropped and nothing is converted twice.

        A dropped entry means missing output; a duplicated one means two
        array tasks writing the same file at the same time.
        """
        for num_cameras in (1, 2, 5):
            args = _videos(num_cameras=num_cameras)
            for num_partitions in (1, 2, 3, 4, 8):
                with self.subTest(
                    num_cameras=num_cameras, num_partitions=num_partitions
                ):
                    partitions = BehaviorVideoJob._partition_args(
                        args, num_partitions
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
        args = _videos(num_cameras=2)
        for num_partitions in (1, 2, 3, 4, 8):
            with self.subTest(num_partitions=num_partitions):
                partitions = BehaviorVideoJob._partition_args(
                    args, num_partitions
                )
                self.assertEqual(len(partitions), num_partitions)

    def test_assignment_is_deterministic(self):
        """Nodes derive the same split without coordinating.

        Each array task calls this on its own, so an assignment that
        depended on input ordering would let two nodes claim one video.
        """
        args = _videos(num_cameras=4)
        first = BehaviorVideoJob._partition_args(args, 3)
        second = BehaviorVideoJob._partition_args(list(reversed(args)), 3)
        self.assertEqual(first, second)

    def test_videos_spread_evenly_by_count(self):
        """Videos are spread round-robin so counts stay balanced."""
        args = [_entry(f"v{i}.mp4") for i in range(8)]
        partitions = BehaviorVideoJob._partition_args(args, 4)
        self.assertEqual([len(part) for part in partitions], [2, 2, 2, 2])

    def test_videos_and_symlinks_partitioned_separately(self):
        """Each partition gets a share of both kinds of work.

        run_job partitions the two lists independently. Partitioning a
        single combined list would let a run of sidecar files shift where
        the videos land, leaving one node with every video and the rest
        with only symlinks.
        """
        videos = _videos(num_cameras=4)
        symlinks = _symlinks(num_cameras=4)

        video_parts = BehaviorVideoJob._partition_args(videos, 4)
        symlink_parts = BehaviorVideoJob._partition_args(symlinks, 4)

        self.assertEqual([len(part) for part in video_parts], [1, 1, 1, 1])
        self.assertEqual([len(part) for part in symlink_parts], [3, 3, 3, 3])

        combined = BehaviorVideoJob._partition_args(videos + symlinks, 4)
        video_names = {params[0] for params in videos}
        clumped = [
            sum(1 for params in part if params[0] in video_names)
            for part in combined
        ]
        self.assertEqual(
            max(clumped),
            4,
            "expected a single combined list to clump the videos",
        )

    def test_empty_input(self):
        """An empty folder yields empty partitions rather than raising."""
        partitions = BehaviorVideoJob._partition_args([], 4)
        self.assertEqual(partitions, [[], [], [], []])

    def test_single_entry(self):
        """One video, several partitions: the surplus come back empty."""
        args = [_entry("only.mp4")]
        partitions = BehaviorVideoJob._partition_args(args, 4)
        self.assertEqual([len(part) for part in partitions], [1, 0, 0, 0])

    def test_single_partition_takes_everything(self):
        """num_partitions=1 is the unpartitioned default path."""
        args = _videos(num_cameras=3)
        partitions = BehaviorVideoJob._partition_args(args, 1)
        self.assertEqual(len(partitions), 1)
        self.assertEqual(len(partitions[0]), len(args))

    def test_more_partitions_than_entries(self):
        """Surplus array tasks get nothing rather than doubling up."""
        args = _videos(num_cameras=2)
        partitions = BehaviorVideoJob._partition_args(args, 6)
        self.assertEqual(len(partitions), 6)
        self.assertEqual(sum(len(part) for part in partitions), len(args))
        self.assertEqual(max(len(part) for part in partitions), 1)


if __name__ == "__main__":
    unittest.main()
