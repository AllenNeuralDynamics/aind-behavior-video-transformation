"""Tests partition assignment in the etl module."""

import tempfile
import unittest
from functools import partial
from pathlib import Path

from aind_behavior_video_transformation.etl import (
    BehaviorVideoJob,
    encode_cost,
)

VIDEO_EXTENSIONS = {".avi", ".mp4"}


def _size_cost(path: Path) -> int:
    """Deterministic stand-in for encode_cost used by balancing tests."""
    return path.stat().st_size


_partition = partial(BehaviorVideoJob._partition_args, cost_fn=_size_cost)


class TestPartitionArgs(unittest.TestCase):
    """Tests BehaviorVideoJob._partition_args"""

    def setUp(self):
        """Create a scratch directory for each test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        """Remove the scratch directory."""
        self._tmp.cleanup()

    def _entry(self, name: str, size_kb: int = 1) -> tuple:
        """Create a file on disk, return a convert_video_args-shaped tuple."""
        path = self.tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * (size_kb * 1024))
        return (path, self.tmp_path / "out" / name, None)

    def _camera_layout(self, num_cameras: int = 4) -> list:
        """One video plus three sidecar files per camera."""
        args = []
        for cam in range(1, num_cameras + 1):
            args.append(self._entry(f"camera{cam}/video.avi", 1024))
            args.append(self._entry(f"camera{cam}/metadata.json"))
            args.append(self._entry(f"camera{cam}/timestamps.csv"))
            args.append(self._entry(f"camera{cam}/notes.txt"))
        args.sort(key=lambda params: str(params[0]))
        return args

    @staticmethod
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

    def test_every_entry_assigned_exactly_once(self):
        """Nothing is dropped and nothing is converted twice."""
        for num_cameras in (1, 2, 5):
            args = self._camera_layout(num_cameras=num_cameras)
            for num_partitions in (1, 2, 3, 4, 8):
                with self.subTest(
                    num_cameras=num_cameras, num_partitions=num_partitions
                ):
                    partitions = _partition(
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
        args = self._camera_layout(num_cameras=2)
        for num_partitions in (1, 2, 3, 4, 8):
            with self.subTest(num_partitions=num_partitions):
                partitions = _partition(
                    args, num_partitions, VIDEO_EXTENSIONS
                )
                self.assertEqual(len(partitions), num_partitions)

    def test_assignment_is_deterministic(self):
        """Nodes derive the same split without coordinating."""
        args = self._camera_layout(num_cameras=4)
        first = _partition(args, 3, VIDEO_EXTENSIONS)
        second = _partition(list(reversed(args)), 3, VIDEO_EXTENSIONS)
        self.assertEqual(first, second)

    def test_videos_spread_when_stride_would_clump(self):
        """The layout that defeats a plain stride must not defeat this."""
        args = self._camera_layout(num_cameras=4)
        naive = [args[i::4] for i in range(4)]
        self.assertEqual(max(self._video_counts(naive)), 4)
        partitions = _partition(args, 4, VIDEO_EXTENSIONS)
        self.assertEqual(self._video_counts(partitions), [1, 1, 1, 1])

    def test_unequal_video_sizes_are_balanced(self):
        """Differently sized videos even out by total cost."""
        sizes = [1000, 900, 800, 700, 300, 200, 200, 100]
        args = [
            self._entry(f"video_{i}.avi", size_kb=size)
            for i, size in enumerate(sizes)
        ]
        partitions = _partition(args, 4, VIDEO_EXTENSIONS)
        loads = [
            sum(params[0].stat().st_size for params in part)
            for part in partitions
        ]
        self.assertLess(max(loads) - min(loads), 0.1 * max(loads))

    def test_empty_input(self):
        """An empty folder yields empty partitions rather than raising."""
        partitions = _partition([], 4, VIDEO_EXTENSIONS)
        self.assertEqual(partitions, [[], [], [], []])

    def test_single_entry(self):
        """One video, several partitions: the surplus come back empty."""
        args = [self._entry("only.mp4", 100)]
        partitions = _partition(args, 4, VIDEO_EXTENSIONS)
        self.assertEqual([len(part) for part in partitions], [1, 0, 0, 0])

    def test_single_partition_takes_everything(self):
        """num_partitions=1 is the unpartitioned default path."""
        args = self._camera_layout(num_cameras=3)
        partitions = _partition(args, 1, VIDEO_EXTENSIONS)
        self.assertEqual(len(partitions), 1)
        self.assertEqual(len(partitions[0]), len(args))

    def test_more_partitions_than_videos(self):
        """Surplus array tasks get no video rather than doubling up."""
        args = self._camera_layout(num_cameras=2)
        partitions = _partition(args, 6, VIDEO_EXTENSIONS)
        self.assertEqual(max(self._video_counts(partitions)), 1)

    def test_no_videos_at_all(self):
        """A symlink-only folder still spreads and does not raise."""
        args = [self._entry(f"meta_{i}.json") for i in range(6)]
        partitions = _partition(args, 4, VIDEO_EXTENSIONS)
        self.assertEqual(sum(len(part) for part in partitions), 6)
        self.assertEqual(self._video_counts(partitions), [0, 0, 0, 0])

    def test_uppercase_extensions_are_recognised(self):
        """A camera writing VIDEO.MP4 must not be taken for a sidecar."""
        args = [
            self._entry("cam1/CLIP.MP4", 500),
            self._entry("cam2/CLIP.MP4", 500),
            self._entry("cam1/notes.CSV"),
        ]
        partitions = _partition(args, 2, VIDEO_EXTENSIONS)
        self.assertEqual(self._video_counts(partitions), [1, 1])

    def test_identically_sized_videos_still_spread(self):
        """Equal sizes leave every load tied; they must not all clump."""
        args = [self._entry(f"v{i}.mp4", 50) for i in range(8)]
        partitions = _partition(args, 4, VIDEO_EXTENSIONS)
        self.assertEqual(self._video_counts(partitions), [2, 2, 2, 2])

    def test_zero_byte_videos_still_spread(self):
        """Truncated or stub recordings weigh nothing but cost time."""
        args = [self._entry(f"z{i}.mp4", 0) for i in range(6)]
        partitions = _partition(args, 3, VIDEO_EXTENSIONS)
        self.assertEqual(self._video_counts(partitions), [2, 2, 2])

    def test_one_huge_video_cannot_be_split(self):
        """A single dominant video bounds wall time; document that."""
        args = [self._entry("huge.mp4", 10000)]
        args += [self._entry(f"tiny_{i}.mp4", 1) for i in range(6)]
        partitions = _partition(args, 3, VIDEO_EXTENSIONS)
        self.assertTrue(
            any(
                any(params[0].name == "huge.mp4" for params in part)
                for part in partitions
            )
        )
        self.assertEqual(sum(len(part) for part in partitions), len(args))

    def test_missing_file_raises(self):
        """Sizing a vanished file fails loudly rather than skewing."""
        args = [self._entry("gone.mp4", 10)]
        args[0][0].unlink()
        with self.assertRaises(FileNotFoundError):
            _partition(args, 2, VIDEO_EXTENSIONS)


class TestEncodeCost(unittest.TestCase):
    """Tests encode_cost against the real sample video."""

    test_vid_path = Path("tests/test_video_in_dir/clip.mp4").resolve()

    def setUp(self):
        """Skip if the sample clip is unavailable."""
        if not self.test_vid_path.is_file():
            self.skipTest(f"sample video not found: {self.test_vid_path}")

    def test_returns_pixel_count(self):
        """Cost is frames times resolution for a real, readable video."""
        cost = encode_cost(self.test_vid_path)
        self.assertGreater(cost, 0)
        self.assertEqual(cost, 501 * 720 * 540)

    def test_corrupt_video_raises(self):
        """A file that is not a decodable video fails loudly."""
        with tempfile.TemporaryDirectory() as tmp:
            bogus = Path(tmp) / "corrupt.mp4"
            bogus.write_bytes(b"not a real video")
            with self.assertRaises(Exception):
                encode_cost(bogus)


if __name__ == "__main__":
    unittest.main()
