"""Host-only partition and coverage tests for the StreamK scheduler."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from tensile.scheduler import compute_schedule


class TestComputeSchedule:
    """Verify compute_schedule produces correct partition math."""

    def test_basic_partition(self):
        # M=N=256, K=512, bm=bn=128, bk=64, grid=4
        sched = compute_schedule(M=256, N=256, K=512, bm=128, bn=128, bk=64, grid=4)
        assert sched["totalTiles"] == 4  # (256//128) * (256//128)
        assert sched["itersPerTile"] == 8  # 512 // 64
        assert sched["totalIters"] == 32  # 4 * 8
        assert sched["itersPerWg"] == 8  # 32 // 4
        assert sched["extraIters"] == 0  # 32 % 4

    def test_coverage_no_remainder(self):
        """Per-WG ranges must cover [0, totalIters) exactly."""
        sched = compute_schedule(M=256, N=256, K=512, bm=128, bn=128, bk=64, grid=4)
        total = sched["totalIters"]
        ipw = sched["itersPerWg"]
        extra = sched["extraIters"]
        covered = set()
        for wg in range(4):
            start = wg * ipw + min(wg, extra)
            end = start + (ipw + 1 if wg < extra else ipw)
            assert start >= 0 and end <= total
            for i in range(start, end):
                assert i not in covered, f"iteration {i} covered twice"
                covered.add(i)
        assert covered == set(range(total))

    def test_coverage_with_remainder(self):
        """Remainder iters distributed correctly across WGs."""
        # totalIters=33, grid=4 → itersPerWg=8, extraIters=1
        # (M=N=128 for 1 tile, K=512 for 8 iters, but we force via parameters)
        sched = compute_schedule(M=128, N=128, K=512, bm=128, bn=128, bk=64, grid=4)
        # 1 tile * 8 iters/tile = 8 totalIters, itersPerWg=2, extra=0
        # That's not interesting; try a case with extra:
        sched = compute_schedule(M=256, N=256, K=512, bm=128, bn=128, bk=64, grid=5)
        # totalTiles=4, itersPerTile=8, totalIters=32, grid=5
        # itersPerWg = 32//5 = 6, extraIters = 32%5 = 2
        assert sched["itersPerWg"] == 6
        assert sched["extraIters"] == 2
        total = sched["totalIters"]
        ipw = sched["itersPerWg"]
        extra = sched["extraIters"]
        covered = set()
        for wg in range(5):
            start = wg * ipw + min(wg, extra)
            end = start + (ipw + 1 if wg < extra else ipw)
            assert start >= 0 and end <= total
            for i in range(start, end):
                assert i not in covered, f"iteration {i} covered twice"
                covered.add(i)
        assert covered == set(range(total))

    def test_k_divisibility_check(self):
        with pytest.raises(AssertionError, match="divisible"):
            compute_schedule(M=128, N=128, K=100, bm=128, bn=128, bk=64, grid=4)

    def test_single_wg(self):
        """Single WG gets all iterations."""
        sched = compute_schedule(M=128, N=128, K=256, bm=128, bn=128, bk=64, grid=1)
        assert sched["totalIters"] == 4
        assert sched["itersPerWg"] == 4
        assert sched["extraIters"] == 0

    def test_more_wgs_than_iters(self):
        """WGs beyond total iterations get zero iterations."""
        sched = compute_schedule(M=128, N=128, K=128, bm=128, bn=128, bk=32, grid=10)
        # totalTiles=1, itersPerTile=4, totalIters=4, grid=10
        # itersPerWg=0, extraIters=4
        assert sched["totalIters"] == 4
        assert sched["itersPerWg"] == 0
        assert sched["extraIters"] == 4
        total = sched["totalIters"]
        ipw = sched["itersPerWg"]
        extra = sched["extraIters"]
        covered = set()
        for wg in range(10):
            start = wg * ipw + min(wg, extra)
            end = start + (ipw + 1 if wg < extra else ipw)
            if end > start:
                for i in range(start, end):
                    assert i not in covered
                    covered.add(i)
        assert covered == set(range(total))

    def test_workspace_sizing_keys(self):
        """SlotsPerTile and workspaceElems are correct."""
        sched = compute_schedule(M=256, N=256, K=512, bm=128, bn=128, bk=64, grid=4)
        total_tiles = sched["totalTiles"]  # 4
        iters_per_tile = sched["itersPerTile"]  # 8
        assert sched["slotsPerTile"] == iters_per_tile
        assert sched["workspaceElems"] == total_tiles * sched["slotsPerTile"]
        # Verify against explicit values.
        assert sched["slotsPerTile"] == 512 // 64
        assert sched["workspaceElems"] == 4 * 8
