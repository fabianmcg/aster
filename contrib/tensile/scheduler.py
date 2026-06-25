# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""StreamK tile/iteration scheduler.

Host-side math mirrors Tensile/Components/StreamK.py:361-388. IR
emission helpers produce the per-WG range decomposition.
"""

from __future__ import annotations

from aster import ir
from aster.dialects import arith


def compute_schedule(
    M: int, N: int, K: int, bm: int, bn: int, bk: int, grid: int
) -> dict:
    """Compute StreamK partition parameters (host-side).

    Args:
        M, N, K: Problem dimensions. K % bk == 0 is required.
        bm, bn:  Output tile sizes (rows, cols).
        bk:      K-tile size (K elements per k-iteration step).
        grid:    Number of persistent workgroups.

    Returns a dict with:
        totalTiles    -- total number of C output tiles = (M // bm) * (N // bn)
        itersPerTile  -- K-iterations per tile = K // bk
        totalIters    -- totalTiles * itersPerTile
        itersPerWg    -- base iterations per WG (floor)
        extraIters    -- number of WGs that get one extra iteration
    """
    assert K % bk == 0, f"k={K} must be divisible by bk={bk}"
    total_tiles = (M // bm) * (N // bn)
    iters_per_tile = K // bk
    total_iters = total_tiles * iters_per_tile
    iters_per_wg = total_iters // grid
    extra_iters = total_iters % grid
    return {
        "totalTiles": total_tiles,
        "itersPerTile": iters_per_tile,
        "totalIters": total_iters,
        "itersPerWg": iters_per_wg,
        "extraIters": extra_iters,
        "slotsPerTile": iters_per_tile,
        "workspaceElems": total_tiles * iters_per_tile,
    }


def emit_wg_iter_range(b, wg_id: ir.Value, sched: dict) -> tuple[ir.Value, ir.Value]:
    """Emit IR for the per-WG iteration range [iterStart, iterEnd).

    Even-split-plus-remainder: the first extraIters WGs each handle
    (itersPerWg + 1) iterations; the rest handle itersPerWg iterations.

    Args:
        b:       KernelBuilder instance.
        wg_id:   SSA index value for the workgroup ID.
        sched:   dict returned by compute_schedule.

    Returns (iterStart, iterEnd) as SSA index values.
    """
    iters_per_wg = sched["itersPerWg"]
    extra_iters = sched["extraIters"]
    total_iters = sched["totalIters"]

    c_ipw = b.constant_index(iters_per_wg)
    c_ipw1 = b.constant_index(iters_per_wg + 1)
    c_extra = b.constant_index(extra_iters)
    c_total = b.constant_index(total_iters)

    # Extra work units are distributed one per WG to the first extraIters
    # workgroups so that the iteration load is balanced as evenly as possible.
    is_extra = arith.cmpi(
        arith.CmpIPredicate.ult, wg_id, c_extra, loc=b._loc, ip=b._kip
    )

    # iter_start = wg_id < extra_iters ? wg_id*(ipw+1) : wg_id*ipw + extra_iters
    d0 = ir.AffineExpr.get_dim(0)
    start_extra = b.affine_apply(d0 * (iters_per_wg + 1), [wg_id])
    start_normal = b.affine_apply(d0 * iters_per_wg + extra_iters, [wg_id])
    iter_start = arith.select(
        is_extra, start_extra, start_normal, loc=b._loc, ip=b._kip
    )

    iter_end_unclamped = arith.addi(
        iter_start,
        arith.select(is_extra, c_ipw1, c_ipw, loc=b._loc, ip=b._kip),
        loc=b._loc,
        ip=b._kip,
    )
    # Clamp iter_end to total_iters in case of rounding.
    iter_end = arith.minui(iter_end_unclamped, c_total, loc=b._loc, ip=b._kip)

    return iter_start, iter_end


def emit_tile_decompose(
    b,
    k_iter: ir.Value,
    iters_per_tile: int,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Decompose a global iteration index into tile + local range.

    Args:
        b:              KernelBuilder instance.
        k_iter:         Global iteration index (SSA index value).
        iters_per_tile: K-iterations per output tile (static constant).

    Returns:
        tile       -- tile index (k_iter // iters_per_tile)
        localStart -- first iteration within the tile (k_iter % iters_per_tile)
        tileEnd    -- global iteration index of the start of the next tile
    """
    c_ipt = b.constant_index(iters_per_tile)
    tile = arith.divui(k_iter, c_ipt, loc=b._loc, ip=b._kip)
    local_start = arith.remui(k_iter, c_ipt, loc=b._loc, ip=b._kip)
    # tileEnd is the first iteration of the next tile.
    d0 = ir.AffineExpr.get_dim(0)
    tile_end = b.affine_apply(d0 * iters_per_tile + iters_per_tile, [tile])
    return tile, local_start, tile_end
