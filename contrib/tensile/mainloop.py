# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Reusable GEMM mainloop with dynamic K-range for StreamK."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from aster import ir
from aster.dialects import arith as _arith
from aster.layout import Layout, Symbol, Tensor, enumerate_flat_coords


# ---------------------------------------------------------------------------
# Cooperative-load helpers (self-contained copy from kittens/test/coop.py)
# ---------------------------------------------------------------------------


def _coop_2d_split(num_tiles: int, num_waves: int, kt: int):
    """Pick (waves_s, waves_k, coop_s, coop_k) minimising clamping waste."""
    best = None
    for ws in range(1, num_waves + 1):
        if num_waves % ws != 0:
            continue
        wk = num_waves // ws
        cs = math.ceil(num_tiles / ws)
        ck = math.ceil(kt / wk)
        wasted = ws * wk * cs * ck - num_tiles * kt
        over_excess = max(0, ws - num_tiles) + max(0, wk - kt)
        key = (wasted, over_excess, -ws)
        if best is None or key < best[0]:
            best = (key, ws, wk, cs, ck)
    _, waves_s, waves_k, coop_s, coop_k = best
    return waves_s, waves_k, coop_s, coop_k


@dataclass(frozen=True, slots=True)
class CoopLoadPlan:
    """Per-operand cooperative-load plan."""

    global_layout: Layout
    global_wave_off: ir.Value
    lds_layout: Layout
    lds_wave_off: ir.Value
    unroll_axes: tuple


def make_coop_load_plan(
    b,
    wid: ir.Value,
    *,
    num_waves: int,
    wg_tile_global: Layout,
    wg_tile_lds: Layout,
    spatial_axis: Symbol,
    k_axis: Symbol,
) -> CoopLoadPlan:
    """Build a CoopLoadPlan from per-tile WG-level Layouts."""
    num_tiles, kt = wg_tile_global.sizes
    s_stride_g, k_stride_g = wg_tile_global.strides
    s_stride_l, k_stride_l = wg_tile_lds.strides
    assert wg_tile_lds.sizes == (num_tiles, kt)

    waves_s, waves_k, coop_s, coop_k = _coop_2d_split(num_tiles, num_waves, kt)
    max_s_start = max(0, num_tiles - coop_s)
    max_k_start = max(0, kt - coop_k)

    per_wave_global = Layout(
        (coop_s, coop_k), (s_stride_g, k_stride_g), axes=(spatial_axis, k_axis)
    )
    per_wave_lds = Layout(
        (coop_s, coop_k), (s_stride_l, k_stride_l), axes=(spatial_axis, k_axis)
    )

    wave_s_idx, wave_k_idx = b.delinearize_index(wid, (waves_s, waves_k))
    d0 = ir.AffineExpr.get_dim(0)

    s_start = b.arith_minui(
        b.affine_apply(d0 * coop_s, [wave_s_idx]), b.constant_index(max_s_start)
    )
    k_start = b.arith_minui(
        b.affine_apply(d0 * coop_k, [wave_k_idx]), b.constant_index(max_k_start)
    )
    wg_global_full = Layout(
        (num_tiles, kt), (s_stride_g, k_stride_g), axes=(spatial_axis, k_axis)
    )
    wg_lds_full = Layout(
        (num_tiles, kt), (s_stride_l, k_stride_l), axes=(spatial_axis, k_axis)
    )
    global_wave_off = b.layout_apply((s_start, k_start), wg_global_full)
    lds_wave_off = b.layout_apply((s_start, k_start), wg_lds_full)

    return CoopLoadPlan(
        global_layout=per_wave_global,
        global_wave_off=global_wave_off,
        lds_layout=per_wave_lds,
        lds_wave_off=lds_wave_off,
        unroll_axes=(spatial_axis, k_axis),
    )


# ---------------------------------------------------------------------------
# Mainloop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MainloopConfig:
    """Static configuration for build_mainloop."""

    # Layouts
    mfma_opcode: str  # e.g. "v_mfma_f32_16x16x16_f16"
    mfma_m: int
    mfma_n: int
    mfma_k: int
    tile_k_elems: int  # k-elements per k-tile (e.g. 32 for 2x16)
    elt_bytes: int  # bytes per element (2 for f16/bf16)
    # Per-wave tile counts
    m_per_wave: int
    n_per_wave: int
    k_t: int  # k-tile count
    # Number of waves
    num_waves: int
    wave_size: int
    # Copy descriptors (TiledCopy objects)
    tc_load_a: Any
    tc_load_b: Any
    tc_dsw_a: Any
    tc_dsw_b: Any
    tc_dsr_a: Any
    tc_dsr_b: Any
    # Axes
    m_axis: Symbol
    n_axis: Symbol
    k_tile_axis: Symbol
    global_k_axis: Symbol
    wave_m_axis: Symbol
    wave_n_axis: Symbol
    m_load_a_axis: Symbol
    k_load_a_axis: Symbol
    n_load_b_axis: Symbol
    k_load_b_axis: Symbol
    # LDS sizes
    lds_total_a: int
    lds_total_b: int
    # Coop-load plans
    plan_a: CoopLoadPlan
    plan_b: CoopLoadPlan


def build_mainloop(
    b,
    *,
    a_ptr: ir.Value,
    b_ptr: ir.Value,
    TA: Tensor,
    TB: Tensor,
    lds_a_layout,
    lds_b_layout,
    k_iter_start: ir.Value,
    k_iter_end: ir.Value,
    acc_inits: list[ir.Value],
    cfg: MainloopConfig,
    wave_m_idx: ir.Value,
    wave_n_idx: ir.Value,
    k_offset: ir.Value = None,
) -> list[ir.Value]:
    """Emit the GEMM k-loop from k_iter_start to k_iter_end.

    k_offset is an optional SSA index added to the loop induction
    variable before slicing TA/TB; this allows the persistent StreamK
    kernel to shift the k-window to the correct position within an
    output tile.

    Returns the list of final accumulator values (same length as
    acc_inits).
    """
    c = cfg
    n_frags = c.tile_k_elems // c.mfma_k

    accs_final: list[ir.Value] = []
    c1 = b.constant_index(1)

    @b.loop(k_iter_start, k_iter_end, c1, iter_args=acc_inits, results=accs_final)
    def _body(k_iv, *accs):
        accs = list(accs)

        lds_a_h, sA_full = b.alloc_lds_tensor(c.lds_total_a, layout=lds_a_layout)
        k_global = (
            _arith.addi(k_iv, k_offset, loc=b._loc, ip=b._kip)
            if k_offset is not None
            else k_iv
        )
        ta_iter = b.slice(TA, {c.global_k_axis: k_global})
        ta_load = Tensor(
            a_ptr,
            b.layout_sum(ta_iter.offset, c.plan_a.global_wave_off),
            c.plan_a.global_layout,
        )
        a_load = b.transfer_tiles(
            ta_load, c.tc_load_a, unroll_axes=c.plan_a.unroll_axes
        )

        lds_b_h, sB_full = b.alloc_lds_tensor(c.lds_total_b, layout=lds_b_layout)
        tb_iter = b.slice(TB, {c.global_k_axis: k_global})
        tb_load = Tensor(
            b_ptr,
            b.layout_sum(tb_iter.offset, c.plan_b.global_wave_off),
            c.plan_b.global_layout,
        )
        b_load = b.transfer_tiles(
            tb_load, c.tc_load_b, unroll_axes=c.plan_b.unroll_axes
        )

        b.wait_deps(a_load)
        sA_write = Tensor(sA_full.ptr, c.plan_a.lds_wave_off, c.plan_a.lds_layout)
        a_write = b.transfer_tiles(
            sA_write, c.tc_dsw_a, unroll_axes=c.plan_a.unroll_axes, data=a_load
        )

        b.wait_deps(b_load)
        sB_write = Tensor(sB_full.ptr, c.plan_b.lds_wave_off, c.plan_b.lds_layout)
        b_write = b.transfer_tiles(
            sB_write, c.tc_dsw_b, unroll_axes=c.plan_b.unroll_axes, data=b_load
        )

        b.wait_deps(a_write)
        b.barrier()
        sA_read = b.slice(sA_full, {c.wave_m_axis: wave_m_idx})
        a_frags = b.transfer_tiles(
            sA_read, c.tc_dsr_a, unroll_axes=(c.m_axis, c.k_tile_axis)
        )

        b.wait_deps(b_write)
        b.barrier()
        sB_read = b.slice(sB_full, {c.wave_n_axis: wave_n_idx})
        b_frags = b.transfer_tiles(
            sB_read, c.tc_dsr_b, unroll_axes=(c.n_axis, c.k_tile_axis)
        )

        b.wait_deps(a_frags, b_frags)
        for fi, ki, mi, ni in enumerate_flat_coords(
            (n_frags, c.k_t, c.m_per_wave, c.n_per_wave)
        ):
            ai = mi * c.n_per_wave + ni
            a_d = a_frags.data_at((mi, ki, fi))
            b_d = b_frags.data_at((ni, ki, fi))
            accs[ai] = b.mfma(c.mfma_opcode, accs[ai], a_d, b_d)
        b.dealloc_lds(lds_a_h)
        b.dealloc_lds(lds_b_h)
        return accs

    return accs_final
