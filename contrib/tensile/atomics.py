# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Workspace store/load-sum helpers for the StreamK partial-tile epilogue."""

from __future__ import annotations

import aster._mlir_libs._amdgcn as _amdgcn_lib
from aster.dialects._amdgcn_ops_gen import v_accvgpr_read
from aster.layout import Layout, LayoutValues, Tensor, enumerate_flat_coords


def _agpr_to_vgpr(b, val):
    """Move a scalar AGPR value into a fresh VGPR via v_accvgpr_read_b32.

    global_store_dword only accepts VGPR for the data operand. Returns
    the value-semantic SSA result of the read instruction.
    """
    dst = b.alloca_vgpr()
    return v_accvgpr_read(dst, val, loc=b._loc, ip=b._kip)


def store_acc_to_workspace(
    b,
    ws_tensor: Tensor,
    acc_bundle: LayoutValues,
    tiled_copy_desc,
    *,
    unroll_axes: tuple,
) -> None:
    """Store accumulator fragments to the workspace buffer (plain stores).

    Mirrors the structure of atomic_add_f32_tile but writes directly instead
    of atomically adding. AGPR fragments are first moved to VGPRs because
    global_store_dword only accepts VGPR as the data operand.

    The caller is responsible for issuing b.wait_vmcnt(0) after this call
    to establish the release fence before the counter atomic.

    Args:
        b:              KernelBuilderWithLayouts instance.
        ws_tensor:      Destination workspace tensor for this producer's slot.
        acc_bundle:     LayoutValues holding one 4-AGPR (or 4-VGPR) range per
                        sub-tile coordinate.
        tiled_copy_desc: TiledCopy descriptor defining thread/value layout.
        unroll_axes:    Axes to unroll over (same as transfer_tiles).
    """
    tc = tiled_copy_desc
    L = ws_tensor.layout
    assert L is not None and L.axes is not None, (
        "store_acc_to_workspace expects a Tensor layout with axes"
    )

    offsets = b.thread_value_offsets(tc.pid, tc.thread_layout, tc.value_layout)
    n_per_tile = tc.value_layout.size()
    sub_layout = b._filter_layout_by_axes(L, unroll_axes)

    for coords in enumerate_flat_coords(sub_layout.flat_sizes):
        tile_rel = b.layout_apply(
            tuple(b.constant_index(c) for c in coords), sub_layout
        )
        tile_off = (
            tile_rel
            if ws_tensor.offset is None
            else b.layout_sum(ws_tensor.offset, tile_rel)
        )
        sub_tensor = Tensor(ws_tensor.ptr, tile_off)
        base = acc_bundle.data_at(coords)
        scalars = list(b.split_register_range(base, n_per_tile))
        vgprs = [
            _agpr_to_vgpr(b, s) if _amdgcn_lib.AGPRType.isinstance(s.type) else s
            for s in scalars
        ]
        for v, voff in enumerate(offsets):
            off = b.index_to_vgpr(sub_tensor.byte_offset(b, voff))
            b.global_store_dword(vgprs[v], sub_tensor.ptr, dynamic_offset=off, nt=True)


def load_and_sum_workspace(
    b,
    ws_tensor: Tensor,
    slot_byte_offsets: list,
    tiled_copy_desc,
    *,
    unroll_axes: tuple,
) -> LayoutValues:
    """Load and sum all workspace slots into per-lane VGPR accumulators.

    For each slot in slot_byte_offsets, loads all per-thread values via sc1
    global_load_dword (device-scope acquire) and accumulates them with
    v_add_f32. Returns a LayoutValues with the same structure as acc_bundle
    so the result can be passed directly to b.transfer_tiles.

    The caller is responsible for issuing b.wait_vmcnt(0) after this call
    before writing to C.

    Args:
        b:                  KernelBuilderWithLayouts instance.
        ws_tensor:          Workspace tensor (base ptr + tile-base layout).
        slot_byte_offsets:  list of ir.Value byte offsets, one per k-iter slot.
        tiled_copy_desc:    TiledCopy descriptor defining thread/value layout.
        unroll_axes:        Axes to unroll over (same as transfer_tiles).
    """
    tc = tiled_copy_desc
    L = ws_tensor.layout
    assert L is not None and L.axes is not None, (
        "load_and_sum_workspace expects a Tensor layout with axes"
    )

    offsets = b.thread_value_offsets(tc.pid, tc.thread_layout, tc.value_layout)
    n_per_tile = tc.value_layout.size()
    sub_layout = b._filter_layout_by_axes(L, unroll_axes)
    n_tiles = sub_layout.size()

    # Allocate VGPR running sums: n_tiles * n_per_tile VGPRx1 values, init to 0.
    c_zero = b.constant_i32(0)
    sums = [
        [b.vop1("v_mov_b32", c_zero) for _ in range(n_per_tile)] for _ in range(n_tiles)
    ]

    # Accumulate each slot into the running sums.
    # slot_byte_offsets[s] is the absolute byte offset from ws_ptr to the
    # start of slot s's tile region. ws_tensor.offset carries the
    # wave-level intra-slot offset (identical across all slots because the
    # slot layout mirrors C). We build a per-slot tensor by replacing only
    # the slot-base component, exactly as _make_ws_tensor_producer does for
    # the producer.
    for slot_byte_off in slot_byte_offsets:
        for ti, coords in enumerate(enumerate_flat_coords(sub_layout.flat_sizes)):
            tile_rel = b.layout_apply(
                tuple(b.constant_index(c) for c in coords), sub_layout
            )
            # slot_byte_off is the slot base; add the intra-slot (wave + tile)
            # offset the same way _make_ws_tensor_producer does.
            intra_off = (
                tile_rel
                if ws_tensor.offset is None
                else b.layout_sum(ws_tensor.offset, tile_rel)
            )
            slot_base_plus_intra = b.layout_sum(slot_byte_off, intra_off)
            sub_tensor = Tensor(ws_tensor.ptr, slot_base_plus_intra)
            # Issue sc1 loads (device-scope acquire) for all per-thread values.
            loaded = []
            for voff in offsets:
                elem_off = b.index_to_vgpr(sub_tensor.byte_offset(b, voff))
                val, _tok = b.global_load_dword(
                    sub_tensor.ptr, dynamic_offset=elem_off, sc1=True
                )
                loaded.append(val)
            b.wait_vmcnt(0)
            for v, val in enumerate(loaded):
                sums[ti][v] = b.vop2("v_add_f32", sums[ti][v], val)

    # Pack each tile's sums into a VGPRx4 range.
    payloads = []
    for ti in range(n_tiles):
        vx4 = b._make_register_range(sums[ti])
        payloads.append(vx4)

    acc_layout = Layout(sub_layout.flat_sizes)
    return LayoutValues.from_flat(acc_layout, payloads=tuple(payloads))
