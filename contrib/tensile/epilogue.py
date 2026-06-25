# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""StreamK epilogue: full-tile store vs. workspace + counter-gated consumer."""

from __future__ import annotations

from aster import ir
from aster.dialects import arith
from aster.dialects import lsir as _lsird
from aster.layout import LayoutValues, Layout, Tensor

from tensile.atomics import store_acc_to_workspace, load_and_sum_workspace


def _scf_if_else(b, condition: ir.Value, then_fn, else_fn) -> None:
    """Emit an scf.if with both then and else regions."""
    if_op = ir.Operation.create(
        "scf.if",
        results=[],
        operands=[condition],
        regions=2,
        loc=b._loc,
        ip=b._kip,
    )
    saved_ip = b._kip

    # Retarget the builder's insertion point into each region; restored below.
    then_block = ir.Block.create_at_start(if_op.regions[0], [], [])
    b._kip = ir.InsertionPoint(then_block)
    then_fn()
    ir.Operation.create("scf.yield", operands=[], loc=b._loc, ip=b._kip)

    else_block = ir.Block.create_at_start(if_op.regions[1], [], [])
    b._kip = ir.InsertionPoint(else_block)
    else_fn()
    ir.Operation.create("scf.yield", operands=[], loc=b._loc, ip=b._kip)

    b._kip = saved_ip


def flush_tile(
    b,
    *,
    c_tensor: Tensor,
    ws_tensor: Tensor,
    ws_consumer_tensor: Tensor,
    counter_ptr,
    counter_byte_offset,
    accs: list[ir.Value],
    store_desc,
    ws_store_desc,
    ws_load_desc,
    acc_layout: Layout,
    local_start: ir.Value,
    local_end: ir.Value,
    iters_per_tile: int,
    slot_byte_offsets: list,
    m_axis,
    n_axis,
) -> None:
    """Epilogue: full-tile store or workspace + counter-gated consumer finalize.

    Case A (is_start_zero AND is_end_full): this WG owns the whole tile; store
    the accumulator directly to C.

    Case B/C (partial tile): write the accumulator to the workspace buffer
    (one slot per k-iteration), then atomically increment the tile's i32
    counter by the number of k-iterations this WG contributed. The WG whose
    increment brings the counter to iters_per_tile becomes the consumer and
    reads all workspace slots, sums them, and writes C via plain stores.

    Args:
        b:                  KernelBuilderWithLayouts instance.
        c_tensor:           Destination C tensor.
        ws_tensor:          Workspace tensor rooted at this WG's producer slot.
        ws_consumer_tensor: Workspace tensor carrying only the wave-level intra-slot
                            offset (no slot/tile base); used for consumer loads.
        counter_ptr:        Base ptr to the per-tile i32 counter buffer.
        counter_byte_offset: Byte offset of this tile's counter within counter_ptr.
        accs:               Flat list of 4-AGPR accumulator ranges (from mainloop).
        store_desc:         TiledCopy descriptor for the plain C store path.
        ws_store_desc:      TiledCopy descriptor for the workspace store path.
        ws_load_desc:       TiledCopy descriptor for the workspace load path.
        acc_layout:         Layout describing the accumulator grid.
        local_start:        SSA index: first k-iteration within this tile.
        local_end:          SSA index: past-the-end k-iteration within this tile.
        iters_per_tile:     Static constant: total k-iterations per tile.
        slot_byte_offsets:  list of ir.Value byte offsets for all slots (consumer).
        m_axis:             Symbol for the m output axis.
        n_axis:             Symbol for the n output axis.
    """
    c_ipt = b.constant_index(iters_per_tile)
    c_zero = b.constant_index(0)

    acc_bundle = LayoutValues.from_flat(acc_layout, payloads=tuple(accs))

    def _store():
        b.transfer_tiles(
            c_tensor, store_desc, unroll_axes=(m_axis, n_axis), data=acc_bundle
        )

    def _partial():
        # Store accumulator to this WG's workspace slot.
        store_acc_to_workspace(
            b, ws_tensor, acc_bundle, ws_store_desc, unroll_axes=(m_axis, n_axis)
        )
        # Release fence: all workspace stores must be visible before the atomic.
        b.wait_vmcnt(0)

        # Compute number of k-iterations contributed by this WG.
        contributed_idx = arith.subi(local_end, local_start, loc=b._loc, ip=b._kip)

        # Issue the counter atomic from all threads but only thread 0 passes a
        # non-zero value, so the counter advances by exactly `contributed`.
        # arith.select on i1 avoids lsir.extui which is disallowed in normal form.
        is_thread0 = arith.cmpi(
            arith.CmpIPredicate.eq,
            b.linear_thread_id(),
            b.constant_index(0),
            loc=b._loc,
            ip=b._kip,
        )
        # thread 0: contributed_idx; others: 0. Both index-typed for index_to_vgpr.
        atomic_data_idx = b.select(is_thread0, contributed_idx, b.constant_index(0))
        atomic_data_vgpr = b.index_to_vgpr(atomic_data_idx)

        old_v, _tok = b.global_atomic_add_ret(
            atomic_data_vgpr,
            counter_ptr,
            dynamic_offset=b.index_to_vgpr(counter_byte_offset),
            sc1=True,
        )
        b.wait_vmcnt(0)

        # Thread 0 holds the real pre-op counter value in old_v; broadcast to
        # all lanes via v_readfirstlane (thread 0 is always first in wave 0).
        old_val_sgpr = b.v_readfirstlane(old_v)
        i32_type = ir.IntegerType.get_signless(32, b._ctx)
        old_val_i32 = _lsird.from_reg(i32_type, old_val_sgpr, loc=b._loc, ip=b._kip)
        old_val_idx = arith.index_cast(b.idx_type, old_val_i32, loc=b._loc, ip=b._kip)

        # Determine if this WG is the consumer (its increment completes the tile).
        old_plus_contrib = arith.addi(
            old_val_idx, contributed_idx, loc=b._loc, ip=b._kip
        )
        is_consumer_raw = arith.cmpi(
            arith.CmpIPredicate.eq, old_plus_contrib, c_ipt, loc=b._loc, ip=b._kip
        )
        is_consumer = b.assume_uniform(is_consumer_raw)

        @b.scf_if(is_consumer)
        def _():
            # Acquire fence: sc1 loads in load_and_sum_workspace ensure
            # producers' stores are visible before we sum.
            summed = load_and_sum_workspace(
                b,
                ws_consumer_tensor,
                slot_byte_offsets,
                ws_load_desc,
                unroll_axes=(m_axis, n_axis),
            )
            b.wait_vmcnt(0)
            b.transfer_tiles(
                c_tensor, store_desc, unroll_axes=(m_axis, n_axis), data=summed
            )

    # is_full = (local_start == 0) AND (local_end == iters_per_tile).
    # Implemented as nested scf.if to avoid arith.andi on i1 (no lowering).
    is_start_zero = arith.cmpi(
        arith.CmpIPredicate.eq, local_start, c_zero, loc=b._loc, ip=b._kip
    )

    def _check_end():
        is_end_full = arith.cmpi(
            arith.CmpIPredicate.eq, local_end, c_ipt, loc=b._loc, ip=b._kip
        )
        _scf_if_else(b, is_end_full, _store, _partial)

    _scf_if_else(b, is_start_zero, _check_end, _partial)
