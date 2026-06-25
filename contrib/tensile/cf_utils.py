# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Control-flow utility helpers for kernel builders."""

from __future__ import annotations

from aster import ir


def emit_cf_while_loop(b, init_args: list, cond_fn, body_fn) -> None:
    """Emit a while loop using CF blocks directly.

    Avoids scf.while which is not supported by the amdgcn-convert-scf-
    control-flow pass.  The loop structure is:

    (current block tail) -> cf.cond_br -> bbBody / bbEnd bbBody ->
    body_fn -> cf.cond_br -> bbBody / bbEnd bbEnd  -> (continues in
    b._kip after return)

    cond_fn(block_args) -> i1 condition value. body_fn(block_args) ->
    list of updated values for the next iteration.

    After return, b._kip is set to bbEnd so callers can emit post-loop
    code.
    """
    arg_types = [a.type for a in init_args]
    arg_locs = [b._loc] * len(arg_types)

    current_block = b._kip.block

    # Create bbEnd after the current block (empty; continues after the loop).
    bb_end = current_block.create_after()
    # Create bbBody before bbEnd (so bbBody precedes bbEnd in the block list).
    bb_body = bb_end.create_before(*arg_types, arg_locs=arg_locs)

    # operandSegmentSizes for cf.cond_br: [1 (cond), n_true_args, n_false_args].
    # The true branch takes the loop variable(s); the false branch (exit) takes none.
    n_args = len(arg_types)
    seg_sizes = ir.DenseI32ArrayAttr.get([1, n_args, 0])

    # Emit initial condition check at the tail of the current block.
    init_cond = cond_fn(init_args)
    ir.Operation.create(
        "cf.cond_br",
        operands=[init_cond] + list(init_args),
        successors=[bb_body, bb_end],
        attributes={"operandSegmentSizes": seg_sizes},
        loc=b._loc,
        ip=b._kip,
    )

    # Emit the loop body in bbBody.
    b._kip = ir.InsertionPoint.at_block_begin(bb_body)
    body_args = list(bb_body.arguments)
    updated = body_fn(body_args)

    # Emit back-edge condition and branch.
    back_cond = cond_fn(updated)
    ir.Operation.create(
        "cf.cond_br",
        operands=[back_cond] + list(updated),
        successors=[bb_body, bb_end],
        attributes={"operandSegmentSizes": seg_sizes},
        loc=b._loc,
        ip=b._kip,
    )

    # Set insertion point to bbEnd for post-loop code.
    b._kip = ir.InsertionPoint.at_block_begin(bb_end)
