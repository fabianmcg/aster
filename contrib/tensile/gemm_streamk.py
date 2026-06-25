# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""StreamK GEMM kernel builder."""

from __future__ import annotations

from aster import ir
from aster.layout import Layout, LayoutValues, Swizzle, Symbol, Tensor, tile
from aster.dialects.amdgcn import AccessKind
from aster.dialects.kernel_builder_with_layouts import (
    ds_read_64b,
    ds_write_64b,
    global_load_dwordx4,
    global_store_dword,
    KernelBuilderWithLayouts as KernelBuilder,
)

from tensile.mainloop import (
    MainloopConfig,
    make_coop_load_plan,
    build_mainloop,
)
from tensile.epilogue import flush_tile
from tensile.scheduler import compute_schedule, emit_wg_iter_range, emit_tile_decompose
from tensile.cf_utils import emit_cf_while_loop as _emit_cf_while_loop
from aster.dialects import arith as _arith
from aster.core.target import Target

# MFMA tile shape for 16x16x16 f16/bf16.
_MFMA_M = 16
_MFMA_N = 16
_MFMA_K = 16
_TILE_K_ELEMS = 32
_ELT_BYTES = 2
_TILE_BYTES_A = _MFMA_M * _TILE_K_ELEMS * _ELT_BYTES  # 1024
_TILE_BYTES_B = _MFMA_N * _TILE_K_ELEMS * _ELT_BYTES  # 1024
_LDS_SWIZZLE = Swizzle(bits=3, base=3, shift=3)
_WAVE_SIZE = 64

# Axes.
_m = Symbol("m")
_n = Symbol("n")
_k_tile = Symbol("k_tile")
_global_k = Symbol("global_k")
_wg_m = Symbol("wg_m")
_wg_n = Symbol("wg_n")
_wave_m = Symbol("wave_m")
_wave_n = Symbol("wave_n")
_m_load_a = Symbol("m_load_a")
_k_load_a = Symbol("k_load_a")
_n_load_b = Symbol("n_load_b")
_k_load_b = Symbol("k_load_b")


def _mfma_opcode(dtype: str) -> str:
    """Return MFMA opcode for the given dtype."""
    if dtype in ("f16", "float16"):
        return "v_mfma_f32_16x16x16_f16"
    if dtype in ("bf16", "bfloat16"):
        return "v_mfma_f32_16x16x16_bf16"
    raise ValueError(f"unsupported dtype: {dtype!r}")


def _check_lds_capacity(target: str, m_t: int, n_t: int, k_t: int) -> None:
    """Reject tile configs whose LDS footprint exceeds the target's capacity.

    Computes the single-buffer footprint for the A and B staging tiles.
    The SCF pipeline may multi-buffer LDS, so actual usage can be
    higher; this guard only catches configs that cannot fit even
    unbuffered, converting an opaque assemble/launch failure into an
    early, actionable error.  gfx942 (CDNA3) has 64 KB of LDS per CU
    versus 160 KB on gfx950 (CDNA4).
    """
    lds_bytes = m_t * k_t * _TILE_BYTES_A + n_t * k_t * _TILE_BYTES_B
    lds_limit = Target.from_mcpu(target).lds_per_cu
    assert lds_bytes <= lds_limit, (
        f"LDS footprint {lds_bytes} bytes exceeds {target} capacity "
        f"{lds_limit} bytes; reduce bm/bn/bk or use a target with more LDS"
    )


def _make_copy_descriptors(b, stride_a: int, stride_b: int, stride_c: int):
    """Build the six tiled-copy descriptors for global→LDS and LDS→MFMA."""
    tc_load_a = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout((_MFMA_M, 4), (stride_a, 16)),
        value_layout=Layout(1, 0),
    )
    tc_load_b = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout((_MFMA_N, 4), (stride_b, 16)),
        value_layout=Layout(1, 0),
    )
    tc_dsw_a = b.make_tiled_copy_descriptor(
        ds_write_64b,
        thread_layout=Layout((_MFMA_M, 4), (64, 16)),
        value_layout=Layout(_TILE_K_ELEMS // _MFMA_K, 8),
        swizzle=_LDS_SWIZZLE,
    )
    tc_dsw_b = b.make_tiled_copy_descriptor(
        ds_write_64b,
        thread_layout=Layout((_MFMA_N, 4), (64, 16)),
        value_layout=Layout(_TILE_K_ELEMS // _MFMA_K, 8),
        swizzle=_LDS_SWIZZLE,
    )
    tc_dsr_a = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((4, _MFMA_M), (8, 64)),
        value_layout=Layout(_TILE_K_ELEMS // _MFMA_K, _MFMA_K * _ELT_BYTES),
        swizzle=_LDS_SWIZZLE,
    )
    tc_dsr_b = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((4, _MFMA_N), (8, 64)),
        value_layout=Layout(_TILE_K_ELEMS // _MFMA_K, _MFMA_K * _ELT_BYTES),
        swizzle=_LDS_SWIZZLE,
    )
    tc_store_c = b.make_tiled_copy_descriptor(
        global_store_dword,
        thread_layout=Layout((4, _MFMA_N), (4 * stride_c, 4)),
        value_layout=Layout(4, stride_c),
    )
    return tc_load_a, tc_load_b, tc_dsw_a, tc_dsw_b, tc_dsr_a, tc_dsr_b, tc_store_c


def _make_mainloop_config(
    dtype: str,
    m_t: int,
    n_t: int,
    k_t: int,
    nw: int,
    wpw_m: int,
    wpw_n: int,
    tc_load_a,
    tc_load_b,
    tc_dsw_a,
    tc_dsw_b,
    tc_dsr_a,
    tc_dsr_b,
    plan_a,
    plan_b,
) -> MainloopConfig:
    """Assemble a MainloopConfig from pre-built descriptors and plans."""
    m_per_wave = m_t // wpw_m
    n_per_wave = n_t // wpw_n
    lds_total_a = m_t * k_t * _TILE_BYTES_A
    lds_total_b = n_t * k_t * _TILE_BYTES_B
    return MainloopConfig(
        mfma_opcode=_mfma_opcode(dtype),
        mfma_m=_MFMA_M,
        mfma_n=_MFMA_N,
        mfma_k=_MFMA_K,
        tile_k_elems=_TILE_K_ELEMS,
        elt_bytes=_ELT_BYTES,
        m_per_wave=m_per_wave,
        n_per_wave=n_per_wave,
        k_t=k_t,
        num_waves=nw,
        wave_size=_WAVE_SIZE,
        tc_load_a=tc_load_a,
        tc_load_b=tc_load_b,
        tc_dsw_a=tc_dsw_a,
        tc_dsw_b=tc_dsw_b,
        tc_dsr_a=tc_dsr_a,
        tc_dsr_b=tc_dsr_b,
        m_axis=_m,
        n_axis=_n,
        k_tile_axis=_k_tile,
        global_k_axis=_global_k,
        wave_m_axis=_wave_m,
        wave_n_axis=_wave_n,
        m_load_a_axis=_m_load_a,
        k_load_a_axis=_k_load_a,
        n_load_b_axis=_n_load_b,
        k_load_b_axis=_k_load_b,
        lds_total_a=lds_total_a,
        lds_total_b=lds_total_b,
        plan_a=plan_a,
        plan_b=plan_b,
    )


def _build_cfg(
    b,
    m_t: int,
    n_t: int,
    k_t: int,
    wpw_m: int,
    wpw_n: int,
    stride_a: int,
    stride_b: int,
    stride_c: int,
    dtype: str,
) -> tuple[MainloopConfig, object]:
    """Build MainloopConfig, cooperative-load plans, and store descriptor."""
    nw = wpw_m * wpw_n

    tc_load_a, tc_load_b, tc_dsw_a, tc_dsw_b, tc_dsr_a, tc_dsr_b, tc_store_c = (
        _make_copy_descriptors(b, stride_a, stride_b, stride_c)
    )

    wid = b.wave_id()
    plan_a = make_coop_load_plan(
        b,
        wid,
        num_waves=nw,
        wg_tile_global=Layout(
            (m_t, k_t), (_MFMA_M * stride_a, _TILE_K_ELEMS * _ELT_BYTES)
        ),
        wg_tile_lds=Layout((m_t, k_t), (k_t * _TILE_BYTES_A, _TILE_BYTES_A)),
        spatial_axis=_m_load_a,
        k_axis=_k_load_a,
    )
    plan_b = make_coop_load_plan(
        b,
        wid,
        num_waves=nw,
        wg_tile_global=Layout(
            (n_t, k_t), (_MFMA_N * stride_b, _TILE_K_ELEMS * _ELT_BYTES)
        ),
        wg_tile_lds=Layout((n_t, k_t), (k_t * _TILE_BYTES_B, _TILE_BYTES_B)),
        spatial_axis=_n_load_b,
        k_axis=_k_load_b,
    )

    cfg = _make_mainloop_config(
        dtype,
        m_t,
        n_t,
        k_t,
        nw,
        wpw_m,
        wpw_n,
        tc_load_a,
        tc_load_b,
        tc_dsw_a,
        tc_dsw_b,
        tc_dsr_a,
        tc_dsr_b,
        plan_a,
        plan_b,
    )
    return cfg, tc_store_c


def _make_global_layouts(
    M,
    N,
    K,
    m_t,
    n_t,
    k_t,
    m_per_wave,
    n_per_wave,
    wpw_m,
    wpw_n,
    stride_a,
    stride_b,
    stride_c,
    bn,
):
    """Build tiled Layout objects for A, B, C, workspace, and LDS buffers."""
    A_TILED = tile(
        Layout((M, K), (stride_a, _ELT_BYTES)),
        tile_sizes=((_MFMA_M, m_t), (_TILE_K_ELEMS, k_t)),
        axes=((_m, _wg_m), (_k_tile, _global_k)),
    )
    B_TILED = tile(
        Layout((N, K), (stride_b, _ELT_BYTES)),
        tile_sizes=((_MFMA_N, n_t), (_TILE_K_ELEMS, k_t)),
        axes=((_n, _wg_n), (_k_tile, _global_k)),
    )
    C_TILED = tile(
        Layout((M, N), (stride_c, 4)),
        tile_sizes=((_MFMA_M, m_per_wave, wpw_m), (_MFMA_N, n_per_wave, wpw_n)),
        axes=((_m, _wave_m, _wg_m), (_n, _wave_n, _wg_n)),
    )
    # Workspace tile layout: same element structure as C but with compact
    # row stride (bn*4) since workspace tiles are packed contiguously.
    WS_TILED = tile(
        Layout((M, N), (bn * 4, 4)),
        tile_sizes=((_MFMA_M, m_per_wave, wpw_m), (_MFMA_N, n_per_wave, wpw_n)),
        axes=((_m, _wave_m, _wg_m), (_n, _wave_n, _wg_n)),
    )
    LDS_A_TILED = tile(
        Layout((m_t, k_t * _TILE_BYTES_A), (k_t * _TILE_BYTES_A, 1)),
        tile_sizes=((1, m_per_wave), _TILE_BYTES_A),
        axes=((_m, _wave_m), _k_tile),
    )
    LDS_B_TILED = tile(
        Layout((n_t, k_t * _TILE_BYTES_B), (k_t * _TILE_BYTES_B, 1)),
        tile_sizes=((1, n_per_wave), _TILE_BYTES_B),
        axes=((_n, _wave_n), _k_tile),
    )
    return A_TILED, B_TILED, C_TILED, WS_TILED, LDS_A_TILED, LDS_B_TILED


def _make_ws_tensor_producer(
    b, ws_ptr, TWS, tile_id, local_start, ws_tile_bytes, ws_slot_bytes
):
    """Build the workspace Tensor rooted at this producer's slot.

    The resulting offset positions stores at:
      ws_ptr + tile_id * ws_tile_bytes + local_start * ws_slot_bytes + wave_offset + elem_offset
    """
    d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
    prod_slot_start = b.affine_apply(
        d0 * ws_tile_bytes + d1 * ws_slot_bytes, [tile_id, local_start]
    )
    new_off = (
        b.layout_sum(prod_slot_start, TWS.offset)
        if TWS.offset is not None
        else prod_slot_start
    )
    return Tensor(ws_ptr, new_off, TWS.layout)


def _make_slot_byte_offsets(b, tile_id, iters_per_tile, ws_tile_bytes, ws_slot_bytes):
    """Return per-slot byte offsets from ws_ptr for the consumer to read.

    slot_byte_offsets[s] = tile_id * ws_tile_bytes + s * ws_slot_bytes
    """
    d0 = ir.AffineExpr.get_dim(0)
    tile_ws_base = b.affine_apply(d0 * ws_tile_bytes, [tile_id])
    result = []
    for s in range(iters_per_tile):
        if s == 0:
            result.append(tile_ws_base)
        else:
            slot_off = b.affine_apply(
                ir.AffineExpr.get_dim(0) + s * ws_slot_bytes, [tile_ws_base]
            )
            result.append(slot_off)
    return result


def build_dp_gemm(
    M: int,
    N: int,
    K: int,
    bm: int = 128,
    bn: int = 128,
    bk: int = 64,
    dtype: str = "f16",
    target: str = "gfx942",
    kernel_name: str = "dp_gemm",
) -> ir.Module:
    """Build a data-parallel GEMM kernel (single WG per output tile).

    K % bk == 0 is required.  All K iterations are handled by a single
    call to build_mainloop.
    """
    assert M % bm == 0, f"m={M} must be divisible by bm={bm}"
    assert N % bn == 0, f"n={N} must be divisible by bn={bn}"
    assert K % bk == 0, f"k={K} must be divisible by bk={bk}"

    m_t = bm // _MFMA_M  # tiles along m
    n_t = bn // _MFMA_N  # tiles along n
    k_t = bk // _TILE_K_ELEMS  # k-tiles per iteration

    assert k_t >= 1, f"bk={bk} must be >= tile_k_elems={_TILE_K_ELEMS}"
    _check_lds_capacity(target, m_t, n_t, k_t)

    k_step = k_t * _TILE_K_ELEMS
    k_iters = K // k_step

    wpw_m, wpw_n = 2, 2
    nw = wpw_m * wpw_n
    num_threads = nw * _WAVE_SIZE
    m_per_wave = m_t // wpw_m
    n_per_wave = n_t // wpw_n

    wg_m_count = M // bm
    wg_n_count = N // bn
    stride_a = K * _ELT_BYTES
    stride_b = K * _ELT_BYTES
    stride_c = N * 4  # f32 output

    b = KernelBuilder(f"{kernel_name}_mod", kernel_name, target=target)
    b.set_grid_dims(wg_m_count * wg_n_count)
    b.set_block_dims(num_threads)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    cfg, tc_store_c = _build_cfg(
        b, m_t, n_t, k_t, wpw_m, wpw_n, stride_a, stride_b, stride_c, dtype
    )

    A_TILED, B_TILED, C_TILED, _WS_TILED, LDS_A_TILED, LDS_B_TILED = (
        _make_global_layouts(
            M,
            N,
            K,
            m_t,
            n_t,
            k_t,
            m_per_wave,
            n_per_wave,
            wpw_m,
            wpw_n,
            stride_a,
            stride_b,
            stride_c,
            bn,
        )
    )

    wg_m_idx, wg_n_idx = b.delinearize_index(
        b.linear_block_id(), (wg_m_count, wg_n_count)
    )
    wid = b.wave_id()
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (wpw_m, wpw_n))

    TA = b.slice(Tensor(a_ptr, layout=A_TILED), {_wg_m: wg_m_idx})
    TB = b.slice(Tensor(b_ptr, layout=B_TILED), {_wg_n: wg_n_idx})
    TC = b.slice(
        Tensor(c_ptr, layout=C_TILED),
        {_wg_m: wg_m_idx, _wg_n: wg_n_idx, _wave_m: wave_m_idx, _wave_n: wave_n_idx},
    )

    n_accs = m_per_wave * n_per_wave
    acc_inits = [b.init_agprx4(b.constant_i32(0)) for _ in range(n_accs)]

    c0 = b.constant_index(0)
    k_end = b.constant_index(k_iters)

    accs_final = build_mainloop(
        b,
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        TA=TA,
        TB=TB,
        lds_a_layout=LDS_A_TILED,
        lds_b_layout=LDS_B_TILED,
        k_iter_start=c0,
        k_iter_end=k_end,
        acc_inits=acc_inits,
        cfg=cfg,
        wave_m_idx=wave_m_idx,
        wave_n_idx=wave_n_idx,
    )

    accs_bundle = LayoutValues.from_flat(
        Layout((m_per_wave, n_per_wave)), payloads=tuple(accs_final)
    )
    b.transfer_tiles(TC, tc_store_c, unroll_axes=(_m, _n), data=accs_bundle)
    return b.build()


def build_partial_gemm(
    M: int,
    N: int,
    K: int,
    k_start_iter: int,
    k_end_iter: int,
    bm: int = 128,
    bn: int = 128,
    bk: int = 32,
    dtype: str = "f16",
    target: str = "gfx942",
    kernel_name: str = "partial_gemm",
) -> ir.Module:
    """Build a partial-tile GEMM kernel using the workspace+counter epilogue.

    This kernel handles k-iterations [k_start_iter, k_end_iter) for tile
    0. It writes accumulator data to a workspace buffer and atomically
    increments a per-tile counter. The WG whose increment completes the
    tile reads all workspace slots, sums them, and writes C via plain
    stores.

    Multiple kernels with non-overlapping ranges covering the full K can
    be launched with shared ws/ctr buffers to reconstruct the full GEMM
    result.

    K % bk == 0 is required.
    """
    assert K % bk == 0, f"k={K} must be divisible by bk={bk}"
    assert 0 <= k_start_iter <= k_end_iter
    assert k_end_iter <= K // bk, (
        f"k_end_iter={k_end_iter} exceeds iters_per_tile={K // bk}; "
        "build_partial_gemm only handles tile 0"
    )

    m_t = bm // _MFMA_M
    n_t = bn // _MFMA_N
    k_t = bk // _TILE_K_ELEMS
    assert k_t >= 1, f"bk={bk} must be >= tile_k_elems={_TILE_K_ELEMS}"
    _check_lds_capacity(target, m_t, n_t, k_t)

    k_step = k_t * _TILE_K_ELEMS
    iters_per_tile = K // k_step

    wpw_m, wpw_n = 2, 2
    nw = wpw_m * wpw_n
    num_threads = nw * _WAVE_SIZE
    m_per_wave = m_t // wpw_m
    n_per_wave = n_t // wpw_n

    stride_a = K * _ELT_BYTES
    stride_b = K * _ELT_BYTES
    stride_c = N * 4  # f32 output

    # Workspace geometry.
    ws_slot_bytes = bm * bn * 4  # f32 per C tile element
    ws_tile_bytes = iters_per_tile * ws_slot_bytes

    b = KernelBuilder(f"{kernel_name}_mod", kernel_name, target=target)
    b.set_grid_dims(1)
    b.set_block_dims(num_threads)
    b.add_ptr_arg(AccessKind.ReadOnly)  # A
    b.add_ptr_arg(AccessKind.ReadOnly)  # B
    b.add_ptr_arg(AccessKind.WriteOnly)  # C
    b.add_ptr_arg(AccessKind.ReadWrite)  # workspace (f32)
    b.add_ptr_arg(AccessKind.ReadWrite)  # counters (i32)
    a_ptr, b_ptr, c_ptr, ws_ptr, ctr_ptr = b.load_args()

    cfg, tc_store_c = _build_cfg(
        b, m_t, n_t, k_t, wpw_m, wpw_n, stride_a, stride_b, stride_c, dtype
    )

    A_TILED, B_TILED, C_TILED, WS_TILED, LDS_A_TILED, LDS_B_TILED = (
        _make_global_layouts(
            M,
            N,
            K,
            m_t,
            n_t,
            k_t,
            m_per_wave,
            n_per_wave,
            wpw_m,
            wpw_n,
            stride_a,
            stride_b,
            stride_c,
            bn,
        )
    )

    # Workspace and C share the same per-wave TiledCopy descriptor.
    tc_ws = tc_store_c

    # Allocate 4-byte LDS scratch for counter broadcast (lane 0 → all lanes).

    c_zero_idx = b.constant_index(0)
    wid = b.wave_id()
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (wpw_m, wpw_n))

    # Tile 0 only: slice A, B, C, WS at wg_m=0, wg_n=0.
    TA = b.slice(Tensor(a_ptr, layout=A_TILED), {_wg_m: c_zero_idx})
    TB = b.slice(Tensor(b_ptr, layout=B_TILED), {_wg_n: c_zero_idx})
    TC = b.slice(
        Tensor(c_ptr, layout=C_TILED),
        {
            _wg_m: c_zero_idx,
            _wg_n: c_zero_idx,
            _wave_m: wave_m_idx,
            _wave_n: wave_n_idx,
        },
    )
    TWS = b.slice(
        Tensor(ws_ptr, layout=WS_TILED),
        {
            _wg_m: c_zero_idx,
            _wg_n: c_zero_idx,
            _wave_m: wave_m_idx,
            _wave_n: wave_n_idx,
        },
    )

    n_accs = m_per_wave * n_per_wave
    acc_inits = [b.init_agprx4(b.constant_i32(0)) for _ in range(n_accs)]

    k_start = b.constant_index(k_start_iter)
    k_end = b.constant_index(k_end_iter)

    accs_final = build_mainloop(
        b,
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        TA=TA,
        TB=TB,
        lds_a_layout=LDS_A_TILED,
        lds_b_layout=LDS_B_TILED,
        k_iter_start=k_start,
        k_iter_end=k_end,
        acc_inits=acc_inits,
        cfg=cfg,
        wave_m_idx=wave_m_idx,
        wave_n_idx=wave_n_idx,
    )

    acc_layout = Layout((m_per_wave, n_per_wave))
    # Tile 0: global k-iter indices equal tile-local indices (tile offset is 0).
    local_start = b.constant_index(k_start_iter)
    local_end = b.constant_index(k_end_iter)

    # Tile 0 counter at byte offset 0.
    counter_byte_offset = b.constant_index(0)

    # Producer's workspace tensor (rooted at this slot).
    tile_id_zero = b.constant_index(0)
    ws_producer = _make_ws_tensor_producer(
        b, ws_ptr, TWS, tile_id_zero, local_start, ws_tile_bytes, ws_slot_bytes
    )

    # All slot byte offsets for the consumer.
    slot_byte_offsets = _make_slot_byte_offsets(
        b, tile_id_zero, iters_per_tile, ws_tile_bytes, ws_slot_bytes
    )

    # Consumer's workspace tensor (base for load_and_sum_workspace).
    ws_consumer = Tensor(ws_ptr, TWS.offset, TWS.layout)

    flush_tile(
        b,
        c_tensor=TC,
        ws_tensor=ws_producer,
        ws_consumer_tensor=ws_consumer,
        counter_ptr=ctr_ptr,
        counter_byte_offset=counter_byte_offset,
        accs=accs_final,
        store_desc=tc_store_c,
        ws_store_desc=tc_ws,
        ws_load_desc=tc_ws,
        acc_layout=acc_layout,
        local_start=local_start,
        local_end=local_end,
        iters_per_tile=iters_per_tile,
        slot_byte_offsets=slot_byte_offsets,
        m_axis=_m,
        n_axis=_n,
    )
    return b.build()


def _emit_persistent_loop_body(
    b,
    *,
    iter_end_i32,
    iters_per_tile: int,
    wg_m_count: int,
    wg_n_count: int,
    a_ptr,
    b_ptr,
    c_ptr,
    ws_ptr,
    ctr_ptr,
    A_TILED,
    B_TILED,
    C_TILED,
    WS_TILED,
    LDS_A_TILED,
    LDS_B_TILED,
    cfg,
    tc_store_c,
    tc_ws,
    wave_m_idx,
    wave_n_idx,
    n_accs: int,
    acc_layout,
    ws_tile_bytes: int,
    ws_slot_bytes: int,
):
    """Return the body function for the persistent StreamK while-loop."""
    idx_type = ir.IndexType.get(b._ctx)

    def _body(args):
        k_cur = args[0]

        k_cur_idx = _arith.index_cast(idx_type, k_cur, loc=b._loc, ip=b._kip)
        iter_end_idx = _arith.index_cast(idx_type, iter_end_i32, loc=b._loc, ip=b._kip)

        tile_id, local_start, tile_end = emit_tile_decompose(
            b, k_cur_idx, iters_per_tile
        )
        local_end_iter = _arith.minui(iter_end_idx, tile_end, loc=b._loc, ip=b._kip)

        tile_wg_m, tile_wg_n = b.delinearize_index(tile_id, (wg_m_count, wg_n_count))

        TA = b.slice(Tensor(a_ptr, layout=A_TILED), {_wg_m: tile_wg_m})
        TB = b.slice(Tensor(b_ptr, layout=B_TILED), {_wg_n: tile_wg_n})
        TC = b.slice(
            Tensor(c_ptr, layout=C_TILED),
            {
                _wg_m: tile_wg_m,
                _wg_n: tile_wg_n,
                _wave_m: wave_m_idx,
                _wave_n: wave_n_idx,
            },
        )
        TWS = b.slice(
            Tensor(ws_ptr, layout=WS_TILED),
            {
                _wg_m: tile_wg_m,
                _wg_n: tile_wg_n,
                _wave_m: wave_m_idx,
                _wave_n: wave_n_idx,
            },
        )

        _emit_slice_and_mainloop(
            b,
            a_ptr=a_ptr,
            b_ptr=b_ptr,
            TA=TA,
            TB=TB,
            LDS_A_TILED=LDS_A_TILED,
            LDS_B_TILED=LDS_B_TILED,
            k_cur_idx=k_cur_idx,
            local_end_iter=local_end_iter,
            tile_id=tile_id,
            local_start=local_start,
            cfg=cfg,
            n_accs=n_accs,
            acc_layout=acc_layout,
            iters_per_tile=iters_per_tile,
            wave_m_idx=wave_m_idx,
            wave_n_idx=wave_n_idx,
            TC=TC,
            TWS=TWS,
            ws_ptr=ws_ptr,
            ctr_ptr=ctr_ptr,
            tc_store_c=tc_store_c,
            tc_ws=tc_ws,
            ws_tile_bytes=ws_tile_bytes,
            ws_slot_bytes=ws_slot_bytes,
        )

        # Advance k_cur to the next tile boundary (as i32).
        local_end_iter_i32 = b.index_cast_i32(local_end_iter)
        return [local_end_iter_i32]

    return _body


def _emit_slice_and_mainloop(
    b,
    *,
    a_ptr,
    b_ptr,
    TA,
    TB,
    LDS_A_TILED,
    LDS_B_TILED,
    k_cur_idx,
    local_end_iter,
    tile_id,
    local_start,
    cfg,
    n_accs: int,
    acc_layout,
    iters_per_tile: int,
    wave_m_idx,
    wave_n_idx,
    TC,
    TWS,
    ws_ptr,
    ctr_ptr,
    tc_store_c,
    tc_ws,
    ws_tile_bytes: int,
    ws_slot_bytes: int,
) -> None:
    """Emit mainloop and flush_tile for one tile slice in the persistent kernel."""
    # Normalize the inner loop to [0, n_iters) so the pipeline pass sees a
    # constant lower bound. n_iters is the number of k-iterations this WG
    # runs within the current tile. k_offset=local_start positions each read
    # at the correct k-tile inside the tile.
    c0 = b.constant_index(0)
    n_iters = _arith.subi(local_end_iter, k_cur_idx, loc=b._loc, ip=b._kip)
    acc_inits = [b.init_agprx4(b.constant_i32(0)) for _ in range(n_accs)]

    accs_final = build_mainloop(
        b,
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        TA=TA,
        TB=TB,
        lds_a_layout=LDS_A_TILED,
        lds_b_layout=LDS_B_TILED,
        k_iter_start=c0,
        k_iter_end=n_iters,
        acc_inits=acc_inits,
        cfg=cfg,
        wave_m_idx=wave_m_idx,
        wave_n_idx=wave_n_idx,
        k_offset=local_start,
    )

    c_ipt = b.constant_index(iters_per_tile)
    local_end = _arith.subi(
        local_end_iter,
        _arith.muli(tile_id, c_ipt, loc=b._loc, ip=b._kip),
        loc=b._loc,
        ip=b._kip,
    )

    # Counter byte offset for this tile (4 bytes per i32 entry).
    d0 = ir.AffineExpr.get_dim(0)
    counter_byte_offset = b.affine_apply(d0 * 4, [tile_id])

    # Producer workspace tensor for this WG's slot.
    ws_producer = _make_ws_tensor_producer(
        b, ws_ptr, TWS, tile_id, local_start, ws_tile_bytes, ws_slot_bytes
    )

    # All slot byte offsets for the consumer.
    slot_byte_offsets = _make_slot_byte_offsets(
        b, tile_id, iters_per_tile, ws_tile_bytes, ws_slot_bytes
    )

    # Consumer's workspace tensor (base for load_and_sum_workspace).
    ws_consumer = Tensor(ws_ptr, TWS.offset, TWS.layout)

    flush_tile(
        b,
        c_tensor=TC,
        ws_tensor=ws_producer,
        ws_consumer_tensor=ws_consumer,
        counter_ptr=ctr_ptr,
        counter_byte_offset=counter_byte_offset,
        accs=accs_final,
        store_desc=tc_store_c,
        ws_store_desc=tc_ws,
        ws_load_desc=tc_ws,
        acc_layout=acc_layout,
        local_start=local_start,
        local_end=local_end,
        iters_per_tile=iters_per_tile,
        slot_byte_offsets=slot_byte_offsets,
        m_axis=_m,
        n_axis=_n,
    )


def build_streamk_gemm(
    M: int,
    N: int,
    K: int,
    bm: int = 128,
    bn: int = 128,
    bk: int = 32,
    grid: int = None,
    dtype: str = "f16",
    target: str = "gfx942",
    kernel_name: str = "streamk_gemm",
) -> ir.Module:
    """Build a persistent StreamK GEMM kernel.

    Each workgroup handles a dynamic range of k-iterations determined at
    runtime by the StreamK scheduler. Partial-tile results are written to a
    workspace buffer and gated by a per-tile i32 counter. The WG that
    completes a tile reads all workspace slots, sums them, and writes C.
    Full-tile results are stored directly to C, bypassing the workspace.

    K % bk == 0 is required.

    grid: number of persistent workgroups.  Defaults to total_tiles (one WG
    per output tile), which reproduces the data-parallel case and is a safe
    correctness baseline.  Pass a smaller value (e.g. the number of CUs on the
    device) to enable the StreamK load-balancing over a persistent grid.
    """
    assert M % bm == 0, f"m={M} must be divisible by bm={bm}"
    assert N % bn == 0, f"n={N} must be divisible by bn={bn}"
    assert K % bk == 0, f"k={K} must be divisible by bk={bk}"

    m_t = bm // _MFMA_M
    n_t = bn // _MFMA_N
    k_t = bk // _TILE_K_ELEMS
    assert k_t >= 1, f"bk={bk} must be >= tile_k_elems={_TILE_K_ELEMS}"
    _check_lds_capacity(target, m_t, n_t, k_t)

    k_step = k_t * _TILE_K_ELEMS
    iters_per_tile = K // k_step

    wpw_m, wpw_n = 2, 2
    nw = wpw_m * wpw_n
    num_threads = nw * _WAVE_SIZE
    m_per_wave = m_t // wpw_m
    n_per_wave = n_t // wpw_n

    wg_m_count = M // bm
    wg_n_count = N // bn
    total_tiles = wg_m_count * wg_n_count
    n_grid = grid if grid is not None else total_tiles
    assert total_tiles % n_grid == 0, (
        f"grid={n_grid} must evenly divide total_tiles={total_tiles} so each "
        "WG owns complete tiles; partial-tile splits require divergent control "
        "flow which is not yet supported"
    )

    stride_a = K * _ELT_BYTES
    stride_b = K * _ELT_BYTES
    stride_c = N * 4  # f32 output

    # Workspace geometry.
    ws_slot_bytes = bm * bn * 4  # f32 per C tile element
    ws_tile_bytes = iters_per_tile * ws_slot_bytes

    sched = compute_schedule(M, N, K, bm, bn, bk, n_grid)

    b = KernelBuilder(f"{kernel_name}_mod", kernel_name, target=target)
    b.set_grid_dims(n_grid)
    b.set_block_dims(num_threads)
    b.add_ptr_arg(AccessKind.ReadOnly)  # A
    b.add_ptr_arg(AccessKind.ReadOnly)  # B
    b.add_ptr_arg(AccessKind.WriteOnly)  # C
    b.add_ptr_arg(AccessKind.ReadWrite)  # workspace (f32)
    b.add_ptr_arg(AccessKind.ReadWrite)  # counters (i32)
    a_ptr, b_ptr, c_ptr, ws_ptr, ctr_ptr = b.load_args()

    cfg, tc_store_c = _build_cfg(
        b, m_t, n_t, k_t, wpw_m, wpw_n, stride_a, stride_b, stride_c, dtype
    )

    # Workspace and C share the same per-wave TiledCopy descriptor.
    tc_ws = tc_store_c

    A_TILED, B_TILED, C_TILED, WS_TILED, LDS_A_TILED, LDS_B_TILED = (
        _make_global_layouts(
            M,
            N,
            K,
            m_t,
            n_t,
            k_t,
            m_per_wave,
            n_per_wave,
            wpw_m,
            wpw_n,
            stride_a,
            stride_b,
            stride_c,
            bn,
        )
    )

    # Allocate 4-byte LDS scratch for per-tile counter broadcast (lane 0 → all).

    wg_id = b.linear_block_id()
    wid = b.wave_id()
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (wpw_m, wpw_n))

    # Per-WG global k-iteration range.
    iter_start, iter_end = emit_wg_iter_range(b, wg_id, sched)

    n_accs = m_per_wave * n_per_wave
    acc_layout = Layout((m_per_wave, n_per_wave))

    # Use i32 for the while-loop variable to avoid index-i32 type mismatches
    # during the LSIR lowering pass.  total_iters must fit in i32 or the cast
    # silently truncates and the loop bound is wrong.
    assert sched["totalIters"] <= 2**31 - 1, (
        f"totalIters={sched['totalIters']} overflows i32; "
        "reduce problem size or tile dimensions"
    )
    k_cur_i32 = b.index_cast_i32(iter_start)
    iter_end_i32 = b.index_cast_i32(iter_end)

    def _cond(args):
        k_cur = args[0]
        return _arith.cmpi(
            _arith.CmpIPredicate.ult, k_cur, iter_end_i32, loc=b._loc, ip=b._kip
        )

    _body = _emit_persistent_loop_body(
        b,
        iter_end_i32=iter_end_i32,
        iters_per_tile=iters_per_tile,
        wg_m_count=wg_m_count,
        wg_n_count=wg_n_count,
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        c_ptr=c_ptr,
        ws_ptr=ws_ptr,
        ctr_ptr=ctr_ptr,
        A_TILED=A_TILED,
        B_TILED=B_TILED,
        C_TILED=C_TILED,
        WS_TILED=WS_TILED,
        LDS_A_TILED=LDS_A_TILED,
        LDS_B_TILED=LDS_B_TILED,
        cfg=cfg,
        tc_store_c=tc_store_c,
        tc_ws=tc_ws,
        wave_m_idx=wave_m_idx,
        wave_n_idx=wave_n_idx,
        n_accs=n_accs,
        acc_layout=acc_layout,
        ws_tile_bytes=ws_tile_bytes,
        ws_slot_bytes=ws_slot_bytes,
    )

    _emit_cf_while_loop(b, [k_cur_i32], _cond, _body)
    return b.build()
