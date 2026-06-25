# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""StreamK GEMM benchmark: TFLOPS for varying grid sizes vs. data-parallel.

Usage::

    python bench_streamk.py                     # sweep default M/N/K sizes
    python bench_streamk.py --M 4096 --N 4096 --K 4096
    python bench_streamk.py --compile-only      # compile only, no GPU needed
    python bench_streamk.py --target gfx942
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from aster import ir
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, InOutArray
from aster.execution.flush_llc import FlushLLC
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig

from aster.core.device import try_query_device

from tensile.gemm_streamk import build_dp_gemm, build_streamk_gemm

_DEFAULT_SIZES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    # 4096^3 disabled: kernel size exceeds the simm16 branch-offset limit.
    # (4096, 4096, 4096),
]
_WAVEFRONT_SIZE = 64
_NUM_THREADS = 4 * _WAVEFRONT_SIZE  # 4 waves per WG

# CU counts per target, used when HIP cannot be queried (e.g. --compile-only on
# a host without a GPU). Values mirror the per-SKU CU counts in
# aster/execution sizing tables; target.py itself does not record CU counts.
_NUM_CU_FALLBACK = {
    "gfx940": 228,  # MI300A
    "gfx942": 304,  # MI300X
    "gfx950": 256,  # MI350X
    "gfx1201": 64,  # RDNA4
}


def _num_cu(target: str) -> int:
    """Return the device CU count, querying HIP when available.

    Falls back to a per-target constant so --compile-only works without
    a GPU.
    """
    props = try_query_device(0)
    if props is not None and props.gcn_arch_name == target:
        return props.multiprocessor_count
    return _NUM_CU_FALLBACK.get(target, 256)


def _tflops(M: int, N: int, K: int, elapsed_ns: int) -> float:
    """Compute TFLOPS from problem size and elapsed time in nanoseconds."""
    flops = 2.0 * M * N * K
    return flops / elapsed_ns * 1e-3


def _compile_kernel(module, target: str, wavefront_size: int = 64) -> str | None:
    """Compile an MLIR module to an HSACO file path."""
    asm = compile_mlir_module_to_asm(
        module,
        pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
    )
    return assemble_to_hsaco(asm, target=target, wavefront_size=wavefront_size)


def _make_inputs(M: int, N: int, K: int, dtype: str):
    """Return random A (M, K) and B (N, K) matrices."""
    np.random.seed(42)
    if dtype == "f16":
        A = (np.random.randn(M, K) * 0.1).astype(np.float16)
        B = (np.random.randn(N, K) * 0.1).astype(np.float16)
    else:
        import ml_dtypes

        A = (np.random.randn(M, K) * 0.1).astype(ml_dtypes.bfloat16)
        B = (np.random.randn(N, K) * 0.1).astype(ml_dtypes.bfloat16)
    return A, B


def _bench_config(
    M: int,
    N: int,
    K: int,
    bk: int,
    dtype: str,
    target: str,
    grid: int | None,
    num_warmup: int,
    num_iters: int,
    compile_only: bool,
) -> dict:
    """Build, compile, and (optionally) time one kernel configuration.

    Returns a dict with keys: label, compiled (bool), tflops (float | None).
    """
    wg_m = M // 128
    wg_n = N // 128
    total_tiles = wg_m * wg_n

    if grid is not None:
        # Snap grid to the largest divisor of total_tiles that is <= the
        # requested grid. This guarantees every WG owns a whole number of
        # complete tiles, avoiding partial-tile splits (divergent control flow).
        grid = min(grid, total_tiles)
        while total_tiles % grid != 0:
            grid -= 1
        if grid < 1:
            grid = 1

    if grid is None:
        label = f"dp  M={M} N={N} K={K} bk={bk} dtype={dtype}"
        kernel_name = f"dp_{dtype}_{M}x{N}x{K}"
    else:
        label = (
            f"sk  M={M} N={N} K={K} bk={bk} dtype={dtype} grid={grid:3d}/{total_tiles}"
        )
        kernel_name = f"sk_{dtype}_{M}x{N}x{K}_g{grid}"

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        if grid is None:
            module = build_dp_gemm(
                M,
                N,
                K,
                bk=bk,
                dtype=dtype,
                target=target,
                kernel_name=kernel_name,
            )
        else:
            module = build_streamk_gemm(
                M,
                N,
                K,
                bk=bk,
                grid=grid,
                dtype=dtype,
                target=target,
                kernel_name=kernel_name,
            )
        path = _compile_kernel(module, target)

    if path is None:
        print(f"  {label}: assembler not available for {target}")
        return {"label": label, "compiled": False, "tflops": None}

    if compile_only or not system_has_mcpu(target):
        print(f"  {label}: compiled ok (no GPU, skipping execution)")
        return {"label": label, "compiled": True, "tflops": None}

    A, B = _make_inputs(M, N, K, dtype)
    C_out = np.zeros(M * N, dtype=np.float32)
    grid_dim = (grid if grid is not None else total_tiles, 1, 1)

    iters_per_tile = K // bk
    # Pre-allocate workspace and counter buffers for StreamK kernels.
    # For the DP baseline (grid is None) these are unused but kept as
    # zero-length placeholders so the call sites are uniform.
    if grid is not None:
        ws_buf = np.zeros(total_tiles * iters_per_tile * 128 * 128, dtype=np.float32)
        ctr_buf = np.zeros(total_tiles, dtype=np.int32)
    else:
        ws_buf = None
        ctr_buf = None

    # Flush the LLC before each timed launch so a hot cache from the previous
    # iteration cannot inflate the result. execute_hsaco owns the object's
    # lifecycle (initialize/flush/cleanup); we only construct it.
    flush = FlushLLC(
        mcpu=target,
        wavefront_size=_WAVEFRONT_SIZE,
        num_workgroups=_num_cu(target),
    )

    def _args():
        base = [
            InputArray(A.flatten()),
            InputArray(B.flatten()),
            InOutArray(C_out),
        ]
        if grid is not None:
            base += [InOutArray(ws_buf), InOutArray(ctr_buf)]
        return base

    with hsaco_file(path):
        # Warmup (no flush — just prime code/data paths).
        for _ in range(num_warmup):
            if grid is not None:
                ws_buf[:] = 0
                ctr_buf[:] = 0
            execute_hsaco(
                hsaco_path=path,
                kernel_name=kernel_name,
                arguments=_args(),
                grid_dim=grid_dim,
                block_dim=(_NUM_THREADS, 1, 1),
            )
        # Timed iterations: re-zero ws/ctr before each launch so the counter
        # never saturates across iterations, then time each launch individually.
        # For DP kernels, delegate batching to execute_hsaco as before.
        if grid is None:
            times_ns = list(
                execute_hsaco(
                    hsaco_path=path,
                    kernel_name=kernel_name,
                    arguments=_args(),
                    grid_dim=grid_dim,
                    block_dim=(_NUM_THREADS, 1, 1),
                    num_iterations=num_iters,
                    flush_llc=flush,
                )
            )
        else:
            times_ns = []
            for _ in range(num_iters):
                ws_buf[:] = 0
                ctr_buf[:] = 0
                elapsed = execute_hsaco(
                    hsaco_path=path,
                    kernel_name=kernel_name,
                    arguments=_args(),
                    grid_dim=grid_dim,
                    block_dim=(_NUM_THREADS, 1, 1),
                    num_iterations=1,
                    flush_llc=flush,
                )
                times_ns.append(elapsed[0])

    median_ns = int(np.median(times_ns))
    peak_ns = int(np.min(times_ns))
    tf_median = _tflops(M, N, K, median_ns)
    tf_peak = _tflops(M, N, K, peak_ns)
    print(
        f"  {label}: {tf_median:.2f} TFLOPS median, "
        f"{tf_peak:.2f} TFLOPS peak  ({median_ns / 1e6:.3f} ms median)"
    )
    return {
        "label": label,
        "compiled": True,
        "tflops": tf_median,
        "tflops_peak": tf_peak,
    }


def run_sweep(
    sizes: list[tuple[int, int, int]],
    bk: int,
    dtype: str,
    target: str,
    grids: list[int | None],
    num_warmup: int,
    num_iters: int,
    compile_only: bool,
) -> None:
    """Run the full benchmark sweep."""
    for M, N, K in sizes:
        print(f"\n{'=' * 60}")
        print(f"M={M} N={N} K={K} dtype={dtype}")
        print(f"{'=' * 60}")
        # A grid larger than the problem's total k-iterations is clamped inside
        # _bench_config; clamp here too so two raw grids collapsing onto the same
        # value are not benchmarked twice with identical output.
        total_iters = (M // 128) * (N // 128) * (K // bk)
        seen: set[int] = set()
        for grid in grids:
            if grid is not None:
                grid = min(grid, total_iters)
                if grid in seen:
                    continue
                seen.add(grid)
            try:
                _bench_config(
                    M,
                    N,
                    K,
                    bk,
                    dtype,
                    target,
                    grid,
                    num_warmup,
                    num_iters,
                    compile_only,
                )
            except Exception as exc:
                label = f"grid={grid}" if grid is not None else "dp"
                print(f"  {label}: ERROR — {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="StreamK GEMM benchmark")
    parser.add_argument("--M", type=int, default=None, help="Matrix M dimension")
    parser.add_argument("--N", type=int, default=None, help="Matrix N dimension")
    parser.add_argument("--K", type=int, default=None, help="Matrix K dimension")
    parser.add_argument("--bk", type=int, default=32, help="K-tile size (default 32)")
    parser.add_argument("--dtype", default="f16", choices=["f16", "bf16"])
    parser.add_argument(
        "--target", default="gfx950", help="GPU target (default gfx950)"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile kernels but skip GPU execution",
    )
    args = parser.parse_args()

    if args.M is not None and args.N is not None and args.K is not None:
        sizes = [(args.M, args.N, args.K)]
    else:
        sizes = _DEFAULT_SIZES

    # Run data-parallel (None) then StreamK with grids tied to the hardware CU
    # count: a persistent grid sized to the device is what makes StreamK
    # load-balancing meaningful. Fractions of total_tiles are unrelated to
    # occupancy and were the cause of the flat ~27 TFLOPS plateau.
    num_cu = _num_cu(args.target)
    print(f"target={args.target} num_cu={num_cu}")
    grids = [
        None,  # data-parallel baseline
        max(1, num_cu // 2),
        num_cu,
        num_cu * 2,
    ]

    run_sweep(
        sizes=sizes,
        bk=args.bk,
        dtype=args.dtype,
        target=args.target,
        grids=grids,
        num_warmup=args.warmup,
        num_iters=args.iters,
        compile_only=args.compile_only,
    )


if __name__ == "__main__":
    main()
