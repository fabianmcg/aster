# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""StreamK GEMM shape and grid sweep: correctness across many configurations.

Parametrizes over (M, N, K, bk, grid, dtype) to systematically validate the
StreamK scheduler and mainloop across non-square, large, and skewed shapes.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import ml_dtypes
import numpy as np
import pytest

from aster import ir
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, InOutArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig

from tensile.gemm_streamk import build_streamk_gemm

RTOL = 1e-2
ATOL = 1e-2


def _make_inputs(M, N, K, dtype: str):
    """Create reproducible random A (M,K) and B (N,K) for the given dtype."""
    np.random.seed(M * 31 + N * 17 + K * 7)
    if dtype == "f16":
        A = (np.random.randn(M, K) * 0.1).astype(np.float16)
        B = (np.random.randn(N, K) * 0.1).astype(np.float16)
    else:
        bf16 = ml_dtypes.bfloat16
        A = (np.random.randn(M, K) * 0.1).astype(bf16)
        B = (np.random.randn(N, K) * 0.1).astype(bf16)
    return A, B


def _run_streamk(M, N, K, bk, grid, dtype, mcpu, kernel_name):
    """Build, compile, execute StreamK GEMM; return (C_out, ref)."""
    A, B = _make_inputs(M, N, K, dtype)
    C_out = np.zeros(M * N, dtype=np.float32)

    total_tiles = (M // 128) * (N // 128)
    iters_per_tile = K // bk

    # Snap grid to a divisor of total_tiles so each WG owns whole tiles,
    # avoiding partial-tile splits (which require divergent control flow).
    grid = min(grid, total_tiles)
    while total_tiles % grid != 0:
        grid -= 1

    ws = np.zeros(total_tiles * iters_per_tile * 128 * 128, dtype=np.float32)
    ctr = np.zeros(total_tiles, dtype=np.int32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = build_streamk_gemm(
            M,
            N,
            K,
            bk=bk,
            grid=grid,
            dtype=dtype,
            target=mcpu,
            kernel_name=kernel_name,
        )
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
        )

    path = assemble_to_hsaco(asm, target=mcpu, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler not compiled with {mcpu} support")

    num_threads = 4 * 64

    with hsaco_file(path):
        if not system_has_mcpu(mcpu):
            pytest.skip(f"{mcpu} GPU not available")
        execute_hsaco(
            hsaco_path=path,
            kernel_name=kernel_name,
            arguments=[
                InputArray(A.flatten()),
                InputArray(B.flatten()),
                InOutArray(C_out),
                InOutArray(ws),
                InOutArray(ctr),
            ],
            grid_dim=(grid, 1, 1),
            block_dim=(num_threads, 1, 1),
        )

    ref = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
    return C_out, ref


# ---------------------------------------------------------------------------
# Shape suite
# ---------------------------------------------------------------------------

# (M, N, K, bk) tuples to sweep.
_SHAPES = [
    # Square shapes at increasing sizes.
    (128, 128, 128, 32),
    (256, 256, 128, 32),
    (256, 256, 256, 32),
    (512, 512, 128, 32),
    (512, 512, 256, 32),
    (512, 512, 512, 32),
    # Non-square: wide N.
    (128, 256, 128, 32),
    (128, 512, 128, 32),
    (256, 512, 128, 32),
    (256, 512, 256, 32),
    # Non-square: tall M.
    (256, 128, 128, 32),
    (512, 128, 128, 32),
    (512, 256, 128, 32),
    (512, 256, 256, 32),
    # Large K relative to M/N.
    (128, 128, 256, 32),
    (128, 128, 512, 32),
    (256, 256, 512, 32),
    # Larger bk (bk=64).
    (128, 128, 128, 64),
    (256, 256, 256, 64),
    (512, 512, 256, 64),
    (512, 512, 512, 64),
    # Larger tile counts — more output tiles stress partial-tile paths.
    (512, 512, 128, 32),
    (1024, 512, 128, 32),
    (512, 1024, 128, 32),
    (1024, 1024, 128, 32),
]


def _grids_for(M, N, bm=128, bn=128):
    """Return a representative set of grid sizes for the given (M, N)."""
    total_tiles = (M // bm) * (N // bn)
    grids = set()
    # Always include 1, total_tiles, and total_tiles*2 (over-subscription).
    grids.update([1, total_tiles, total_tiles * 2])
    # Add fractions: half, third, two-thirds.
    for num, den in [(1, 2), (1, 3), (2, 3)]:
        g = max(1, total_tiles * num // den)
        grids.add(g)
    # A few small absolutes to exercise the remainder-heavy scheduler path.
    for g in [2, 3, 5, 7]:
        if g < total_tiles * 2:
            grids.add(g)
    return sorted(grids)


# Build the full parametrize list at module import time.
_PARAMS = []
for _shape in _SHAPES:
    _M, _N, _K, _bk = _shape
    for _grid in _grids_for(_M, _N):
        for _dtype in ["f16", "bf16"]:
            _PARAMS.append((_M, _N, _K, _bk, _grid, _dtype))


class TestStreamKShapes:
    """StreamK correctness across many GEMM and tile shapes."""

    @pytest.mark.parametrize("mcpu", ["gfx942", "gfx950"])
    @pytest.mark.parametrize("M,N,K,bk,grid,dtype", _PARAMS)
    def test_streamk_shape(self, M, N, K, bk, grid, dtype, mcpu):
        if not system_has_mcpu(mcpu):
            pytest.skip(f"{mcpu} GPU not available")

        kname = f"sk_{dtype}_m{M}_n{N}_k{K}_bk{bk}_g{grid}"
        C_out, ref = _run_streamk(M, N, K, bk, grid, dtype, mcpu, kname)
        np.testing.assert_allclose(C_out, ref, rtol=RTOL, atol=ATOL)
