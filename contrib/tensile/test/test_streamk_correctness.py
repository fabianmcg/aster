"""StreamK GEMM correctness tests."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import ml_dtypes
import numpy as np
import pytest

from aster import ir
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, InOutArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig

from tensile.gemm_streamk import build_dp_gemm, build_partial_gemm, build_streamk_gemm

RTOL = 1e-2
ATOL = 1e-2


def _make_inputs(M, N, K, dtype: str):
    """Create random A (M,K) and B (N,K) matrices for the given dtype."""
    np.random.seed(M + N + K)
    if dtype == "f16":
        A = (np.random.randn(M, K) * 0.1).astype(np.float16)
        B = (np.random.randn(N, K) * 0.1).astype(np.float16)
    else:
        bf16 = ml_dtypes.bfloat16
        A = (np.random.randn(M, K) * 0.1).astype(bf16)
        B = (np.random.randn(N, K) * 0.1).astype(bf16)
    return A, B


def _run_gemm(M, N, K, bk, dtype, mcpu, kernel_name="dp_gemm_test"):
    """Build, compile, assemble, and execute a DP GEMM kernel."""
    A, B = _make_inputs(M, N, K, dtype)
    C_out = np.zeros(M * N, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = build_dp_gemm(
            M, N, K, bk=bk, dtype=dtype, target=mcpu, kernel_name=kernel_name
        )
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
        )

    path = assemble_to_hsaco(asm, target=mcpu, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler not compiled with {mcpu} support")

    num_threads = 4 * 64
    wg_m = M // 128
    wg_n = N // 128

    with hsaco_file(path):
        if not system_has_mcpu(mcpu):
            pytest.skip(f"{mcpu} GPU not available")
        execute_hsaco(
            hsaco_path=path,
            kernel_name=kernel_name,
            arguments=[
                InputArray(A.flatten()),
                InputArray(B.flatten()),
                OutputArray(C_out),
            ],
            grid_dim=(wg_m * wg_n, 1, 1),
            block_dim=(num_threads, 1, 1),
        )

    ref = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
    return C_out, ref


def _compile_partial(M, N, K, k_start_iter, k_end_iter, bk, dtype, mcpu, kernel_name):
    """Build and compile a partial GEMM kernel, return hsaco path."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = build_partial_gemm(
            M,
            N,
            K,
            k_start_iter=k_start_iter,
            k_end_iter=k_end_iter,
            bk=bk,
            dtype=dtype,
            target=mcpu,
            kernel_name=kernel_name,
        )
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
        )
    return assemble_to_hsaco(asm, target=mcpu, wavefront_size=64)


class TestMainloopDP:
    """Data-parallel GEMM validates build_mainloop with static K bounds."""

    @pytest.mark.parametrize("mcpu", ["gfx942", "gfx950"])
    @pytest.mark.parametrize("dtype", ["f16", "bf16"])
    def test_mainloop_dp_128x128x128(self, dtype, mcpu):
        if not system_has_mcpu(mcpu):
            pytest.skip(f"{mcpu} GPU not available")

        # bk=32 gives k_iters=4 which meets the 4-stage pipeline minimum for K=128.
        C_out, ref = _run_gemm(
            128,
            128,
            128,
            bk=32,
            dtype=dtype,
            mcpu=mcpu,
            kernel_name=f"dp_gemm_{dtype}",
        )
        np.testing.assert_allclose(C_out, ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("mcpu", ["gfx942", "gfx950"])
    @pytest.mark.parametrize("dtype", ["f16", "bf16"])
    def test_mainloop_dp_128x256x128(self, dtype, mcpu):
        """Non-square M!=N ensures B is shaped (N,K) not (M,K)."""
        if not system_has_mcpu(mcpu):
            pytest.skip(f"{mcpu} GPU not available")

        C_out, ref = _run_gemm(
            128,
            256,
            128,
            bk=32,
            dtype=dtype,
            mcpu=mcpu,
            kernel_name=f"dp_gemm_nonsq_{dtype}",
        )
        np.testing.assert_allclose(C_out, ref, rtol=RTOL, atol=ATOL)


class TestAtomicFlush:
    """Two-WG StreamK accumulation via workspace + counter epilogue."""

    @pytest.mark.parametrize("mcpu", ["gfx942", "gfx950"])
    @pytest.mark.parametrize("dtype", ["f16", "bf16"])
    @pytest.mark.skip(
        reason="Partial-tile epilogue requires divergent control flow which is "
        "not yet supported. The workspace+counter infrastructure is preserved "
        "for future use; enforce non-divergent grids via build_streamk_gemm assertion."
    )
    def test_atomic_flush_accumulates(self, dtype, mcpu):
        """Two partial kernels each computing half of K accumulate C via workspace.

        This validates flush_tile's partial path: both kernels share the
        same ws and ctr buffers. The first kernel stores its partial
        results and increments the counter. The second kernel becomes
        the consumer (its increment completes the tile) and reads all
        workspace slots, sums them, and writes C via plain stores.
        """
        if not system_has_mcpu(mcpu):
            pytest.skip(f"{mcpu} GPU not available")

        M, N, K = 128, 128, 256
        bk = 32
        iters_per_tile = K // bk  # 8
        k_half = iters_per_tile // 2  # 4, satisfies the 4-stage pipeline minimum

        A, B = _make_inputs(M, N, K, dtype)
        # ws and ctr are shared across both launches; do NOT zero between them.
        ws = np.zeros(iters_per_tile * M * N, dtype=np.float32)
        ctr = np.zeros(1, dtype=np.int32)  # 1 tile (tile 0)
        C_out = np.zeros(M * N, dtype=np.float32)

        kname0 = f"partial_gemm_{dtype}_0"
        kname1 = f"partial_gemm_{dtype}_1"

        path0 = _compile_partial(M, N, K, 0, k_half, bk, dtype, mcpu, kname0)
        path1 = _compile_partial(
            M, N, K, k_half, iters_per_tile, bk, dtype, mcpu, kname1
        )

        if path0 is None or path1 is None:
            pytest.skip(f"LLVM assembler not compiled with {mcpu} support")

        num_threads = 4 * 64  # nw=4 waves * 64 lanes

        with hsaco_file(path0), hsaco_file(path1):
            if not system_has_mcpu(mcpu):
                pytest.skip(f"{mcpu} GPU not available")
            # First partial kernel: stores k-iters [0, k_half) to workspace and
            # increments ctr. ctr goes from 0 → k_half, so this WG is not consumer.
            execute_hsaco(
                hsaco_path=path0,
                kernel_name=kname0,
                arguments=[
                    InputArray(A.flatten()),
                    InputArray(B.flatten()),
                    InOutArray(C_out),
                    InOutArray(ws),
                    InOutArray(ctr),
                ],
                grid_dim=(1, 1, 1),
                block_dim=(num_threads, 1, 1),
            )
            # Second partial kernel: stores k-iters [k_half, iters_per_tile) and
            # increments ctr. ctr goes from k_half → iters_per_tile, so this WG
            # is the consumer and writes C via load_and_sum_workspace.
            execute_hsaco(
                hsaco_path=path1,
                kernel_name=kname1,
                arguments=[
                    InputArray(A.flatten()),
                    InputArray(B.flatten()),
                    InOutArray(C_out),
                    InOutArray(ws),
                    InOutArray(ctr),
                ],
                grid_dim=(1, 1, 1),
                block_dim=(num_threads, 1, 1),
            )

        ref = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_out, ref, rtol=RTOL, atol=ATOL)


class TestStreamKEndToEnd:
    """Persistent StreamK GEMM end-to-end correctness."""

    @pytest.mark.parametrize("mcpu", ["gfx942", "gfx950"])
    @pytest.mark.parametrize("dtype", ["f16", "bf16"])
    @pytest.mark.parametrize("grid", [1, 2, 4, 8, 16])
    def test_streamk_end_to_end(self, dtype, grid, mcpu):
        """StreamK with varying grid sizes produces the correct GEMM result.

        Grid is snapped to a divisor of total_iters to avoid partial-
        tile splits, which require divergent control flow not yet
        supported.
        """
        if not system_has_mcpu(mcpu):
            pytest.skip(f"{mcpu} GPU not available")

        M, N, K = 256, 256, 128
        bk = 32
        total_tiles = (M // 128) * (N // 128)  # 4
        iters_per_tile = K // bk  # 4

        # Snap grid to a divisor of total_tiles so each WG owns whole tiles.
        grid = min(grid, total_tiles)
        while total_tiles % grid != 0:
            grid -= 1
        kernel_name = f"streamk_{dtype}_g{grid}"

        A, B = _make_inputs(M, N, K, dtype)
        C_out = np.zeros(M * N, dtype=np.float32)
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
        np.testing.assert_allclose(C_out, ref, rtol=RTOL, atol=ATOL)
