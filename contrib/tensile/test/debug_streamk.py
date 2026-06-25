"""Structured StreamK debugging: test progressively more complex cases.

Level 0: single 1x1 tile, grid=1, K=128 (1 tile, 4 iters, no partial).
Level 1: single 1x1 tile, grid=1, larger K=256 (1 tile, 8 iters, no partial).
Level 2: 2x2 tiles (M=N=256), grid=4 (4 WGs, 1 tile each - data-parallel equiv).
Level 3: 2x2 tiles (M=N=256), grid=3 (partial tiles).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import ml_dtypes
import numpy as np

from aster import ir
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, InOutArray
from aster.execution.utils import system_has_mcpu
from aster.execution.helpers import hsaco_file
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig

from tensile.gemm_streamk import build_streamk_gemm, build_dp_gemm

MCPU = "gfx950"


def make_inputs(M, N, K, dtype="f16", seed=42):
    """Create small, structured inputs for easier debugging."""
    np.random.seed(seed)
    if dtype == "f16":
        A = (np.random.randn(M, K) * 0.1).astype(np.float16)
        B = (np.random.randn(N, K) * 0.1).astype(np.float16)
    else:
        bf16 = ml_dtypes.bfloat16
        A = (np.random.randn(M, K) * 0.1).astype(bf16)
        B = (np.random.randn(N, K) * 0.1).astype(bf16)
    return A, B


def run_streamk(M, N, K, bk, grid, dtype, label):
    """Compile and run a StreamK kernel, compare to numpy reference."""
    print(f"\n=== {label}: M={M} N={N} K={K} bk={bk} grid={grid} dtype={dtype} ===")

    A, B = make_inputs(M, N, K, dtype)
    C_out = np.zeros(M * N, dtype=np.float32)
    ref = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()

    total_tiles = (M // 128) * (N // 128)
    iters_per_tile = K // bk
    ws = np.zeros(total_tiles * iters_per_tile * 128 * 128, dtype=np.float32)
    ctr = np.zeros(total_tiles, dtype=np.int32)

    kernel_name = f"sk_{label}"
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
            target=MCPU,
            kernel_name=kernel_name,
        )
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
        )

    path = assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
    if path is None:
        print("  SKIP: assembler not available")
        return

    if not system_has_mcpu(MCPU):
        print("  SKIP: GPU not available")
        return

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

    wrong = np.sum(np.abs(C_out - ref) > 1e-2 * np.abs(ref) + 1e-2)
    total = C_out.size
    max_re = np.max(np.abs(C_out - ref) / (np.abs(ref) + 1e-6))
    match = np.allclose(C_out, ref, rtol=1e-2, atol=1e-2)

    print(f"  result: {'PASS' if match else 'FAIL'}")
    print(f"  wrong elements: {wrong}/{total} ({100 * wrong / total:.1f}%)")
    print(f"  max rel err: {max_re:.4f}")
    if not match:
        # Print first few mismatches.
        for i in range(min(5, total)):
            if abs(C_out[i] - ref[i]) > 1e-2 * abs(ref[i]) + 1e-2:
                print(f"  mismatch[{i}]: got={C_out[i]:.6f}  ref={ref[i]:.6f}")


def run_dp(M, N, K, bk, dtype, label):
    """Run the data-parallel GEMM as reference point."""
    print(f"\n=== {label}: M={M} N={N} K={K} bk={bk} dtype={dtype} (DP) ===")

    A, B = make_inputs(M, N, K, dtype)
    C_out = np.zeros(M * N, dtype=np.float32)
    ref = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()

    kernel_name = f"dp_{label}"
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = build_dp_gemm(
            M,
            N,
            K,
            bk=bk,
            dtype=dtype,
            target=MCPU,
            kernel_name=kernel_name,
        )
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
        )

    path = assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
    if path is None:
        print("  SKIP: assembler not available")
        return

    if not system_has_mcpu(MCPU):
        print("  SKIP: GPU not available")
        return

    num_threads = 4 * 64
    wg_m = M // 128
    wg_n = N // 128
    with hsaco_file(path):
        execute_hsaco(
            hsaco_path=path,
            kernel_name=kernel_name,
            arguments=[
                InputArray(A.flatten()),
                InputArray(B.flatten()),
                InOutArray(C_out),
            ],
            grid_dim=(wg_m * wg_n, 1, 1),
            block_dim=(num_threads, 1, 1),
        )

    wrong = np.sum(np.abs(C_out - ref) > 1e-2 * np.abs(ref) + 1e-2)
    total = C_out.size
    match = np.allclose(C_out, ref, rtol=1e-2, atol=1e-2)
    print(f"  result: {'PASS' if match else 'FAIL'}")
    print(f"  wrong elements: {wrong}/{total} ({100 * wrong / total:.1f}%)")


def main():
    print("StreamK structured debugging")
    print("=" * 60)

    # Level 0: Single tile, grid=1. iters_per_tile=4, WG handles all 4 iters.
    # The while-loop body executes once, local_start=0, local_end=4.
    # flush_tile should take the full-tile store path.
    run_streamk(128, 128, 128, bk=32, grid=1, dtype="f16", label="L0_1tile_1wg")

    # Level 0b: Same as L0 but compare with DP.
    run_dp(128, 128, 128, bk=32, dtype="f16", label="L0_dp_ref")

    # Level 1: Single tile, grid=1, bigger K. iters_per_tile=8, WG handles all 8 iters.
    run_streamk(128, 128, 256, bk=32, grid=1, dtype="f16", label="L1_1tile_1wg_K256")

    # Level 2: 4 tiles (M=N=256), grid=4. Each WG handles exactly 1 tile (4 iters).
    # No partial tiles. Should match DP exactly.
    run_streamk(256, 256, 128, bk=32, grid=4, dtype="f16", label="L2_4tiles_4wg")

    # Level 2b: DP reference for 4-tile case.
    run_dp(256, 256, 128, bk=32, dtype="f16", label="L2_dp_ref")

    # Level 3: 4 tiles (M=N=256), grid=3. Forces partial tiles.
    run_streamk(256, 256, 128, bk=32, grid=3, dtype="f16", label="L3_4tiles_3wg")

    print("\n" + "=" * 60)
    print("Done")


if __name__ == "__main__":
    main()
