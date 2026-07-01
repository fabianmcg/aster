"""E2E test: C[i,j] = C[i,j] / sum(D[i,:]) for all rows i and columns j.

Kernel: row_div in sandbox/row_div.mlir.

Launch: block_dim=(64,1,1), grid_dim=(M,1,1) — one wavefront per row.

The warp reduction uses ds_bpermute_b32 for an in-wavefront butterfly sum
(6 rounds), then broadcasts the row total back to all lanes.

C is bf16 in memory; D is f32. The kernel loads bf16→f32, divides in f32,
and stores the result as f32→bf16 (truncate toward zero, no rounding).

Parametrized over (M, N, K) where D has shape (M, N//K):
  - n_d = N // K (columns of D)
"""

import os

import ml_dtypes
import numpy as np
import pytest

from aster.execution.core import InOutArray
from aster.execution.helpers import compile_and_run
from aster.pass_pipelines import PipelineConfig, make_default_pass_pipeline

PASS_PIPELINE = make_default_pass_pipeline(PipelineConfig(scf_pipeline=False))

MLIR_FILE = os.path.join(os.path.dirname(__file__), "row_div.mlir")
KERNEL_NAME = "row_div"
WAVE_SIZE = 64
BF16 = ml_dtypes.bfloat16

# (M, N, K) — D has shape (M, N//K).
SHAPES = [
    (4, 64, 2),
    (8, 128, 4),
    (1, 64, 1),
    (16, 256, 8),
    (3, 192, 3),
    (5, 64, 64),  # n_d = 1: single-element D row
    (2, 128, 2),
]


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_row_div(M, N, K):
    """C[i,j] / sum(D[i,:]) matches bf16 reference for all (i,j)."""
    n_d = N // K

    rng = np.random.default_rng(seed=42)
    C_f32 = rng.standard_normal((M, N)).astype(np.float32)
    # D rows must have positive sums so division is well-defined.
    D = rng.uniform(0.1, 2.0, (M, n_d)).astype(np.float32)

    # Expected: (C_bf16[i,j] / sum(D[i,:])) stored as bf16.
    # Round-trip C through bf16 first to match what the kernel reads.
    # The kernel truncates (no rounding): top 16 bits of the f32 result.
    C_bf16 = C_f32.astype(BF16)
    row_sums = D.sum(axis=1, keepdims=True)  # shape (M, 1)
    result_f32 = (C_bf16.astype(np.float32) / row_sums).astype(np.float32)
    # Truncate f32 → bf16 by discarding the low 16 mantissa bits (no rounding).
    result_u32 = result_f32.view(np.uint32) >> 16
    expected = result_u32.astype(np.uint16).view(BF16)

    # Flatten to 1-D for kernel; kernel uses M, N, n_d scalars.
    C_flat = np.ascontiguousarray(C_bf16).ravel()
    D_flat = np.ascontiguousarray(D).ravel()

    def verify(inputs, outputs):
        # C_flat is InOutArray, so it lands in outputs[0] after execution.
        C_out = outputs[0].view(BF16).reshape(M, N)
        np.testing.assert_array_equal(
            C_out,
            expected,
            err_msg=f"M={M} N={N} K={K}: mismatch in C(bf16)/reduction(D)",
        )

    times_ns = compile_and_run(
        MLIR_FILE,
        KERNEL_NAME,
        # Kernel signature: row_div(c_ptr, d_ptr, n, n_d).
        # Pass all args through input_data in order; C_flat uses InOutArray
        # so it is uploaded before and downloaded after the kernel.
        input_data=[InOutArray(C_flat), D_flat, np.int32(N), np.int32(n_d)],
        output_data=[],
        pass_pipeline=PASS_PIPELINE,
        block_dim=(WAVE_SIZE, 1, 1),
        grid_dim=(M, 1, 1),
        verify_fn=verify,
        library_paths=[],
        num_iterations=10,
    )

    if times_ns:
        elapsed_s = min(times_ns) * 1e-9

        # FLOPs: n_d adds per row for partial D sums, 6 adds per row for the
        # butterfly reduction, and N divs per row for the C update.
        flops = M * (n_d + 6 + N)
        # Memory: C read (bf16) + D read (f32) + C write (bf16).
        bytes_transferred = M * N * 2 + M * n_d * 4 + M * N * 2

        print(
            f"\n  M={M} N={N} K={K}: {elapsed_s * 1e6:.2f} µs | "
            f"{flops / elapsed_s / 1e9:.2f} GFLOP/s | "
            f"{bytes_transferred / elapsed_s / 1e9:.2f} GB/s"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
