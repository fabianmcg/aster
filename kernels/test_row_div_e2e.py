"""E2E test: C[i,j] = C[i,j] / sum(D[i,:]) for all rows i and columns j.

Kernel: row_div in kernels/row_div.mlir.

Launch: block_dim=(128,1,1) — 2 wavefronts per block.
        grid_dim=(M // rows_per_block, 1, 1) — one block per row-group.

Wave 0 reduces rows_per_block rows of D (each row has n_d <= 64 elements)
via buffer_load_dword with the OOB trick, 6-round butterfly reduction, and
writes f32 row sums to LDS. All threads then cooperatively divide C.

C is bf16 in memory; D is f32. The kernel loads bf16->f32, divides, and
truncates back to bf16 (top-16-bit truncation, no rounding).

Parametrised over (M, N, K, rows_per_block) where D has shape (M, N//K)
and n_d = N // K <= 64. M must be divisible by rows_per_block.
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
BLOCK_DIM = 2 * WAVE_SIZE  # 2 waves per block
BF16 = ml_dtypes.bfloat16

# (M, N, K, rows_per_block) — D shape is (M, N//K), n_d = N//K <= 64.
# M must be divisible by rows_per_block.
SHAPES = [
    # --- original shapes ---
    (4, 128, 4, 2),  # n_d=32, rpb=2
    (8, 256, 8, 4),  # n_d=32, rpb=4
    (4, 64, 2, 4),  # n_d=32, rpb=4
    (8, 128, 4, 8),  # n_d=32, rpb=8
    (4, 64, 64, 4),  # n_d=1,  rpb=4
    (4, 256, 8, 2),  # n_d=32, rpb=2
    (16, 128, 4, 4),  # n_d=32, rpb=4
    (8, 64, 2, 2),  # n_d=32, rpb=2
    # --- edge / irregular ---
    (1, 64, 1, 1),  # M=1, n_d=64, single row
    (1, 128, 2, 1),  # M=1, n_d=64, N=128
    (3, 96, 3, 3),  # M=rpb (1 block), n_d=32
    (5, 200, 10, 3),  # M=5, rpb=3 → last block has 2 rows
    (7, 112, 7, 3),  # M=7, rpb=3 → blocks of 3, 3, 1
    (9, 108, 3, 4),  # M=9, rpb=4 → blocks of 4, 4, 1; n_d=36
    (6, 100, 4, 4),  # M=6, rpb=4 → blocks of 4, 2; n_d=25
    # --- n_d > 64 ---
    (4, 512, 4, 2),  # n_d=128 (2× wave)
    (8, 512, 4, 4),  # n_d=128, larger M
    (4, 1024, 4, 4),  # n_d=256 (4× wave)
    (2, 128, 1, 2),  # n_d=128, M=2
    # --- large N ---
    (16, 1024, 32, 4),  # n_d=32, N=1024
    (32, 2048, 64, 8),  # n_d=32, N=2048
    (16, 4096, 64, 8),  # n_d=64, N=4096
    # --- large M with partial last block ---
    (100, 256, 8, 7),  # 14 full blocks + 1 partial (2 rows)
    (127, 128, 4, 16),  # 7 full blocks + 1 partial (15 rows)
]


@pytest.mark.parametrize("M,N,K,rows_per_block", SHAPES)
def test_row_div(M, N, K, rows_per_block):
    """C[i,j] / sum(D[i,:]) matches bf16 reference for all (i,j)."""
    n_d = N // K

    rng = np.random.default_rng(seed=42)
    C_f32 = rng.standard_normal((M, N)).astype(np.float32)
    # D rows must have positive sums so division is well-defined.
    D = rng.uniform(0.1, 2.0, (M, n_d)).astype(np.float32)

    # Reference: round-trip C through bf16 (matches what the kernel reads),
    # divide by row sum, then truncate f32->bf16 (top 16 bits, no rounding).
    C_bf16 = C_f32.astype(BF16)
    row_sums = D.sum(axis=1, keepdims=True)  # (M, 1)
    result_f32 = (C_bf16.astype(np.float32) / row_sums).astype(np.float32)
    result_u32 = result_f32.view(np.uint32) >> 16
    expected = result_u32.astype(np.uint16).view(BF16)

    C_flat = np.ascontiguousarray(C_bf16).ravel()
    D_flat = np.ascontiguousarray(D).ravel()

    def verify(inputs, outputs):
        C_out = outputs[0].view(BF16).reshape(M, N)
        # Use allclose: the kernel butterfly reduces in a different order than
        # numpy's sum, causing at most 1 ULP of bf16 difference on large n_d.
        np.testing.assert_allclose(
            C_out.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=0,
            err_msg=f"M={M} N={N} K={K} rpb={rows_per_block}: mismatch",
        )

    times_ns = compile_and_run(
        MLIR_FILE,
        KERNEL_NAME,
        input_data=[
            InOutArray(C_flat),
            D_flat,
            np.int32(M),
            np.int32(N),
            np.int32(n_d),
            np.int32(rows_per_block),
        ],
        output_data=[],
        pass_pipeline=PASS_PIPELINE,
        block_dim=(BLOCK_DIM, 1, 1),
        grid_dim=(-(-M // rows_per_block), 1, 1),
        verify_fn=verify,
        library_paths=[],
        num_iterations=10,
    )

    if times_ns:
        elapsed_s = min(times_ns) * 1e-9
        flops = M * (n_d + 6 + N)
        bytes_xfer = M * N * 2 + M * n_d * 4 + M * N * 2
        print(
            f"\n  M={M} N={N} K={K} rpb={rows_per_block}: "
            f"{elapsed_s * 1e6:.2f} µs | "
            f"{flops / elapsed_s / 1e9:.2f} GFLOP/s | "
            f"{bytes_xfer / elapsed_s / 1e9:.2f} GB/s"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
