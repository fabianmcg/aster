"""E2E test: C[i,j] = C[i,j] / sqrt(inv_d * sum(D[i,:]) + eps) for all rows i and columns j.

Kernel: col_div in kernels/col_div.mlir.

Same computation as row_div, but C is stored column-major in memory
(C[i,j] at element offset j*M + i) while D remains row-major. Phase 3
groups threads 8-at-a-time; each group covers one 8-row band of C, with
the 8 lanes hitting 8 consecutive columns via buffer_load_dwordx4 /
buffer_store_dwordx4 (16 bytes = 8 bf16 rows of one column per lane, 128
bytes = one cache line per group) — the D row-sum reduction in phase 1 is
similarly vectorised via dwordx4 f32 loads.

Shape preconditions required by the kernel:
  n_d              % 16 == 0  — enables dwordx4 D loads.
  rows_per_block   %  8 == 0  — every block's row range is a full
                                multiple of 8 (no partial row-bands).
  M % rows_per_block == 0     — every block, including the last, covers
                                exactly rows_per_block rows.

Launch: block_dim=(128,1,1) — 2 wavefronts per block; must be a multiple
        of 8 (16 groups of 8 threads).
        grid_dim=(M // rows_per_block, 1, 1) — one block per row-group.

Parametrised over (M, N, K, rows_per_block) where D has shape (M, N//K)
and n_d = N // K. M must be divisible by rows_per_block.
"""

import os

import ml_dtypes
import numpy as np
import pytest

from aster.execution.core import InOutArray
from aster.execution.helpers import compile_and_run
from aster.pass_pipelines import PipelineConfig, make_default_pass_pipeline

PASS_PIPELINE = make_default_pass_pipeline(PipelineConfig(scf_pipeline=False))

MLIR_FILE = os.path.join(os.path.dirname(__file__), "col_div.mlir")
KERNEL_NAME = "col_div"
WAVE_SIZE = 64
BLOCK_DIM = 2 * WAVE_SIZE  # 2 waves per block
BF16 = ml_dtypes.bfloat16

# (M, N, K, rows_per_block) — D shape is (M, N//K), n_d = N//K.
# Constraints: n_d % 16 == 0, rows_per_block % 8 == 0, M % rows_per_block == 0.
SHAPES = [
    # --- basic shapes, rpb=8 (1 band per block) ---
    (8, 128, 4, 8),  # n_d=32, single block
    (16, 256, 8, 8),  # n_d=32, 2 blocks
    (32, 64, 2, 8),  # n_d=32, 4 blocks
    (8, 64, 4, 8),  # n_d=16 (minimum), single block
    # --- multi-band blocks, rpb=16/24/32 ---
    (16, 128, 4, 16),  # n_d=32, rpb=16 (2 bands/block)
    (32, 256, 8, 16),  # n_d=32, rpb=16, 2 blocks
    (24, 96, 3, 24),  # n_d=32, rpb=24 (3 bands), 1 block
    (64, 128, 4, 32),  # n_d=32, rpb=32 (4 bands), 2 blocks
    # --- n_d edge values (multiples of 16) ---
    (8, 96, 3, 8),  # n_d=32
    (8, 128, 8, 8),  # n_d=16
    (16, 128, 2, 16),  # n_d=64
    (16, 256, 2, 16),  # n_d=128 (2x wave)
    (8, 512, 4, 8),  # n_d=128
    (16, 1024, 4, 16),  # n_d=256 (4x wave)
    # --- N not a multiple of 8 (partial column tile); n_d = N//K stays a
    # multiple of 16 even though N itself is not. ---
    (8, 65, 4, 8),  # n_d=16, N=65 (partial tile of 1 col)
    (8, 97, 6, 8),  # n_d=16, N=97 (partial tile of 1 col)
    (16, 33, 2, 16),  # n_d=16, N=33 (partial tile of 1 col)
    # --- large shapes ---
    (64, 1024, 32, 8),  # n_d=32, N=1024, many blocks
    (128, 2048, 64, 16),  # n_d=32, N=2048
    (64, 4096, 64, 8),  # n_d=64, N=4096
    (256, 256, 8, 8),  # n_d=32, many blocks
    (128, 128, 4, 32),  # n_d=32, rpb=32
]


INV_D = np.float32(1.0 / 32.0)
EPS = np.float32(1e-5)


@pytest.mark.parametrize("M,N,K,rows_per_block", SHAPES)
def test_col_div(M, N, K, rows_per_block):
    """C[i,j] / sqrt(inv_d * sum(D[i,:]) + eps) matches bf16 reference, C column-major."""
    n_d = N // K

    rng = np.random.default_rng(seed=42)
    C_f32 = rng.standard_normal((M, N)).astype(np.float32)
    # D rows must have positive sums so division is well-defined.
    D = rng.uniform(0.1, 2.0, (M, n_d)).astype(np.float32)

    # Reference: round-trip C through bf16 (matches what the kernel reads),
    # divide by sqrt(inv_d * row_sum + eps), truncate f32->bf16 (top 16 bits).
    C_bf16 = C_f32.astype(BF16)
    row_sums = D.sum(axis=1, keepdims=True)  # (M, 1)
    denom = np.sqrt(INV_D * row_sums + EPS).astype(np.float32)
    result_f32 = (C_bf16.astype(np.float32) / denom).astype(np.float32)
    result_u32 = result_f32.view(np.uint32) >> 16
    expected = result_u32.astype(np.uint16).view(BF16)

    # C is column-major in memory: flatten in Fortran (column-major) order.
    C_flat = np.asfortranarray(C_bf16).ravel(order="F")
    D_flat = np.ascontiguousarray(D).ravel()

    def verify(inputs, outputs):
        C_out = outputs[0].view(BF16).reshape((M, N), order="F")
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
            INV_D,
            EPS,
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
