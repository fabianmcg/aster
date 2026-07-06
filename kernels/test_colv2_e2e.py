"""E2E test: C[i,j] = C[i,j] / sqrt(inv_d * sum(D[i,:]) + eps) for all rows i and columns j.

Kernel: colv2 in kernels/colv2.mlir.

C is column-major bf16 (C[i,j] at element offset j*M+i); D is row-major f32.
Constraint: n_d <= 64 (one D row per lane, reduced across a wavefront).

Grid: (ceil(M/256), ceil(N/256), 1).  Block: 256 threads = 4 wavefronts.
"""

import os

import ml_dtypes
import numpy as np
import pytest

from aster.execution.core import InOutArray
from aster.execution.helpers import compile_and_run
from aster.pass_pipelines import PipelineConfig, make_default_pass_pipeline

PASS_PIPELINE = make_default_pass_pipeline(PipelineConfig(scf_pipeline=False))

MLIR_FILE = os.path.join(os.path.dirname(__file__), "colv2.mlir")
KERNEL_NAME = "colv2"
BLOCK_DIM = 256
BF16 = ml_dtypes.bfloat16

INV_D = np.float32(1.0 / 32.0)
EPS = np.float32(1e-5)

# (M, N, n_d) — n_d <= 64.
SHAPES = [
    # Small, exact multiples of 256.
    (256, 256, 32),
    (512, 512, 16),
    (256, 1024, 64),
    # Partial column tiles (N not a multiple of 256).
    (256, 300, 32),
    (512, 100, 8),
    # Partial row tiles (M not a multiple of 256).
    (300, 256, 32),
    (100, 512, 16),
    (513, 257, 31),
    # Various n_d values.
    (256, 256, 1),
    (256, 256, 8),
    (512, 512, 31),
    (256, 256, 63),
    (256, 256, 64),
    # Combined partial tiles.
    (300, 300, 17),
    (513, 300, 32),
    # n_d edge: single element.
    (256, 512, 1),
    # Large shapes.
    (2048, 4096, 32),
    (2048, 8192, 64),
    # Both dims partial.
    (300, 700, 32),
    (700, 300, 16),
    # n_d=16 and n_d=8.
    (512, 256, 16),
    (256, 512, 8),
]


@pytest.mark.parametrize("M,N,n_d", SHAPES)
def test_colv2(M, N, n_d):
    """C[i,j] / sqrt(inv_d * sum(D[i,:]) + eps) matches bf16 reference, C column-major."""
    rng = np.random.default_rng(seed=42)
    C_f32 = rng.standard_normal((M, N)).astype(np.float32)
    # D rows have positive sums so division is well-defined.
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
        np.testing.assert_allclose(
            C_out.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=0,
            err_msg=f"M={M} N={N} n_d={n_d}: mismatch",
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
            INV_D,
            EPS,
        ],
        output_data=[],
        pass_pipeline=PASS_PIPELINE,
        block_dim=(BLOCK_DIM, 1, 1),
        grid_dim=(-(-M // 256), -(-N // 256), 1),
        verify_fn=verify,
        library_paths=[],
        num_iterations=10,
    )

    if times_ns:
        elapsed_s = min(times_ns) * 1e-9
        flops = M * (n_d + 6 + N)
        bytes_xfer = M * N * 2 + M * n_d * 4 + M * N * 2
        print(
            f"\n  M={M} N={N} n_d={n_d}: "
            f"{elapsed_s * 1e6:.2f} µs | "
            f"{flops / elapsed_s / 1e9:.2f} GFLOP/s | "
            f"{bytes_xfer / elapsed_s / 1e9:.2f} GB/s"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
