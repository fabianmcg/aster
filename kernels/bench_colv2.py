"""Benchmark for the colv2 kernel.

Usage:
    python bench_colv2.py --m M --n N --n_d N_D
                          [--iters ITERS] [--block_dim BLOCK_DIM]
                          [--inv_d INV_D] [--eps EPS]

C[i,j] = C[i,j] / sqrt(inv_d * sum(D[i,:]) + eps).  C is column-major bf16
(element offset j*M+i); D is row-major f32.  n_d must be <= 64.

Grid: (ceil(M/256), ceil(N/256), 1) — bid.x selects a 256-row band;
      bid.y selects a 256-column tile.

Reports memory bandwidth relative to a perfect memcopy of C (read + write),
i.e. 2 * M * N * 2 bytes / elapsed_s.

Correctness is verified on the first run against a numpy reference.
"""

import argparse
import os
import sys

import ml_dtypes
import numpy as np

from aster.execution.core import InOutArray
from aster.execution.helpers import compile_and_run
from aster.pass_pipelines import PipelineConfig, make_default_pass_pipeline

MLIR_FILE = os.path.join(os.path.dirname(__file__), "colv2.mlir")
KERNEL_NAME = "colv2"
BF16 = ml_dtypes.bfloat16


def run(
    m: int,
    n: int,
    n_d: int,
    iters: int,
    block_dim: int,
    inv_d: float,
    eps: float,
) -> None:
    pass_pipeline = make_default_pass_pipeline(PipelineConfig(scf_pipeline=False))

    inv_d_f32 = np.float32(inv_d)
    eps_f32 = np.float32(eps)

    rng = np.random.default_rng(seed=0)
    C_f32 = rng.standard_normal((m, n)).astype(np.float32)
    D = rng.uniform(0.1, 2.0, (m, n_d)).astype(np.float32)

    C_bf16 = C_f32.astype(BF16)
    # C is column-major in memory: flatten in Fortran (column-major) order.
    C_flat = np.asfortranarray(C_bf16).ravel(order="F")
    D_flat = np.ascontiguousarray(D).ravel()

    # Numpy reference: C / sqrt(inv_d * sum(D) + eps)
    row_sums = D.sum(axis=1, keepdims=True)
    denom = np.sqrt(inv_d_f32 * row_sums + eps_f32).astype(np.float32)
    result_f32 = (C_bf16.astype(np.float32) / denom).astype(np.float32)
    result_u32 = result_f32.view(np.uint32) >> 16
    expected = result_u32.astype(np.uint16).view(BF16)

    verified = [False]
    error_msg = [None]

    def verify(inputs, outputs):
        C_out = outputs[0].view(BF16).reshape((m, n), order="F")
        try:
            np.testing.assert_allclose(
                C_out.astype(np.float32),
                expected.astype(np.float32),
                rtol=1e-2,
                atol=0,
            )
            verified[0] = True
        except AssertionError as e:
            error_msg[0] = str(e)

    grid_dim = (-(-m // 256), -(-n // 256), 1)

    times_ns = compile_and_run(
        MLIR_FILE,
        KERNEL_NAME,
        input_data=[
            InOutArray(C_flat),
            D_flat,
            np.int32(m),
            np.int32(n),
            np.int32(n_d),
            inv_d_f32,
            eps_f32,
        ],
        output_data=[],
        pass_pipeline=pass_pipeline,
        block_dim=(block_dim, 1, 1),
        grid_dim=grid_dim,
        verify_fn=verify,
        library_paths=[],
        num_iterations=iters,
    )

    if error_msg[0]:
        print(f"VERIFICATION FAILED:\n{error_msg[0]}", file=sys.stderr)
        sys.exit(1)

    if not verified[0]:
        print("VERIFICATION SKIPPED (no GPU?)", file=sys.stderr)
        return

    print("Verification: PASSED")

    if not times_ns:
        return

    # C memory: read bf16 + write bf16 = 2 * m * n * 2 bytes.
    c_bytes = 2 * m * n * 2
    best_ns = min(times_ns)
    elapsed_s = best_ns * 1e-9

    bw_gbs = c_bytes / elapsed_s / 1e9

    print(
        f"\nConfig: M={m}  N={n}  n_d={n_d}"
        f"  inv_d={inv_d}  eps={eps}  block_dim={block_dim}  iters={iters}"
    )
    print(f"Best time:  {elapsed_s * 1e6:.3f} µs")
    print(f"C bandwidth (read + write bf16): {bw_gbs:.1f} GB/s")
    print(f"  (C bytes transferred: {c_bytes / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark colv2 kernel")
    parser.add_argument(
        "--m", type=int, required=True, help="Number of rows in C and D"
    )
    parser.add_argument("--n", type=int, required=True, help="Number of columns in C")
    parser.add_argument(
        "--n_d", type=int, required=True, help="Number of columns in D (max 64)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Kernel iterations for timing (default: 20)",
    )
    parser.add_argument(
        "--block_dim",
        type=int,
        default=256,
        help="Block dimension (threads per block, default: 256)",
    )
    parser.add_argument(
        "--inv_d",
        type=float,
        default=1.0,
        help="Inverse dimension scale factor (default: 1.0)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Epsilon added before sqrt for numerical stability (default: 1e-5)",
    )
    args = parser.parse_args()

    run(
        m=args.m,
        n=args.n,
        n_d=args.n_d,
        iters=args.iters,
        block_dim=args.block_dim,
        inv_d=args.inv_d,
        eps=args.eps,
    )


if __name__ == "__main__":
    main()
