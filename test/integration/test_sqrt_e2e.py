"""E2E test: per-lane IEEE f32 sqrt via lsir.sqrtf → v_sqrt_f32.

Kernel: sqrt_f32_kernel in sqrt-e2e.mlir.
Each thread computes sqrt(input[tid]) and stores to output[tid].

Verified against numpy.sqrt with 1ULP tolerance (v_sqrt_f32 is 1ULP accurate,
denormals are flushed, so we test only positive normal values).
"""

import os

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.pass_pipelines import PipelineConfig, make_default_pass_pipeline

PASS_PIPELINE = make_default_pass_pipeline(PipelineConfig(scf_pipeline=False))
MLIR_FILE = os.path.join(os.path.dirname(__file__), "sqrt-e2e.mlir")
KERNEL_NAME = "sqrt_f32_kernel"
WAVE_SIZE = 64


@pytest.mark.parametrize("n_elems", [64, 128, 256, 512])
def test_sqrt_f32(n_elems):
    """Sqrt(x) matches numpy.sqrt for positive normal f32 inputs."""
    rng = np.random.default_rng(seed=7)
    # Only positive normals: v_sqrt_f32 flushes denormals.
    inputs = rng.uniform(0.01, 100.0, n_elems).astype(np.float32)
    outputs = np.zeros(n_elems, dtype=np.float32)

    expected = np.sqrt(inputs)

    def verify(_, out_args):
        result = out_args[0].view(np.float32)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=2e-7,  # 1ULP for f32 is ~1.2e-7; allow 2× for rounding.
            atol=0,
            err_msg=f"n_elems={n_elems}: sqrt mismatch",
        )

    grid_dim = (n_elems // WAVE_SIZE, 1, 1)
    compile_and_run(
        MLIR_FILE,
        KERNEL_NAME,
        input_data=[inputs],
        output_data=[outputs],
        pass_pipeline=PASS_PIPELINE,
        block_dim=(WAVE_SIZE, 1, 1),
        grid_dim=grid_dim,
        verify_fn=verify,
        library_paths=[],
        num_iterations=5,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
