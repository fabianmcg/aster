"""End-to-end test for floating-point conversion ops (sitofp / extf / truncf / fptosi).

Runs a GPU kernel that computes per-lane:
  f32  = sitofp a[i] : i32 -> f32
  f64  = extf f32    : f32 -> f64
  f32b = truncf f64  : f64 -> f32
  r    = fptosi f32b : f32 -> i32
Inputs are in [-1000, 1000) so every value is exactly representable in f32,
making the roundtrip exact: r == a[i].

Parametrized over CDNA3 (gfx942) and CDNA4 (gfx950): both are wave64.
"""

import re

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

# Wave64 targets.
TARGET_CONFIGS = ["gfx942", "gfx950"]

WAVEFRONT_SIZE = 64


def _retarget(mcpu):
    """Return a preprocess function that rewrites the MLIR target string."""

    def preprocess(mlir_src):
        return re.sub(r"#amdgcn\.target<\w+>", f"#amdgcn.target<{mcpu}>", mlir_src)

    return preprocess


@pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
def test_fpconv_roundtrip(mcpu):
    """Roundtrip sitofp/extf/truncf/fptosi is exact for small integers."""
    N = 64
    block = (64, 1, 1)

    rng = np.random.default_rng(42)
    # All integers in [-1000, 1000) are exactly representable in f32.
    a = rng.integers(-1000, 1000, size=N, dtype=np.int32)

    out = np.zeros(N, dtype=np.int32)

    def verify(inputs, outputs):
        np.testing.assert_array_equal(outputs[0], inputs[0])

    compile_and_run(
        "fpconv-e2e.mlir",
        "fpconv_kernel",
        input_data=[a],
        output_data=[out],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=mcpu,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=block,
        grid_dim=(1, 1, 1),
        verify_fn=verify,
        library_paths=[],
        preprocess=_retarget(mcpu),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
