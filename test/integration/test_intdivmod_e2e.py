"""End-to-end test for integer unsigned divide and modulo.

Runs a GPU kernel that computes per-lane unsigned division and modulo
using the lsir.divui / lsir.remui ops lowered to AMDGCN via the Newton-
Raphson reciprocal sequence. Checks that GPU results match NumPy
reference on random inputs.
"""

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

MCPU = "gfx950"
WAVEFRONT_SIZE = 64


def test_intdivmod_random():
    """Unsigned divide and modulo with random positive inputs match numpy."""
    N = 64
    block = (64, 1, 1)

    rng = np.random.default_rng(0)
    # Dividend in [1, 1_000_000), divisor in [1, 1000) to avoid div-by-zero.
    a = rng.integers(1, 1_000_000, size=N, dtype=np.int32)
    b = rng.integers(1, 1000, size=N, dtype=np.int32)

    div_out = np.zeros(N, dtype=np.int32)
    mod_out = np.zeros(N, dtype=np.int32)

    def verify(inputs, outputs):
        # Use unsigned semantics matching lsir.divui / lsir.remui behaviour.
        a32 = inputs[0].astype(np.uint32)
        b32 = inputs[1].astype(np.uint32)
        exp_div = (a32 // b32).astype(np.int32)
        exp_mod = (a32 % b32).astype(np.int32)
        np.testing.assert_array_equal(outputs[0], exp_div)
        np.testing.assert_array_equal(outputs[1], exp_mod)

    compile_and_run(
        "intdivmod-e2e.mlir",
        "intdivmod_kernel",
        input_data=[a, b],
        output_data=[div_out, mod_out],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=block,
        grid_dim=(1, 1, 1),
        verify_fn=verify,
        library_paths=[],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
