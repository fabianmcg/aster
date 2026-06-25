"""E2E test for divergent control-flow lowering.

A divergent condition (tid_x < 32) gates a per-lane global store via
lsir.cond_br on VCC.  Lanes 0..31 write 1; lanes 32..63 leave the
output at 0 (the host pre-initializes the buffer to 0).

This exercises the LegalizeCF / VCC-branch path end-to-end on real GPU
hardware and verifies per-lane EXEC semantics.

Parametrized over CDNA3 (gfx942) and CDNA4 (gfx950): both are wave64
and emit s_mov_b64 / s_and_b64 / s_or_b64 for divergent control flow.
"""

import re

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

# Wave64 targets.
TARGET_CONFIGS = ["gfx942", "gfx950"]

WAVEFRONT_SIZE = 64
TOTAL_LANES = 64
MLIR_FILE = "divergent-cf-e2e.mlir"


def _retarget(mcpu):
    """Return a preprocess function that rewrites the MLIR target string."""

    def preprocess(mlir_src):
        return re.sub(r"#amdgcn\.target<\w+>", f"#amdgcn.target<{mcpu}>", mlir_src)

    return preprocess


@pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
def test_divergent_cf_per_lane(mcpu):
    """Lanes 0..31 write 1; lanes 32..63 skip -> output stays 0.

    Verifies that divergent conditional execution correctly splits lanes:
    only the lanes satisfying tid < 32 perform the store.
    """
    output = np.zeros(TOTAL_LANES, dtype=np.int32)

    def verify(inputs, outputs):
        buf = outputs[0]
        expected = np.zeros(TOTAL_LANES, dtype=np.int32)
        expected[:32] = 1
        np.testing.assert_array_equal(
            buf,
            expected,
            err_msg="lanes 0..31 should be 1, lanes 32..63 should be 0",
        )

    compile_and_run(
        MLIR_FILE,
        "divergent_cf_kernel",
        input_data=[],
        output_data=[output],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=mcpu,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=(TOTAL_LANES, 1, 1),
        grid_dim=(1, 1, 1),
        verify_fn=verify,
        library_paths=[],
        preprocess=_retarget(mcpu),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
