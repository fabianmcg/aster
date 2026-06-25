"""End-to-end test for integer conversion ops (trunci / extui / select).

Runs a GPU kernel that computes per-lane:
  roundtrip = extui(trunci(a[i] : i32 -> i16) : i16 -> i32)  (zero-extends low 16 bits)
  select    = a[i] < b[i] ? a[i] : b[i]                       (signed elementwise min)
Checks that GPU results match the NumPy reference on random inputs.

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
def test_intconv_random(mcpu):
    """Roundtrip trunci/extui and signed select match numpy reference."""
    N = 64
    block = (64, 1, 1)

    rng = np.random.default_rng(42)
    a = rng.integers(-(1 << 31), (1 << 31), size=N, dtype=np.int32)
    b = rng.integers(-(1 << 31), (1 << 31), size=N, dtype=np.int32)

    roundtrip_out = np.zeros(N, dtype=np.int32)
    select_out = np.zeros(N, dtype=np.int32)

    def verify(inputs, outputs):
        a32 = inputs[0]
        b32 = inputs[1]
        # trunci i32->i16 keeps the low 16 bits; extui i16->i32 zero-extends.
        exp_roundtrip = (a32.astype(np.uint32) & 0xFFFF).astype(np.int32)
        # lsir.cmpi slt is a signed less-than; select picks a when a < b.
        exp_select = np.where(a32 < b32, a32, b32).astype(np.int32)
        np.testing.assert_array_equal(outputs[0], exp_roundtrip)
        np.testing.assert_array_equal(outputs[1], exp_select)

    compile_and_run(
        "intconv-e2e.mlir",
        "intconv_kernel",
        input_data=[a, b],
        output_data=[roundtrip_out, select_out],
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
