"""E2E tests for scf.for lowering in Aster's SCF-to-CF pass.

Five kernels exercise the main for-loop paths:
  - Uniform no iter_args: all lanes execute 8 iterations; result verified via tid*8.
  - With iter_args: one accumulator, sum(0..7) = 28.
  - Multiple iter_args: two accumulators (sum and last-step value).
  - Zero trip: lb >= ub; body never executes; init value flows through unchanged.
  - Divergent body: uniform bounds with a divergent nested scf.if.

Uniform tests use TEST_LOOP_PASS_PIPELINE.
The divergent-body test uses TEST_SROA_PASS_PIPELINE and is parametrized over
CDNA3 (gfx942) and CDNA4 (gfx950).
"""

import re

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_LOOP_PASS_PIPELINE, TEST_SROA_PASS_PIPELINE

TARGET_CONFIGS = ["gfx942", "gfx950"]

WAVEFRONT_SIZE = 64
TOTAL_LANES = 64
MLIR_FILE = "scf-for-e2e.mlir"


def _retarget(mcpu):
    """Return a preprocess function that rewrites the MLIR target string."""

    def preprocess(mlir_src):
        return re.sub(r"#amdgcn\.target<\w+>", f"#amdgcn.target<{mcpu}>", mlir_src)

    return preprocess


class TestForUniformNoIterArgs:
    """Uniform for loop with no iter_args: 8 iterations, result verified via tid*8."""

    def test_for_uniform_no_iter_args(self):
        """Output[i] should equal i * 8 after 8 uniform iterations."""
        output = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            expected = np.arange(TOTAL_LANES, dtype=np.int32) * 8
            np.testing.assert_array_equal(
                buf,
                expected,
                err_msg="output[i] should equal i * 8",
            )

        compile_and_run(
            MLIR_FILE,
            "test_for_uniform_no_iter_args",
            input_data=[],
            output_data=[output],
            pass_pipeline=TEST_LOOP_PASS_PIPELINE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestForWithIterArgs:
    """For loop with one iter_arg accumulating sum(0..7) = 28."""

    def test_for_with_iter_args(self):
        """Output[0] should be 28 (sum of integers 0 through 7)."""
        output = np.zeros(1, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            np.testing.assert_array_equal(
                buf,
                np.array([28], dtype=np.int32),
                err_msg="output[0] should be 28",
            )

        compile_and_run(
            MLIR_FILE,
            "test_for_with_iter_args",
            input_data=[],
            output_data=[output],
            pass_pipeline=TEST_LOOP_PASS_PIPELINE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestForMultipleIterArgs:
    """For loop with two iter_args: sum accumulator and last-step value."""

    def test_for_multiple_iter_args(self):
        """Output[0] should be 28 (sum), output[1] should be 8 (last i+1)."""
        output = np.zeros(2, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            np.testing.assert_array_equal(
                buf,
                np.array([28, 8], dtype=np.int32),
                err_msg="output[0]=28 (sum), output[1]=8 (last step)",
            )

        compile_and_run(
            MLIR_FILE,
            "test_for_multiple_iter_args",
            input_data=[],
            output_data=[output],
            pass_pipeline=TEST_LOOP_PASS_PIPELINE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestForZeroTrip:
    """Zero-trip for loop: lb >= ub, body never executes, init value unchanged."""

    def test_for_zero_trip(self):
        """Output[0] should be 99 (the init value, unchanged because body never runs)."""
        output = np.full(1, 99, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            np.testing.assert_array_equal(
                buf,
                np.array([99], dtype=np.int32),
                err_msg="output[0] should remain 99 (zero-trip loop)",
            )

        compile_and_run(
            MLIR_FILE,
            "test_for_zero_trip",
            input_data=[],
            output_data=[output],
            pass_pipeline=TEST_LOOP_PASS_PIPELINE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestForDivergentBody:
    """Uniform scf.for body containing a divergent nested scf.if."""

    @pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
    def test_for_divergent_body(self, mcpu):
        """Lanes 0..31 write their tid; lanes 32..63 skip -> output stays 0."""
        output = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            expected = np.zeros(TOTAL_LANES, dtype=np.int32)
            expected[:32] = np.arange(32, dtype=np.int32)
            np.testing.assert_array_equal(
                buf,
                expected,
                err_msg="lanes 0..31 should store their tid; lanes 32..63 should be 0",
            )

        compile_and_run(
            MLIR_FILE,
            "test_for_divergent_body",
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
