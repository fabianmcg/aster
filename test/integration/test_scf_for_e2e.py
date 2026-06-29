"""E2E tests for scf.for lowering in Aster's SCF-to-CF pass.

Three kernels exercise the main for-loop paths:
  - With iter_args: one accumulator, sum(0..7) = 28.
  - Multiple iter_args: two accumulators (sum and last-step value).
  - Zero trip: lb >= ub; body never executes; init value flows through unchanged.

All tests use PASS_PIPELINE (default pipeline with scf_pipeline disabled).
"""

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.pass_pipelines import PipelineConfig, make_default_pass_pipeline

PASS_PIPELINE = make_default_pass_pipeline(PipelineConfig(scf_pipeline=False))

TOTAL_LANES = 64
MLIR_FILE = "scf-for-e2e.mlir"


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
            pass_pipeline=PASS_PIPELINE,
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
            pass_pipeline=PASS_PIPELINE,
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
            pass_pipeline=PASS_PIPELINE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
