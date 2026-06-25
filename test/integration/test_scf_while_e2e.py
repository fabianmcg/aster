"""E2E tests for scf.while lowering in Aster's SCF-to-CF pass.

Four kernels exercise the main while-loop paths:
  - Uniform: all lanes run the same number of iterations.
  - Iter_args: accumulator pattern with two iteration variables.
  - Divergent: per-lane iteration count (different lanes exit at different times).
  - Early_exit: lanes cap out at 32 iterations (divergent condition).

Uniform and iter_args use TEST_LOOP_PASS_PIPELINE (no EXEC manipulation needed).
Divergent and early_exit use TEST_SROA_PASS_PIPELINE and are parametrized over
both CDNA3 (gfx942) and CDNA4 (gfx950) since they exercise the divergent
save_cf_mask / restore_cf_mask path.
"""

import re

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_LOOP_PASS_PIPELINE, TEST_SROA_PASS_PIPELINE

TARGET_CONFIGS = ["gfx942", "gfx950"]

WAVEFRONT_SIZE = 64
TOTAL_LANES = 64
MLIR_FILE = "scf-while-e2e.mlir"


def _retarget(mcpu):
    """Return a preprocess function that rewrites the MLIR target string."""

    def preprocess(mlir_src):
        return re.sub(r"#amdgcn\.target<\w+>", f"#amdgcn.target<{mcpu}>", mlir_src)

    return preprocess


class TestWhileUniform:
    """Uniform while loop: all 64 lanes run exactly 8 iterations."""

    def test_while_uniform(self):
        """Output[i] should equal i * 8 after the loop completes."""
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
            "test_while_uniform",
            input_data=[],
            output_data=[output],
            pass_pipeline=TEST_LOOP_PASS_PIPELINE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestWhileIterArgs:
    """While loop with two iter_args accumulating sum(0..7) = 28."""

    def test_while_iter_args(self):
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
            "test_while_iter_args",
            input_data=[],
            output_data=[output],
            pass_pipeline=TEST_LOOP_PASS_PIPELINE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestWhileDivergent:
    """Divergent while loop: lane i runs exactly i iterations."""

    @pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
    def test_while_divergent(self, mcpu):
        """Output[i] should equal i for all lanes 0..63."""
        output = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            expected = np.arange(TOTAL_LANES, dtype=np.int32)
            np.testing.assert_array_equal(
                buf,
                expected,
                err_msg="output[i] should equal i",
            )

        compile_and_run(
            MLIR_FILE,
            "test_while_divergent",
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


class TestWhileEarlyExit:
    """Divergent while loop capped at 32 iterations per lane."""

    @pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
    def test_while_early_exit(self, mcpu):
        """Output[i] should equal min(i, 32) for all lanes."""
        output = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            expected = np.minimum(
                np.arange(TOTAL_LANES, dtype=np.int32),
                np.full(TOTAL_LANES, 32, dtype=np.int32),
            )
            np.testing.assert_array_equal(
                buf,
                expected,
                err_msg="output[i] should equal min(i, 32)",
            )

        compile_and_run(
            MLIR_FILE,
            "test_while_early_exit",
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


class TestWhileDivergentTwoIterArgs:
    """Divergent while loop with two iter_args: cnt and accumulator."""

    @pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
    def test_while_divergent_two_iter_args(self, mcpu):
        """Output[i] should equal i*(i-1)/2 (partial sum 0+1+...+(i-1))."""
        output = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            lanes = np.arange(TOTAL_LANES, dtype=np.int32)
            expected = lanes * (lanes - 1) // 2
            np.testing.assert_array_equal(
                buf,
                expected,
                err_msg="output[i] should equal i*(i-1)/2",
            )

        compile_and_run(
            MLIR_FILE,
            "test_while_divergent_two_iter_args",
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


class TestWhileZeroTrip:
    """Zero-trip while loop: condition false on first evaluation; init unchanged."""

    def test_while_zero_trip(self):
        """Output[0] should be 99 (the init value, unchanged because body never runs)."""
        output = np.full(1, 99, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            np.testing.assert_array_equal(
                buf,
                np.array([99], dtype=np.int32),
                err_msg="output[0] should remain 99 (zero-trip while)",
            )

        compile_and_run(
            MLIR_FILE,
            "test_while_zero_trip",
            input_data=[],
            output_data=[output],
            pass_pipeline=TEST_LOOP_PASS_PIPELINE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
