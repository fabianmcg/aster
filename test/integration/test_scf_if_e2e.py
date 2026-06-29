"""E2E tests for scf.if lowering in Aster's SCF-to-CF pass.

Six kernels exercise the main if/else paths:
  - Else uniform: uniform condition; else branch; no results.
  - Divergent no-else: per-lane condition; no else; no results.
  - Divergent with else: per-lane condition; else branch; no results.
  - Divergent with results: per-lane condition; stores tid/0 per branch.
  - Complex AND condition: (tid >= 16) AND (tid < 48); no else.
  - Complex OR condition: (tid < 16) OR (tid >= 48); no else.

All tests use PASS_PIPELINE (default pipeline with scf_pipeline disabled).
Divergent and compound-condition tests are parametrized over CDNA3 (gfx942)
and CDNA4 (gfx950).
"""

import re

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.pass_pipelines import PipelineConfig, make_default_pass_pipeline

PASS_PIPELINE = make_default_pass_pipeline(PipelineConfig(scf_pipeline=False))

TARGET_CONFIGS = ["gfx942", "gfx950"]

WAVEFRONT_SIZE = 64
TOTAL_LANES = 64
MLIR_FILE = "scf-if-e2e.mlir"


def _retarget(mcpu):
    """Return a preprocess function that rewrites the MLIR target string."""

    def preprocess(mlir_src):
        return re.sub(r"target = <\w+>", f"target = <{mcpu}>", mlir_src)

    return preprocess


class TestIfElseUniform:
    """Uniform scf.if with else: lanes 0..31 write 1, lanes 32..63 write 2."""

    def test_if_else_uniform(self):
        """Output[0..31] should be 1, output[32..63] should be 2."""
        output = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            expected = np.empty(TOTAL_LANES, dtype=np.int32)
            expected[:32] = 1
            expected[32:] = 2
            np.testing.assert_array_equal(
                buf,
                expected,
                err_msg="output[0..31] should be 1, output[32..63] should be 2",
            )

        compile_and_run(
            MLIR_FILE,
            "test_if_else_uniform",
            input_data=[],
            output_data=[output],
            pass_pipeline=PASS_PIPELINE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestIfDivergentNoElse:
    """Divergent scf.if with no else: lanes 0..31 write 1, others skip."""

    @pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
    def test_if_divergent_no_else(self, mcpu):
        """Output[0..31] should be 1; output[32..63] should remain 0."""
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
            "test_if_divergent_no_else",
            input_data=[],
            output_data=[output],
            pass_pipeline=PASS_PIPELINE,
            mcpu=mcpu,
            wavefront_size=WAVEFRONT_SIZE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
            preprocess=_retarget(mcpu),
        )


class TestIfDivergentWithElse:
    """Divergent scf.if with else: lanes 0..31 write 1, lanes 32..63 write 2."""

    @pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
    def test_if_divergent_with_else(self, mcpu):
        """Output[0..31] should be 1, output[32..63] should be 2."""
        output = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            expected = np.empty(TOTAL_LANES, dtype=np.int32)
            expected[:32] = 1
            expected[32:] = 2
            np.testing.assert_array_equal(
                buf,
                expected,
                err_msg="lanes 0..31 should be 1, lanes 32..63 should be 2",
            )

        compile_and_run(
            MLIR_FILE,
            "test_if_divergent_with_else",
            input_data=[],
            output_data=[output],
            pass_pipeline=PASS_PIPELINE,
            mcpu=mcpu,
            wavefront_size=WAVEFRONT_SIZE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
            preprocess=_retarget(mcpu),
        )


class TestIfDivergentWithResults:
    """Divergent scf.if: then-branch stores tid, else-branch stores 0."""

    @pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
    def test_if_divergent_with_results(self, mcpu):
        """Output[0..31] should equal the lane index; output[32..63] should be 0."""
        output = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            expected = np.zeros(TOTAL_LANES, dtype=np.int32)
            expected[:32] = np.arange(32, dtype=np.int32)
            np.testing.assert_array_equal(
                buf,
                expected,
                err_msg="lanes 0..31 should store their tid, lanes 32..63 should be 0",
            )

        compile_and_run(
            MLIR_FILE,
            "test_if_divergent_with_results",
            input_data=[],
            output_data=[output],
            pass_pipeline=PASS_PIPELINE,
            mcpu=mcpu,
            wavefront_size=WAVEFRONT_SIZE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
            preprocess=_retarget(mcpu),
        )


class TestIfComplexConditionAnd:
    """Compound AND condition: (tid >= 16) AND (tid < 48)."""

    @pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
    def test_if_complex_condition_and(self, mcpu):
        """Output[16..47] should be 1; all other slots should be 0."""
        output = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            expected = np.zeros(TOTAL_LANES, dtype=np.int32)
            expected[16:48] = 1
            np.testing.assert_array_equal(
                buf,
                expected,
                err_msg="lanes 16..47 should be 1 (AND condition), others 0",
            )

        compile_and_run(
            MLIR_FILE,
            "test_if_complex_condition_and",
            input_data=[],
            output_data=[output],
            pass_pipeline=PASS_PIPELINE,
            mcpu=mcpu,
            wavefront_size=WAVEFRONT_SIZE,
            block_dim=(TOTAL_LANES, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
            preprocess=_retarget(mcpu),
        )


class TestIfComplexConditionOr:
    """Compound OR condition: (tid < 16) OR (tid >= 48)."""

    @pytest.mark.parametrize("mcpu", TARGET_CONFIGS)
    def test_if_complex_condition_or(self, mcpu):
        """Output[0..15] and output[48..63] should be 1; output[16..47] should be 0."""
        output = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            buf = outputs[0]
            expected = np.zeros(TOTAL_LANES, dtype=np.int32)
            expected[:16] = 1
            expected[48:] = 1
            np.testing.assert_array_equal(
                buf,
                expected,
                err_msg="lanes 0..15 and 48..63 should be 1 (OR condition), others 0",
            )

        compile_and_run(
            MLIR_FILE,
            "test_if_complex_condition_or",
            input_data=[],
            output_data=[output],
            pass_pipeline=PASS_PIPELINE,
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
