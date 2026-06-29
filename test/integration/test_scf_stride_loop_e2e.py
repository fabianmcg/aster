"""E2E tests for scf.while stride-loop + scf.if control-flow lowering.

Five kernels exercise stride-loop patterns on gfx942 (block=128, grid=1):
  - StrideLoop: for (i=tid; i<n; i+=128) x[i]=i.
  - StrideLoopIfElse: store i+1 if i%2==0, else 2*i+1.
  - StrideLoopIfOnly: store i+1 only if i%2==0 (no else; odd slots stay -1).
  - StrideLoopElseOnly: store 2*i+1 only if i%2!=0 (no else; even slots stay -1).
  - StrideLoopCompoundAnd: store 6*i+1 if i%2==0&&i%3==0, else store i.

All tests use PASS_PIPELINE (default pipeline with scf_pipeline disabled) and are parametrized over N_VALUES.
"""

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.pass_pipelines import PipelineConfig, make_default_pass_pipeline

PASS_PIPELINE = make_default_pass_pipeline(PipelineConfig(scf_pipeline=False))

MCPU = "gfx942"
BLOCK = 128
STRIDE = 128
N_VALUES = [256, 255, 257, 417, 588]
MLIR_FILE = "scf-stride-loop-e2e.mlir"


class TestStrideLoop:
    """Stride loop storing i at x[i] for all i in [0, n)."""

    @pytest.mark.parametrize("n", N_VALUES)
    def test_stride_loop(self, n):
        """X[i] should equal i for all i in [0, n)."""
        x = np.full(n, -1, dtype=np.int32)

        def verify(inputs, outputs):
            expected = np.arange(n, dtype=np.int32)
            np.testing.assert_array_equal(
                outputs[0],
                expected,
                err_msg=f"stride loop failed for n={n}",
            )

        compile_and_run(
            MLIR_FILE,
            "test_stride_loop",
            input_data=[n],
            output_data=[x],
            pass_pipeline=PASS_PIPELINE,
            mcpu=MCPU,
            block_dim=(BLOCK, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestStrideLoopIfElse:
    """Stride loop with if/else: x[i]=i+1 if even, 2*i+1 if odd."""

    @pytest.mark.parametrize("n", N_VALUES)
    def test_stride_loop_if_else(self, n):
        """X[i] should be i+1 for even i, 2*i+1 for odd i, for i in [0, n)."""
        x = np.full(n, -1, dtype=np.int32)

        def verify(inputs, outputs):
            idx = np.arange(n, dtype=np.int32)
            expected = np.where(idx % 2 == 0, idx + 1, 2 * idx + 1).astype(np.int32)
            np.testing.assert_array_equal(
                outputs[0],
                expected,
                err_msg=f"stride loop if/else failed for n={n}",
            )

        compile_and_run(
            MLIR_FILE,
            "test_stride_loop_if_else",
            input_data=[n],
            output_data=[x],
            pass_pipeline=PASS_PIPELINE,
            mcpu=MCPU,
            block_dim=(BLOCK, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestStrideLoopIfOnly:
    """Stride loop with if only: x[i]=i+1 if even, else unchanged (-1)."""

    @pytest.mark.parametrize("n", N_VALUES)
    def test_stride_loop_if_only(self, n):
        """X[i] should be i+1 if i<n and i even, else -1."""
        x = np.full(n, -1, dtype=np.int32)

        def verify(inputs, outputs):
            idx = np.arange(n, dtype=np.int32)
            expected = np.where(idx % 2 == 0, idx + 1, -1).astype(np.int32)
            np.testing.assert_array_equal(
                outputs[0],
                expected,
                err_msg=f"stride loop if-only failed for n={n}",
            )

        compile_and_run(
            MLIR_FILE,
            "test_stride_loop_if_only",
            input_data=[n],
            output_data=[x],
            pass_pipeline=PASS_PIPELINE,
            mcpu=MCPU,
            block_dim=(BLOCK, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestStrideLoopElseOnly:
    """Stride loop with condition on odd indices: x[i]=2*i+1 if odd, else -1."""

    @pytest.mark.parametrize("n", N_VALUES)
    def test_stride_loop_else_only(self, n):
        """X[i] should be 2*i+1 if i<n and i odd, else -1."""
        x = np.full(n, -1, dtype=np.int32)

        def verify(inputs, outputs):
            idx = np.arange(n, dtype=np.int32)
            expected = np.where(idx % 2 != 0, 2 * idx + 1, -1).astype(np.int32)
            np.testing.assert_array_equal(
                outputs[0],
                expected,
                err_msg=f"stride loop else-only failed for n={n}",
            )

        compile_and_run(
            MLIR_FILE,
            "test_stride_loop_else_only",
            input_data=[n],
            output_data=[x],
            pass_pipeline=PASS_PIPELINE,
            mcpu=MCPU,
            block_dim=(BLOCK, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


class TestStrideLoopCompoundAnd:
    """Stride loop with compound AND: x[i]=6*i+1 if i%6==0, else x[i]=i."""

    @pytest.mark.parametrize("n", N_VALUES)
    def test_stride_loop_compound_and(self, n):
        """X[i] should be 6*i+1 if i%6==0 and i<n, else i."""
        x = np.full(n, -1, dtype=np.int32)

        def verify(inputs, outputs):
            idx = np.arange(n, dtype=np.int32)
            cond = (idx % 2 == 0) & (idx % 3 == 0)
            expected = np.where(cond, 6 * idx + 1, idx).astype(np.int32)
            np.testing.assert_array_equal(
                outputs[0],
                expected,
                err_msg=f"stride loop compound-and failed for n={n}",
            )

        compile_and_run(
            MLIR_FILE,
            "test_stride_loop_compound_and",
            input_data=[n],
            output_data=[x],
            pass_pipeline=PASS_PIPELINE,
            mcpu=MCPU,
            block_dim=(BLOCK, 1, 1),
            grid_dim=(1, 1, 1),
            verify_fn=verify,
            library_paths=[],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
