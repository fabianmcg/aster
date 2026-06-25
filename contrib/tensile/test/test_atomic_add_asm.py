"""HSACO assembly round-trip test for global_atomic_add_f32 across CDNA3/CDNA4.

Float atomics must NOT set sc0 per the CDNA3/4 ISA spec. The builder
emits the no-return form (sc0=0), which the LLVM assembler selects as
the GLOBAL_ATOMIC_ADD_F32_no_rtn variant.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder
from aster.dialects.amdgcn import AccessKind
from aster.dialects._amdgcn_ops_gen import MakeRegisterRangeOp, v_mov_b32
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig

KERNEL_NAME = "atomic_test_kernel"


def _build_atomic_kernel(mcpu):
    """Build a minimal kernel with one global_atomic_add_f32 per thread.

    Uses a 2xVGPR address so no extra offset register is needed. The
    builder emits sc0=0 (no-return form) per the CDNA3/4 ISA spec.
    """
    b = KernelBuilder("atomic_test_mod", KERNEL_NAME, target=mcpu)
    b.set_grid_dims(1)
    b.set_block_dims(64)
    b.add_ptr_arg(AccessKind.ReadWrite)
    (ptr,) = b.load_args()

    c0 = b.constant_i32(0)
    # Build a 2xVGPR address (zero for this test).
    addr_lo = b.alloca_vgpr()
    addr_hi = b.alloca_vgpr()
    v_mov_b32(addr_lo, c0, loc=b._loc, ip=b._kip)
    v_mov_b32(addr_hi, c0, loc=b._loc, ip=b._kip)
    addr = MakeRegisterRangeOp(inputs=[addr_lo, addr_hi], loc=b._loc, ip=b._kip).result

    data = b.alloca_vgpr()
    v_mov_b32(data, c0, loc=b._loc, ip=b._kip)

    # No-return atomic: builder emits sc0=0, which the assembler accepts as
    # the GLOBAL_ATOMIC_ADD_F32_no_rtn variant.
    b.global_atomic_add_f32(data, addr)
    b.wait_vmcnt(0)
    return b.build()


class TestAtomicAddAsm:
    """Verify global_atomic_add_f32 compiles to a non-empty HSACO."""

    @pytest.mark.parametrize("mcpu", ["gfx942", "gfx950"])
    def test_hsaco_nonempty(self, mcpu):
        if not system_has_mcpu(mcpu):
            pytest.skip(f"{mcpu} GPU not available")

        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx:
            module = _build_atomic_kernel(mcpu)
            asm = compile_mlir_module_to_asm(
                module,
                pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
            )

        assert "global_atomic_add_f32" in asm, "expected atomic instruction in ASM"
        assert "sc0" not in asm, "float atomics must not set sc0 (CDNA3/4 ISA spec)"

        path = assemble_to_hsaco(asm, target=mcpu, wavefront_size=64)
        if path is None:
            pytest.skip(f"LLVM assembler not compiled with {mcpu} support")

        with hsaco_file(path):
            assert os.path.getsize(path) > 0, "HSACO must be non-empty"
