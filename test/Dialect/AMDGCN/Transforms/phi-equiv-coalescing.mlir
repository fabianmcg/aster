// RUN: aster-opt %s --aster-amdgcn-bufferization --aster-disable-verifiers --aster-suppress-disabled-verifier-warning | FileCheck %s --check-prefix=COALESCE
// RUN: aster-opt %s --amdgcn-reg-alloc --aster-disable-verifiers --aster-suppress-disabled-verifier-warning | FileCheck %s --check-prefix=ALLOC

// ============================================================================
// Test: Phi-equivalence coalescing in the reg-alloc pipeline
// ============================================================================
//
// When a block argument has a single alloca flowing through it from all
// predecessors, the step-2 fresh alloca and the original alloca are
// phi-equivalent. The reg_coalesce op declares this, causing the
// interference graph to merge them into one node. RegisterColoring then
// assigns them the same register, making the copy a self-copy (eliminated).
//
// This is the fix for Bug 14: without coalescing, the copy would need a
// s_waitcnt before it in the pipelined-loop case.

// --- After bufferization: check that reg_coalesce ops are emitted ---
//
// COALESCE-LABEL: kernel @simple_loop_phi_equiv
//
// Step 2 creates fresh allocas and copies for the block argument.
// reg_coalesce declares them phi-equivalent with the step-1 allocas.
// COALESCE-DAG:   reg_coalesce
//
// Block arguments are eliminated (cf.br with no args).
// COALESCE:        cf.br ^bb1
// COALESCE:      ^bb1:
// COALESCE-NOT:    ^bb1(

// --- After full pipeline: coalesced allocas get the same register ---
//
// ALLOC-LABEL:   kernel @simple_loop_phi_equiv
//
// Only one alloca needed for the loop counter (all phi-equiv merged).
// ALLOC:           %[[REG:.*]] = alloca : !amdgcn.sgpr<[[N:[0-9]+]]>
//
// Loop body: sop2 writes to same register it reads from (self-update).
// No extra copy instructions for the loop-carried value.
// ALLOC:         ^bb1:
// ALLOC:           sop2 s_add_u32 outs %[[REG]] ins %[[REG]],
// ALLOC:           cf.cond_br
//
// No reg_coalesce ops remain in the output.
// ALLOC-NOT:       reg_coalesce

amdgcn.module @phi_equiv_test target = <gfx942> isa = <cdna3> {
  kernel @simple_loop_phi_equiv {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32

    // Single alloca used for the loop counter
    %s_k = alloca : !amdgcn.sgpr
    %s_cmp = alloca : !amdgcn.sgpr

    // Initialize counter
    %k_init = sop1 s_mov_b32 outs %s_k ins %c0 : !amdgcn.sgpr, i32

    cf.br ^loop(%k_init : !amdgcn.sgpr)

  ^loop(%k: !amdgcn.sgpr):
    // Advance counter
    %k_next = sop2 s_add_u32 outs %s_cmp ins %k, %c1 : !amdgcn.sgpr, !amdgcn.sgpr, i32
    %done = lsir.cmpi i32 slt %k_next, %c10 : !amdgcn.sgpr, i32
    cf.cond_br %done, ^loop(%k_next : !amdgcn.sgpr), ^exit

  ^exit:
    end_kernel
  }
}
