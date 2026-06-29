#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Apply the default ASTER pass pipeline to an MLIR input file.

Examples:
  compile.py kernel.mlir
  compile.py kernel.mlir --print-asm
  compile.py kernel.mlir --print-ir-after-all
  compile.py kernel.mlir --print-ir-after-all --print-dir /tmp/out
  compile.py kernel.mlir --print-timings
  compile.py kernel.mlir --kernel my_kernel --print-asm --print-dir /tmp/out
"""

import argparse

from aster import ir
from aster.compiler.core import PrintOptions, compile_mlir_file_to_asm
from aster.pass_pipelines import (
    PipelineConfig,
    make_default_pass_pipeline,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compile an MLIR file through the default ASTER pass pipeline."
    )
    p.add_argument("input", help="Path to the MLIR input file.")
    p.add_argument(
        "--kernel",
        default=None,
        help="Name of the kernel to extract. Uses the first amdgcn.module when omitted.",
    )

    # Pipeline configuration flags.
    pipeline = p.add_argument_group("pipeline options")
    pipeline.add_argument(
        "--no-scf-pipeline",
        dest="scf_pipeline",
        action="store_false",
        default=True,
        help="Disable the SCF pipelining phase.",
    )
    pipeline.add_argument(
        "--lcm-unroll",
        action="store_true",
        default=False,
        help="Enable LCM unrolling in the SCF pipeline.",
    )
    pipeline.add_argument(
        "--unroll-factor-multiplier",
        type=int,
        default=1,
        metavar="N",
        help="Unroll factor multiplier (default: 1).",
    )
    pipeline.add_argument(
        "--no-epilogue-peeling",
        dest="epilogue_peeling",
        action="store_false",
        default=True,
        help="Disable epilogue peeling.",
    )
    pipeline.add_argument(
        "--prologue-peeling",
        type=int,
        default=0,
        metavar="N",
        help="Number of prologue stages to peel (default: 0).",
    )
    pipeline.add_argument(
        "--ll-sched",
        type=int,
        default=0,
        metavar="N",
        help="Low-latency schedule level (default: 0).",
    )
    pipeline.add_argument(
        "--hoist-wait",
        action="store_true",
        default=False,
        help="Hoist iter-arg wait ops in the backend.",
    )
    pipeline.add_argument(
        "--set-mfma-priority",
        action="store_true",
        default=False,
        help="Set MFMA instruction priority in the backend.",
    )
    pipeline.add_argument(
        "--rotate-stage",
        type=int,
        default=None,
        metavar="N",
        help="Enable stage rotation with the given stage count.",
    )
    pipeline.add_argument(
        "--num-vgprs",
        type=int,
        default=256,
        metavar="N",
        help="Maximum VGPRs available to the backend (default: 256).",
    )
    pipeline.add_argument(
        "--num-agprs",
        type=int,
        default=256,
        metavar="N",
        help="Maximum AGPRs available to the backend (default: 256).",
    )

    # Print / diagnostic options.
    printing = p.add_argument_group("print options")
    printing.add_argument(
        "--print-asm",
        action="store_true",
        default=False,
        help="Print the generated assembly to stdout (or --print-dir if set).",
    )
    printing.add_argument(
        "--print-ir-after-all",
        action="store_true",
        default=False,
        help="Print the IR after every compiler pass.",
    )
    printing.add_argument(
        "--print-preprocessed-ir",
        action="store_true",
        default=False,
        help="Print the IR as loaded, before any pass runs.",
    )
    printing.add_argument(
        "--print-dir",
        default=None,
        metavar="DIR",
        help=(
            "Root directory for print output. When set, IR and assembly are "
            "written to a timestamped subdirectory instead of stdout."
        ),
    )
    printing.add_argument(
        "--print-timings",
        action="store_true",
        default=False,
        help="Print per-pass timing information.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    config = PipelineConfig(
        lcm_unroll=args.lcm_unroll,
        unroll_factor_multiplier=args.unroll_factor_multiplier,
        epilogue_peeling=args.epilogue_peeling,
        prologue_peeling=args.prologue_peeling,
        ll_sched=args.ll_sched,
        hoist_wait=args.hoist_wait,
        set_mfma_priority=args.set_mfma_priority,
        rotate_stage=args.rotate_stage,
        scf_pipeline=args.scf_pipeline,
    )
    pipeline = make_default_pass_pipeline(
        config,
        num_vgprs=args.num_vgprs,
        num_agprs=args.num_agprs,
    )

    opts = PrintOptions.from_flags(
        print_asm=args.print_asm,
        print_ir_after_all=args.print_ir_after_all,
        print_preprocessed_ir=args.print_preprocessed_ir,
        print_root_dir=args.print_dir,
        print_timings=args.print_timings,
    )

    with ir.Context() as ctx:
        asm, _ = compile_mlir_file_to_asm(
            args.input,
            args.kernel,
            pipeline,
            ctx,
            print_opts=opts,
        )

    if not args.print_asm:
        print(asm)


if __name__ == "__main__":
    main()
