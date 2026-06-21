# Aster Fundamentals

ASTER is an MLIR-based compiler backend and assembly generation toolchain for AMD GPUs, bridging high-level ML compilers and low-level hardware control.

## Dialects

- `AMDGCN` — Low-level AMD GPU ISA instruction representation (VOPx, MUBUF, DS, SMEM, etc.); maps directly to hardware assembly using SSA form.
- `AMDGPU` — Higher-level AMD GPU operations that lower to AMDGCN.
- `LSIR` — Load-Store Instruction Register dialect; intermediate representation layer for load/store register architectures.
- `Layout` — Composable algebra for representing and transforming memory layouts (strided, hierarchical, etc.) without explicit computation.
- `AsterUtils` — Utility operations supporting the ASTER framework and analysis passes.

## Register Semantics

Three kinds, each with a single and range form:

- **Value**: bare type — `!amdgcn.vgpr` (single), `!amdgcn.vgpr<[? + 2]>` (range)
  ```mlir
  %result = amdgcn.v_mov_b32 outs(%dst) ins(%src0) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
  ```
- **Unallocated**: unknown register — `!amdgcn.vgpr<?>` (single), `!amdgcn.vgpr<[? : ? + 2]>` (range)
  ```mlir
  amdgcn.v_mov_b32 outs(%dst) ins(%src0) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.sgpr<?>)
  ```
- **Allocated**: physically assigned — `!amdgcn.vgpr<0>` (single), `!amdgcn.vgpr<[2 : 4]>` (range)
  ```mlir
  amdgcn.v_mov_b32 outs(%dst) ins(%src0) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.sgpr<1>)
  ```

Special registers (e.g. `!amdgcn.exec`, `!amdgcn.scc`) only accept value or allocated semantics and do not accept ranges; `!amdgcn.scc<0>` is *allocated* scc.

Composite registers (e.g. `!amdgcn.vcc`) must be constructed from their parts via `make_register_range` — `vcc` is formed from `vcc_lo` and `vcc_hi`.

## DPS (Destination-Passing Style)

An instruction produces an SSA result only when its `outs` operand has *value* semantics. Allocated and unallocated outs produce no result. The same instruction can appear in all three forms:

```mlir
// Value semantics — produces an SSA result
%result = amdgcn.v_mov_b32 outs(%dst) ins(%src0) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)

// Unallocated — no result
amdgcn.v_mov_b32 outs(%dst) ins(%src0) : outs(!amdgcn.vgpr<?>) ins(!amdgcn.sgpr<?>)

// Allocated — no result
amdgcn.v_mov_b32 outs(%dst) ins(%src0) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.sgpr<1>)
```

---

# Project Info

- LLVM source repo: `~/llvm-project`; LLVM installation (not a repo): `~/shared-llvm`.
- Main aster git repo: `~/aster`.
- FileCheck is usually at `<llvm-project>/build/bin/FileCheck`.
- List of additional instructions that can be added to the project ~/amd-insts.td
- Markdown versions of the AMD GPUS ISA specs can be found in ~/specs.

---

# Coding Style

Follow `llvm/docs/CodingStandards.rst`. LLVM data structures are preferred over STL equivalents; see `llvm/include/ADT`, `llvm/include/Support`, and `llvm/docs/ProgrammersManual.rst`.

- Only use `auto` when the right-hand side is a cast, a constructor, a static get function, or an iterator type; spell out the full type otherwise.
- Never use an integer literal in `SmallVector` template parameters.
- Always pass `Attribute`, `Type`, `Value`, and derived classes by value; they are designed to be copied cheaply.
- Don't put braces around a single-line statement.
- Use `break` and `continue` and invert conditions to reduce nesting depth.
- Prefer early returns.
- If an `if` body ends with a `return`, omit the `else` — place the else code at the outer level:
  ```cpp
  // Don't:
  if (cond) {
    // ...
    return val;
  } else {
    // else body
  }

  // Prefer:
  if (cond) {
    // ...
    return val;
  }
  // else body
  ```
- Mark functions as `static` instead of putting them in an anonymous namespace. Struct declarations in a `.cpp` file should be defined in the anonymous namespace.
- Always declare the pass struct before everything else (after header includes); define its methods outside the struct body.
- Prefer signed types over unsigned; cast unsigned values to signed when needed.
- Never use `\p` or `\c` in C++ comments.
- End comments with a full stop and use proper punctuation. Assertion and diagnostic messages start with a lowercase letter and do not end with a full stop.

---

# Code Quality

- Keep functions short and focused. If a function grows beyond ~40 lines, consider splitting it.
- One responsibility per function or class; avoid mixing concerns.
- Encapsulate state, prefer composition over inheritance, and avoid unnecessary coupling.
- Code should be self-documenting through clear naming. Comments explain *why*, not *what*:
  ```cpp
  // Bad — restates the code.
  // Increment the counter.
  ++counter;

  // Good — explains a non-obvious reason.
  // Skip zero-sized types as they have no runtime representation.
  if (type.isZeroSized())
    return;
  ```
- Keep comments brief — one sentence is usually enough.
- Commit messages should be short and factual, describing what changed and why:
  ```
  // Bad
  Fix the bug in the pass that was causing incorrect results when processing
  operations with multiple results by checking the result count before...

  // Good
  Fix incorrect result handling for multi-result ops in FooPass.
  ```
- Avoid redundant abstractions; don't design for hypothetical future use.

---

# Building

```
source ~/.aster/bin/activate
bash <source-dir>/tools/setup.sh --clang=clang --clang++=clang++ --lld=lld --skip-requirements --skip-llvm --no-install
source <source-dir>/sandbox/bin/activate_sandbox
```

Before running `ninja`, `lit`, or `pytest`, always activate the sandbox first (`source <source-dir>/sandbox/bin/activate_sandbox`). `<source-dir>` refers to `~/aster` or the active worktree.

Use `lit` not `llvm-lit`, and if inside of the sandbox no need to invoke it with python.

Never call `ninja install`; only build targets.

---

# Testing

- Always verify `ninja check-aster` passes before considering a change complete.
- When creating tests, follow `/home/fmoracor/llvm-project/mlir/docs/Diagnostics.md` and https://mlir.llvm.org/getting_started/TestingGuide/.
- Always look for testing gaps: add positive cases (expected success) and negative cases (expected failure/rejection) where meaningful. Do not add redundant or trivially useless tests.

---

# General Guidelines

- Prefer LLVM data structures to STL equivalents.
- If it is unclear which directory or worktree to use, ask the user.
