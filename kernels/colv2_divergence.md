# Divergence Report: `colv2.mlir` vs. `ScaleByRowNorm.cu`

Both kernels implement `C[i,j] = C[i,j] / sqrt(inv_d * sum(D[i,:]) + eps)` where C is
column-major bf16 and D is row-major f32, targeting ROWS_PER_BLOCK=256, COLS_PER_BLOCK=256.

---

## Phase 1 — D Reduction

### 1. OOB load guard

**CUDA:** Ternary — lane loads `d[row*nD + laneId]` only when `laneId < nD && row < m`,
otherwise uses literal `0.0f`. No memory access is issued for out-of-bounds lanes.

**MLIR:** Sets the byte offset to `-1` for OOB lanes (suppressed by the AMD buffer bounds
check on `num_records`), then forces the loaded value to `0.0f` via `arith.select`.

**Verdict:** Intentional/equivalent. The belt-and-suspenders approach (hardware OOB + select)
is safe and avoids a neighbouring live element polluting the sum.

---

### 2. Butterfly reduction rounds

**CUDA:** 5 rounds using `__shfl_xor(v, off)` with `off = 32, 16, 8, 4, 2, 1`.

**MLIR:** 6 rounds using `ds_bpermute_b32` with XOR strides `1, 2, 4, 8, 16, 32`.

**Verdict:** Intentional/equivalent. Both cover all 6 bits of a 64-lane wavefront and
produce the full wavegroup sum in every lane. `ds_bpermute_b32` is the AMDGCN substitute
for `__shfl_xor`.

---

### 3. Who writes to LDS

**CUDA:** Only lane 0 writes (`if (laneId == 0) ldsScale[...] = ...`).

**MLIR:** Only lane 0 writes the scale to LDS, guarded by `scf.if %is_lane0`. The other
63 lanes participate in the butterfly all-reduce but skip the store.

**Verdict:** Fixed — now matches CUDA (lane 0 only).

---

### 4. Value stored in LDS

**CUDA:** Stores `rsqrtf(invD * v + eps)` — the reciprocal square root.

**MLIR:** Stores `rsqrtf(invD * sum + eps)` — the reciprocal square root.

**Verdict:** Equivalent.

---

### 5. Numeric precision

**CUDA:** `rsqrtf` is a hardware-accelerated approximation (~1–2 ULP error).

**MLIR:** `lsir.rsqrtf` (hardware approximation, ~1 ULP).

**Verdict:** Equivalent — same precision as CUDA `rsqrtf`.

---

## Phase 2 — C Update

### 6. Row OOB guard

**CUDA:** `if (globalRow >= m) return;` — early exit for the whole thread.

**MLIR:** `scf.if %row_ok { ... }` wrapping the column loop.

**Verdict:** Equivalent.

---

### 7. Column clamp

**CUDA:** `jEnd = min(jStart + COLS_PER_BLOCK, (int)n)`.

**MLIR:** `arith.minsi %jEnd_unclamped_i, %n`.

**Verdict:** Equivalent.

---

### 8. Scale application

**CUDA:** `c[idx] = __bf16(float(c[idx]) * s)` — multiply by reciprocal sqrt.

**MLIR:** `arith.mulf %cv_f32, %scale` — multiply by reciprocal sqrt.

**Verdict:** Equivalent.

---

### 9. bf16 conversion

**CUDA:** `static_cast<float>(c[idx])` / `static_cast<__bf16>(result)`.

**MLIR:** `arith.extf` (bf16→f32) / `arith.truncf` (f32→bf16).

**Verdict:** Equivalent — both use round-to-nearest.

---

### 10. D pointer constness

**CUDA:** `const float *d`.

**MLIR:** `!ptr.ptr<#amdgcn.addr_space<global, read_write>>` — marked read-write.

**Verdict:** Minor imprecision. D is never written in the MLIR kernel; the attribute
is harmless.

---

## Potential Issues

### Issue A — Missing `nD > 64` guard ⚠️

**CUDA:** The host wrapper (`scaleByRowNorm`) throws `std::runtime_error` before
dispatch if `nD > 64`.

**MLIR:** No such guard exists. The kernel comment states `n_d <= 64` as a precondition,
but it is not enforced. If a caller passes `n_d > 64`, lanes 0–63 each load one element
and columns 64..n_d-1 are silently ignored. The row sum — and therefore every scaled
value in C — will be wrong.

**Recommendation:** Add a precondition assertion in the Python driver (bench/test), or
a
---

### Issue B — `i32` byte-offset truncation for large matrices ⚠️

**CUDA:** The element index is kept as `int64_t` throughout (`idx = j * m + globalRow`).

**MLIR:** The element index is computed in `index` type (64-bit on a 64-bit target), then
cast to `i32` via `arith.index_cast` for the pointer add. This wraps when
`(j * m + globalRow) * 2 > 2^31` (byte offset overflows `i32`).

In practice this is safe for the 8192×8192 benchmark (element offset peaks at ~67M,
byte offset ~134 MB, well within `i32` range), but becomes wrong for matrices where
`m * n > ~10^9` elements (~2 GB of bf16 data).

**Recommendation:** Either document the size limit (`m * n <= 2^30` elements) or compute
the byte offset in `index` and use a 64-bit pointer-add variant.

---

## Summary Table

| # | Category | CUDA | MLIR | Verdict |
|---|---|---|---|---|
| 1 | OOB load guard | Ternary, no load | `-1` offset + `arith.select` | Intentional/equivalent |
| 2 | Butterfly rounds | 5 rounds, `off = 32→1` | 6 rounds, stride `1→32` | Intentional/equivalent |
| 3 | LDS writer | Lane 0 only | Lane 0 only | Fixed — matches CUDA |
| 4 | Value in LDS | `rsqrtf(...)` | `rsqrtf(...)` | Equivalent |
| 5 | Precision | `rsqrtf` (~1 ULP) | `rsqrtf` (~1 ULP) | Equivalent |
| 6 | Row OOB guard | `return` | `scf.if` | Equivalent |
| 7 | Column clamp | `min(jStart+256, n)` | `arith.minsi` | Equivalent |
| 8 | Scale application | multiply by rsqrt | multiply by rsqrt | Equivalent |
| 9 | bf16 conversion | `static_cast` | `arith.extf/truncf` | Equivalent |
| 10 | D pointer constness | `const float*` | `read_write` | Minor imprecision, not a bug |
| A | `nD > 64` guard | Host throws | Absent | **Missing guard — potential bug** |
| B | Large-matrix offset | `int64_t` | `i32` cast | **Overflow risk for large matrices** |
