---
title: "Faster Online Randomized SVD for LLM KV-Cache Compression"
authors:
  - key: zhihao
  - key: chichih
  - key: mohamed
tags:
  - llm
  - kv-cache
  - svd
  - inference
venue: none
year: 2025
date: 2025-03-10
stub: false
materials:
  - name: Paper
    url: https://github.com/bairixie/kv-svd
    type: file-pdf
  - name: Code
    url: https://github.com/bairixie/kv-svd
    type: code
---

# TL;DR

Replacing FP32 Householder QR with 16-bit matrix multiplications and Cholesky QR in randomized SVD achieves a **4.1× speedup** over `torch.svd_lowrank` for online KV-Cache compression, with negligible accuracy loss on the RULER benchmark.

# Introduction

Large Language Models are increasingly deployed at long context lengths — hundreds of thousands of tokens — creating a severe memory bottleneck. During autoregressive generation, the attention mechanism caches every previously computed Key and Value (KV) state. This **KV-Cache** grows as $\mathcal{O}(L \cdot d \cdot N_\text{layers})$, where $L$ is the sequence length and $d$ the per-head hidden dimension. For a 32-layer model with head dimension 512 at 128k-token context in 16-bit precision, the KV-Cache alone requires tens of gigabytes — often comparable to the model weights themselves.

**SVD-based compression** addresses this directly. KV-Cache matrices empirically exhibit rapid singular value decay: most information concentrates in a small number of dominant directions. Truncating to the top-$k$ singular components yields the best possible rank-$k$ approximation, guaranteed optimal by the Eckart-Young theorem.

[//]: **xKV** pushes this further by noting that adjacent transformer layers share nearly identical dominant subspaces. A single shared SVD over a group of $G$ concatenated KV-Caches achieves up to 6.8× higher compression than per-layer methods.

**xKV** [Chang et al., 2025] pushes this further by observing that the dominant singular vectors of KV-Caches are well-aligned *across* adjacent layers. Concatenating the KV-Caches of $G$ adjacent layers and applying one shared SVD extracts a common low-rank subspace for all layers jointly — achieving up to **6.8× higher compression** than prior inter-layer methods while improving accuracy.

The catch: unlike offline weight compression, xKV must compute SVD **online** during the prefill phase of every request, since the KV-Cache is input-dependent. This online SVD step becomes a significant and growing fraction of prefill latency. Even the standard approximate `torch.svd_lowrank` (Halko et al. [2011]) accounts for **13.0%** of total per-sample profiling time — a measurable throughput bottleneck with two clear inefficiencies.

We address both.

<figure>
<img src="imgs/blog/svd_blog/Figure_1_SVD_Time_Proportion.png" alt="SVD Overhead" width="700"/>
<figcaption>Fig. 1 — SVD overhead as a share of total profiling time per sample. Our method reduces SVD's share from 13.0% to 3.6%, a level where it is no longer a dominant cost.</figcaption>
</figure>

## Contributions

Our implementation is publicly available at [github.com/bairixie/kv-svd](https://github.com/bairixie/kv-svd), evaluated within the [xKV framework](https://github.com/abdelfattah-lab/xKV) on an NVIDIA RTX A6000.

We follow the same four-stage structure as `torch.svd_lowrank` with two targeted changes:

* **16-bit power iteration.** Casting matrix multiplications in Stages 1–3 to 16-bit halves memory traffic and enables full Tensor Core utilization. Matrix-multiply time in the power iteration drops from $91.5\text{ s} \to 22.5\text{ s}$ (**4.1×**).

* **Numerically robust Cholesky QR.** We replace Householder QR with Cholesky QR, incorporating Gram matrix symmetrization, adaptive shift regularization [Fukaya et al., 2020], an eigh-based SPD-repair fallback [Yamazaki et al., 2015], and a Householder safety net. Orthogonalization time drops from $222.6\text{ s} \to 37.8\text{ s}$ (**5.9×**).

> **Combined:** total SVD CUDA time $392.0\text{ s} \to 96.7\text{ s}$ — a **4.1× overall speedup** — and SVD's share of per-sample profiling time falls from **13.0% → 3.6%**. Averaged over four RULER subtasks, accuracy (**67.36%**) matches the `torch.svd_lowrank` baseline (67.24%) within 0.12 percentage points.

---

# Background

## Singular Value Decomposition

For any real matrix $A \in \mathbb{R}^{m \times n}$, the **Singular Value Decomposition** (SVD) is [Lee, IBM]:

$$A = U \Sigma V^\top$$

where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal matrices, and $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$, $r = \min(m, n)$.

Truncating to the top-$k$ components gives the **rank-$k$ approximation** $A_k = U_k \Sigma_k V_k^\top$. The **Eckart-Young theorem** guarantees it is optimal:

$$\|A - A_k\|_F = \min_{\text{rank}(B) \leq k} \|A - B\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots}$$

No other rank-$k$ matrix achieves smaller error — this is the theoretical foundation for SVD-based compression throughout machine learning.

## Why KV-Caches Are Compressible

Real-world matrices in deep learning exhibit **rapid spectral decay**: $\sigma_1 \gg \sigma_2 \gg \cdots \gg \sigma_r$. When this holds, a rank-$k$ approximation with $k \ll r$ captures nearly all variance. For LLM KV-Caches, xKV [Chang et al., 2025] demonstrates that 95% of cumulative variance requires only a small fraction of the full rank — a fraction that decreases further when adjacent layers' caches are concatenated, because those layers share nearly identical dominant subspaces.

## Setup & Notation

* Sequence length at prefill: $L$; per-head hidden dimension: $d$; number of layers grouped: $G$.
* KV-Cache for layer $\ell$: $X_\ell \in \mathbb{R}^{L \times d}$.
* **xKV concatenated cache:** $\bigl[X_{\ell_1}, \ldots, X_{\ell_G}\bigr] \in \mathbb{R}^{L \times (Gd)}$ — the matrix $A$ that our SVD operates on.
* Target rank: $k$; oversampling: $p$; power iteration steps: $n_\text{iter}$.

---

# SVD for KV-Cache Compression

## Per-Layer SVD Baseline

The simplest approach computes $X_\ell \approx U_k \Sigma_k V_k^\top$ independently for each layer, storing the compressed pair $(U_k\Sigma_k) \in \mathbb{R}^{L \times k}$ and $V_k^\top \in \mathbb{R}^{k \times d}$ at compression ratio $d/k$. This degrades at high compression (e.g., $8\times$) because approximation errors compound independently across layers.

## Cross-Layer SVD: xKV

xKV [Chang et al., 2025] identifies that adjacent transformer layers share **highly aligned dominant singular vectors**, as quantified by Centered Kernel Alignment (CKA). Exploiting this, xKV concatenates $G$ adjacent KV-Caches horizontally and applies one SVD to extract a **shared low-rank basis** $A \in \mathbb{R}^{L \times r}$ alongside layer-specific reconstruction matrices $B_{\ell_i} \in \mathbb{R}^{r \times d}$. With $G = 4$, xKV achieves near-baseline accuracy at $8\times$ compression on Llama-3.1-8B — a level inaccessible to per-layer methods. All our experiments use $G = 4$.

## Why Online SVD Is Expensive

Both per-layer and cross-layer SVD must be computed **online** during prefill, since the KV-Cache is input-dependent. Full SVD of an $m \times n$ matrix costs $\mathcal{O}(mn^2)$ flops. On our RTX A6000:

* `torch.linalg.svd` (full SVD) accounts for **73.4%** of per-sample profiling time — not viable for production.
* `torch.svd_lowrank` accounts for **13.0%** — a measurable bottleneck we address directly.

---

# Randomized SVD and Its Limitations

## Why Full SVD Is Wasteful

For KV-Cache compression we only need the top-$k$ singular vectors. Full SVD computes all $\min(m, n)$ components — often $10\times$ to $100\times$ more than necessary. Randomized SVD [Halko et al., 2011] reduces the dominant cost from $\mathcal{O}(mn^2)$ to $\mathcal{O}(mnk)$ by identifying the $k$-dimensional dominant subspace directly.

## Four-Stage Algorithm

Both `torch.svd_lowrank` and our method follow Algorithms 4.4 and 5.1 of Halko et al. [2011]:

**Stage 1 — Setup.** Transpose if $m < n$ so all stages operate on a tall matrix. Resolve working dtype; allocate $\mathbf{I}_{k+p}$.

```python
if m < n:  A, M = A.T, M.T
X    = cast(A, working_dtype)
eye_q = identity(k + p, dtype=working_dtype)
```

**Stage 2 — Random Projection.** Draw $R \in \mathbb{R}^{n \times (k+p)}$ and form sketch $Y = (A - M)R$. Orthonormalize to initial basis $Q$.

```python
R = randn(n, k + p, dtype=working_dtype)
Y = (A - M) @ R
Q = orthonormalize(Y)
```

**Stage 3 — Power Iteration.** Alternate $A^\top$ and $A$ to sharpen $Q$:

```python
for _ in range(n_iter):
    Q = orthonormalize((A - M).T @ Q)
    Q = orthonormalize((A - M)  @ Q)
```

Each iteration amplifies eigenvalue ratios by $(\sigma_i / \sigma_j)^{2n_\text{iter}}$, rapidly concentrating $Q$ on the dominant subspace. With $n_\text{iter} = 4$, this stage accounts for **62–80%** of total SVD time depending on implementation.

**Stage 4 — Project and Recover.**

```python
B       = Q.T @ (A - M)                    # shape: (k+p) × n
U_, S, Vt = svd(B.float(), full=False)     # FP32: torch.linalg.svd rejects 16-bit input
U       = Q @ U_
# truncate to top-k; undo transpose if needed
```

Total cost is dominated by the $(2n_\text{iter} + 1)$ multiplications with $A$, each $\mathcal{O}(mn(k+p))$ — a factor of $n/(k+p)$ cheaper than full SVD.

## Limitations of `torch.svd_lowrank`

**1. FP32 throughout — no Tensor Core utilization.** All matrix multiplications in Stages 1–3 run in FP32. Modern NVIDIA GPUs (Ampere, Hopper) deliver substantially higher throughput for 16-bit operations via Tensor Cores. In our profiling, the matrix-multiply sub-cost of the power iteration alone is **91.5 s**.

**2. Householder QR is the orthogonalization bottleneck.** Each `orthonormalize(·)` call invokes `torch.linalg.qr`. While backward-stable, Householder QR's sequential panel factorizations expose limited parallelism for tall-and-skinny shapes ($m \gg k+p$). The QR sub-cost in the power iteration is **222.6 s** — 56.9% of the total 392.0 s baseline SVD time.

---

# Our Method

## Overview

Our method is structurally identical to `torch.svd_lowrank`. We introduce exactly two modifications: (1) **16-bit computation** for all large matrix operations, and (2) **Cholesky QR** for orthogonalization. The design principle is to maximize 16-bit coverage for bandwidth-bound operations while performing a surgical FP32 upgrade only where precision is non-negotiable.

| Stage | Operation | `torch.svd_lowrank` | Ours (16-bit path) |
|-------|-----------|---------------------|--------------------|
| 1. Setup | Cast input, $\mathbf{I}_{k+p}$ | FP32 | **16-bit** |
| 2. Random Projection | $Y = AR$, orthogonalize | FP32 · Householder QR | **16-bit matmul · Cholesky QR** |
| 3. Power Iteration | $A^\top Q$, $AQ$, orth. | FP32 · Householder QR | **16-bit matmuls · Cholesky QR** |
| 4a. Projection | $B = Q^\top(A{-}M)$ | FP32 | **16-bit** |
| **4b. Small SVD** | $\text{svd}(B)$ | FP32 | **FP32** (PyTorch constraint) |
| 4c. Lift & truncate | $U = Q\hat{U}$ | FP32 | **16-bit** |

Two design choices deserve emphasis. `chol_qr` is **16-bit-in / 16-bit-out** with an internal FP32 upgrade: it receives a 16-bit matrix, immediately upcasts to FP32 for Gram matrix computation and Cholesky factorization (where numerical stability matters), then returns $Q$ in 16-bit. Inter-stage memory traffic stays in 16-bit; the factorization runs in FP32.

Stage 4b **must** remain FP32 because `torch.linalg.svd` raises a runtime error on 16-bit input — a hard PyTorch constraint, not a precision choice. Fortunately $B$ has shape $(k+p) \times n$ (e.g., $4 \times 512$ with $p = 0$), making this cost negligible.

## Optimization 1: 16-bit Power Iteration

The power iteration consists of repeated large matrix multiplications:

$$Q \;\leftarrow\; \text{orth}(A^\top Q), \qquad Q \;\leftarrow\; \text{orth}(A\,Q)$$

where $A \in \mathbb{R}^{L \times (Gd)}$ is the grouped KV-Cache. Three properties make this ideal for precision reduction:

* **Memory-bandwidth bound.** The dominant cost is reading $A$ from GPU HBM. Reducing element size from 32-bit to 16-bit directly halves memory traffic.

* **Approximation-tolerant.** The power iteration estimates a subspace, not an exact result. 16-bit rounding errors are equivalent to a small perturbation of the input — precisely the regime that randomized SVD handles robustly [Halko et al., 2011]. Subsequent iterations further suppress single-step errors.

* **Not the final computation.** Stage 3 produces only an intermediate orthonormal basis $Q$. Singular values are computed in Stage 4b in FP32.

On RTX A6000, switching from FP32 to 16-bit reduces the matrix-multiply sub-cost from **91.5 s → 22.5 s (4.1×)**, consistent with expected gains from Tensor Core utilization and halved memory bandwidth. Our implementation supports both IEEE float16 and bfloat16; both yield essentially identical task accuracy and performance on these workloads.

## Optimization 2: Numerically Robust Cholesky QR

Each `orthonormalize(Y)` call takes $Y \in \mathbb{R}^{m \times (k+p)}$ with $m \gg k+p$. All internal computation is FP32; the result is cast back to 16-bit on return.

### Basic Cholesky QR

Cholesky QR [Fukaya et al., 2014] exploits the algebraic identity: if $Y = QR$ then $Y^\top Y = R^\top R$. The $R$ factor is simultaneously the Cholesky factor of the Gram matrix $G = Y^\top Y$:

```python
G = Y_f32.T @ Y_f32        # (k+p)×(k+p) — one SYRK call
R = chol(G, upper=True)    # small Cholesky factor
Q = Y @ R_inv              # triangular solve (TRSM)
```

Compared to Householder QR, Cholesky QR requires roughly **half the total flop count** for tall-skinny matrices [Fukaya et al., 2014]. SYRK and TRSM are Level-3 BLAS routines achieving near-peak GPU throughput; Householder QR's sequential panel updates expose far less parallelism for small $k+p$.

### Gram Matrix Symmetrization

Before factorizing, we explicitly symmetrize $G$:

$$G \;\leftarrow\; 0.5\,(G + G^\top)$$

Floating-point rounding in $Y^\top Y$ accumulates small off-diagonal asymmetries. Explicit symmetrization eliminates this drift before it reaches `cholesky_ex`, reducing spurious factorization failures.

### Adaptive Shift Regularization

Following the shifted Cholesky QR framework of Fukaya et al. [2020], we add a scale-invariant diagonal regularization:

$$G_\text{shifted} = G + \varepsilon \cdot \text{scale} \cdot I, \quad \text{scale} = \text{mean}(\text{diag}(G)).\text{clamp}(\min=10^{-12})$$

We use `torch.linalg.cholesky_ex` — which returns an integer `info` tensor rather than raising an exception — for batch-aware failure detection with exponential backoff:

```python
eps = base_eps
for attempt in range(max_tries):
    R, info = cholesky_ex(G + eps * scale * I, upper=True)
    if all(info == 0):
        Q = solve_triangular(R, Y_f32, upper=True, left=False)
        return cast(Q, dtype_16bit)
    eps = min(eps * 10, max_eps)
```

In the common case (well-conditioned $Y$), the first attempt succeeds and the shift is numerically negligible. The exponential backoff handles progressively more ill-conditioned matrices without manual tuning.

### Eigh SPD-Repair Fallback

If all shifted Cholesky attempts fail, we reconstruct a strictly positive definite approximation of $G$ and Cholesky-factorize that [Yamazaki et al., 2015]:

```python
L, V  = eigh(G)
L     = clamp(L, min=max(1e-4, eps))
G_spd = V @ diag(L) @ V.T
R     = cholesky(G_spd, upper=True)
Q     = solve_triangular(R, Y_f32, upper=True, left=False)
return cast(Q, dtype_16bit)
```

The reconstruct-then-Cholesky design keeps the triangular solve well-conditioned: $R$'s diagonal entries are bounded away from zero by construction, avoiding amplification of clamped eigenvalue errors.

### Householder QR as Final Safety Net

If the eigh path raises any exception, we fall back to standard Householder QR:

```python
Q, _ = torch.linalg.qr(Y_f32, mode="reduced")
return cast(Q, dtype_16bit)
```

This recovers exactly the behavior of `torch.svd_lowrank`, making our implementation **strictly more robust than the baseline** — it can never perform worse. In practice, this path is almost never triggered; it exists as a correctness guarantee.

**Algorithm 1: `chol_qr(Y_16bit)` — Complete Three-Tier Strategy**

```
Input:  Y ∈ ℝ^{m×(k+p)}  in 16-bit
Output: Q ∈ ℝ^{m×(k+p)}  orthonormal columns, in 16-bit

 1.  Y_f32 ← cast(Y, float32)
 2.  G     ← Y_f32ᵀ Y_f32
 3.  G     ← 0.5 · (G + Gᵀ)                        [symmetrize]
 4.  scale ← mean(diag(G)).clamp(min=1e-12)

     // Tier 1: shifted Cholesky QR
 5.  ε ← base_eps
 6.  for attempt = 1 … max_tries:
 7.      R, info ← cholesky_ex(G + ε·scale·I, upper=True)
 8.      if all(info == 0):
 9.          Q ← solve_triangular(R, Y_f32, upper=True, left=False)
10.          return cast(Q, 16-bit)
11.      ε ← min(ε · 10, max_eps)

     // Tier 2: eigh SPD-repair
12.  try:
13.      L, V   ← eigh(G)
14.      L      ← clamp(L, min=max(1e-4, ε))
15.      G_spd  ← V · diag(L) · Vᵀ
16.      R      ← cholesky(G_spd, upper=True)
17.      Q      ← solve_triangular(R, Y_f32, upper=True, left=False)
18.      return cast(Q, 16-bit)
19.  except: pass

     // Tier 3: Householder QR safety net
20.  Q, _ ← torch.linalg.qr(Y_f32, mode="reduced")
21.  return cast(Q, 16-bit)
```

### Example: Unified SVD Interface

Our repo provides a single entry point to call all SVD methods — ours, full SVD, and `torch.svd_lowrank` — for easy benchmarking and drop-in use:

```python
def run_svd(A, k, method='ours', oversample=4, dtype=torch.bfloat16):
    """
    Unified SVD interface.

    Args:
        A        : input matrix (torch.Tensor)
        k        : target rank
        method   : 'ours'          — bf16/fp16 + Cholesky QR (default)
                   'torch_lowrank' — torch.svd_lowrank (fp32 + Householder QR)
                   'full'          — torch.linalg.svd (fp32, exact)
        oversample: extra dimensions for randomized methods (default 4)
        dtype    : 16-bit dtype for 'ours' (torch.bfloat16 or torch.float16)
    Returns:
        U, S, Vh  (top-k components)
    """
    if method == 'ours':
        return randomized_svd_chol(A, k, oversample=oversample, low_dtype=dtype)
    elif method == 'torch_lowrank':
        U, S, V = torch.svd_lowrank(A, q=k)
        return U[:, :k], S[:k], V[:, :k].T
    elif method == 'full':
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        return U[:, :k], S[:k], Vh[:k, :]
    else:
        raise ValueError(f"Unknown method: {method}")
```

---

# Experiments

## Setup

**Hardware.** Single NVIDIA RTX A6000 GPU. All timing figures report self CUDA time via the PyTorch profiler, collected during the profiling (prefill) phase only — steady-state decode-phase evaluation is left for future work.

**Benchmark.** We evaluate within the xKV framework [Chang et al., 2025] at [github.com/abdelfattah-lab/xKV](https://github.com/abdelfattah-lab/xKV). Accuracy is measured on four RULER subtasks [Hsieh et al., 2024]: *Frequent Word Extraction* (FWE), *NIAH MultiKey*, *NIAH Single1*, and *Variable Tracking* (VT). RULER is a long-context evaluation suite designed to stress KV-Cache compression artifacts by requiring retrieval and reasoning at various positions within a long context.

**Configuration.** Layer group size $G = 4$, $n_\text{iter} = 4$ power iteration steps, oversampling $p = 4$. Full SVD was profiled at approximately 10 samples due to OOM at 96 samples; other methods ran for 96 samples.

**Methods compared:**

* `torch.linalg.svd` — full SVD, FP32 (memory-limited reference only)
* `torch.svd_lowrank` — randomized SVD, FP32, Householder QR
* **Ours** — fp16 · Cholesky-QR (this work)

## End-to-End SVD Latency

<figure>
<img src="imgs/blog/svd_blog/Figure_1_SVD_Time_Proportion.png" alt="SVD Overhead per Sample" width="700"/>
<figcaption>Fig. 1 — Per-sample CUDA time decomposed into SVD (dark) and other inference tasks (grey). Our method reduces SVD from 13.0% to 3.6% of total per-sample time.</figcaption>
</figure>

**Takeaways.**

* **Full SVD is not viable:** it consumes 73.4% of profiling time per sample and causes OOM at 96 samples.
* **`torch.svd_lowrank` is still a bottleneck:** 13.0% SVD overhead (4.1 s/sample) limits throughput.
* **Ours drops SVD to 3.6%:** per-sample SVD time falls from 4.1 s → 1.0 s (**4.1×**), a level where SVD is no longer a dominant cost.

| Method | Total / sample | SVD / sample | SVD % |
|--------|---------------|-------------|-------|
| Full SVD (`torch.linalg.svd`, fp32) | 54.2 s | 39.8 s | 73.4% |
| `torch.svd_lowrank` (fp32 · Householder QR) | 31.5 s | 4.1 s | 13.0% |
| **Ours (fp16 · Cholesky-QR)** | 28.4 s | **1.0 s** | **3.6%** |

## Stage-Level Breakdown

<figure>
<img src="imgs/blog/svd_blog/Figure_3_CUDA_Time_by_Stage.png" alt="CUDA Time by Stage" width="700"/>
<figcaption>Fig. 2 — Randomized SVD CUDA time by stage: torch.svd_lowrank vs. Ours (fp16 · Cholesky-QR). RTX A6000 · n_iter=4 · layer group size 4.</figcaption>
</figure>

| Stage | fp32 · Householder QR | fp16 · Cholesky-QR (ours) | Speedup |
|-------|-----------------------|--------------------------|---------|
| 1. Setup (dtype cast / alloc) | $0.017\text{ s}$ $(0.0\%)$ | $3.60\text{ s}$ $(3.7\%)$ | — |
| 2. Random Projection | $39.3\text{ s}$ $(10.0\%)$ | $10.8\text{ s}$ $(11.2\%)$ | $3.6\times$ |
| 3. Power Iteration (×4) | $314.1\text{ s}$ $(80.1\%)$ | $60.3\text{ s}$ $(62.4\%)$ | $5.2\times$ |
| &nbsp;&nbsp;— Matrix Multiply | $91.5\text{ s}$ | $22.5\text{ s}$ | $4.1\times$ |
| &nbsp;&nbsp;— Orthogonalization | $222.6\text{ s}$ | $37.8\text{ s}$ | $5.9\times$ |
| 4. Project & Recover | $38.6\text{ s}$ $(9.9\%)$ | $21.9\text{ s}$ $(22.7\%)$ | $1.8\times$ |
| **Total** | $\mathbf{392.0\text{ s}}$ | $\mathbf{96.7\text{ s}}$ | $\mathbf{4.1\times}$ |

**Takeaways.**

* **Stage 1** adds 3.60 s of one-time dtype cast overhead, fully amortized by Stage 3 savings.
* **Stage 3** is the primary bottleneck and primary gain. Power iteration drops from 314.1 s (80.1%) to 60.3 s (62.4%), a **5.2× speedup** from two independent sources: **matrix multiply** improves **4.1×** from Tensor Core utilization; **orthogonalization** improves **5.9×** from Cholesky QR replacing Householder QR — a particularly large gain because Householder QR's sequential panel structure is ill-suited to tall-and-skinny shapes with small $k+p$.
* **Stage 4** shows a modest **1.8× gain** from the 16-bit projection $B = Q^\top(A-M)$. Its share of total time grows from 9.9% to 22.7% — not because it became slower, but because Stage 3 shrank so dramatically.

## Accuracy vs. Speed Trade-off

<figure>
<img src="imgs/blog/svd_blog/Figure_2_SVD_Accuracy_Comparison_VT.png" alt="Accuracy Comparison" width="700"/>
<figcaption>Fig. 3 — Average accuracy over four RULER subtasks (FWE · NIAH MultiKey · NIAH Single1 · VT). RTX A6000 · n_iter=4 · G=4 · oversampling p=4.</figcaption>
</figure>

| Method | FWE | NIAH MultiKey | NIAH Single1 | VT | **Average** |
|--------|-----|---------------|-------------|-----|-------------|
| Full SVD (`torch.linalg.svd`) | 74.0% | 58.3% | 97.9% | 41.5% | 67.92% |
| `torch.svd_lowrank` (baseline) | 75.0% | 55.2% | 99.0% | 39.8% | 67.24% |
| **Ours (fp16 · Cholesky-QR)** | **74.7%** | **58.3%** | **95.8%** | **40.6%** | **67.36%** |

**Takeaways.**

* Averaged over all four RULER subtasks, our method (**67.36%**) matches the `torch.svd_lowrank` baseline (67.24%) within **0.12 percentage points** — negligible accuracy cost for **4.1× lower SVD latency**.
* On two of four subtasks — NIAH MultiKey and VT — our method *outperforms* the baseline, reflecting that oversampling $p=4$ improves the quality of the random subspace estimate [Halko et al., 2011].
* A modest gap on NIAH Single1 (95.8% vs. 99.0%) likely reflects the slightly lower orthogonality of Cholesky QR for well-conditioned inputs [Fukaya et al., 2014] and minor 16-bit rounding in the power iteration.

---

# Limitations and Future Work

**Profiling phase only.** All measurements cover the prefill phase. Decoding-phase evaluation — where the compressed KV-Cache drives generation — is a critical next step.

**Single model and context length.** We use Meta-Llama-3.1-8B-Instruct at a fixed context length. Broader evaluation across models (Qwen2.5, DeepSeek) and context lengths would provide a more complete picture.

**Fixed rank and group size.** A fixed rank $k$ and $G = 4$ are used throughout. Adaptive rank allocation — assigning more budget to layers or tasks more sensitive to compression — is a promising direction for recovering the NIAH Single1 gap.

**Automation.** Future work can explore automating precision casting and kernel generation via `torch.compile` and template-based approaches, potentially enabling auto-tuned precision schedules beyond the fixed bf16/fp16 choice.

---

# Citing

```bibtex
@misc{abdelfattah2026svd_blog,
      title={Faster Online Randomized SVD for LLM KV-Cache Compression},
      author={Zhihao Mo and Chi-Chih Chang and Mohamed Abdelfattah},
      year={2026},
      url={https://abdelfattah-lab.github.io/blog/svd_blog},
}
```

---

# References

1. **Halko, N., Martinsson, P. G., & Tropp, J. A.** (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217–288. https://doi.org/10.1137/090771806

2. **Fukaya, T., Nakatsukasa, Y., Yanagisawa, Y., & Yamamoto, Y.** (2014). CholeskyQR2: A simple and communication-avoiding algorithm for computing a tall-skinny QR factorization on a large-scale parallel system. *ScalA 2014*, IEEE, pp. 31–38. https://doi.org/10.1109/ScalA.2014.11

3. **Fukaya, T., Kannan, R., Nakatsukasa, Y., Yamamoto, Y., & Yanagisawa, Y.** (2020). Shifted Cholesky QR for computing the QR factorization of ill-conditioned matrices. *SIAM Journal on Scientific Computing*, 42(1), A477–A503. https://doi.org/10.1137/18M1218212

4. **Yamazaki, I., Tomov, S., & Dongarra, J.** (2015). Mixed-precision Cholesky QR factorization and its case studies on multicore CPU with multiple GPUs. *SIAM Journal on Scientific Computing*, 37(3), C307–C330. https://doi.org/10.1137/14M0973773

5. **Chang, C.-C., Lin, C.-Y., Akhauri, Y., Lin, W.-C., Wu, K.-C., Ceze, L., & Abdelfattah, M. S.** (2025). xKV: Cross-layer SVD for KV-cache compression. *arXiv:2503.18893*.

6. **Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., & Ginsburg, B.** (2024). RULER: What's the real context size of your long-context language models? *arXiv:2404.06654*.

7. **Lee, F.** What is singular value decomposition (SVD)? IBM Think. https://www.ibm.com/think/topics/singular-value-decomposition