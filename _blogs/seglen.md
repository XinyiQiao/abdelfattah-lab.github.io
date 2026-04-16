---
title: "Prefix Caching for Hybrid LLMs: From Marconi to SegLen in SGLang"
authors:
  - key: isabella
tags:
  - llm
  - software
venue: none
year: 2026
date: 2026-04-14
stub: false
materials: []
---

# Prefix Caching Optimization for Hybrid LLMs

## Introduction

LLM applications are increasingly using longer context windows during serving. Techniques such as few-shot prompting, chain-of-thought reasoning, retrieval-augmented generation, and structured prompt templates often require providing substantial context to the model. As context length growth, it also significantly increases the cost of the **prefill** phase in autoregressive inference.

Prefix caching has therefore become one of the most important serving optimiztions widely adopted by modern serving systems such as vLLM and SGLang, which enables reuse of cached states across requests that share common prefixes. When two requests share a prefix, the system can skip recomputation for the shared tokens and directly reuse previously computed states. Existing prefix caching mechanisms are primarily designed for **dense attention-based architectures**, where key-value (KV) cache entries are stored per token and can be partially reused.

Recent shift in model architecture towards hybrid model has introduces new challenges for prefix caching. Hybrid models interleave attention layers with state-based recurrent layers and these two layer types exhibit fundamentally different caching behavior. These differences introduce new challenges for efficient cross-reqest prefix caching.

 <div style="text-align:center;">
    <img src="/imgs/blog/seglen/hybridmodel.png" width="40%" />
  </div>  

In this work, we study prefix caching optimization for hybrid models. Inspired by Marconi’s core insight of FLOP-aware eviction policy, we analyze the evaluate its comp of integrating it into SGLang and propose a simpler heuristic, **SegLen**, that approximates recomputation cost while remaining model architecture-agnostic. We implement SegLen in SGLang and evaluate it across multiple workloads, showing that it achieves superior performance while significantly reducing integration complexity into real serving system.

## Problem

Prefix caching is well understood for dense attention models, where KV cache entries are stored per token and can be partially reused. However, hybrid architectures introduce **recurrent states** (e.g., SSM states) that behave very differently from token-level KV caches. These differences fundamentally change the reuse pattern and make cache eviction more challenging.

### 2.1 Observed Properties: Recurrent State vs KV Cache

Recurrent state exhibit following properties

- Recurrent states are constant-sized regardless of the number of tokens it represent

- Recurrent states are orders of magnitude larger than the KVs of a single token

- Recurrent states are updated in-place, so a request's states cannot be rolled back to represent its prefixes

 <div style="text-align:center;">
    <img src="/imgs/blog/seglen/recurrentstate.png" width="60%" />
  </div>  

These properties lead to fundamentally different reuse behavior compared to attention-only models and make the prefix cache complicated.


### 2.2 Core Problem: Sparse Reuse with Exact Prefix Matching
To realize prefix reuse, we need one recurrent state that excatly matrches all prefix tokens. As a result, recurrent state exhibit "all or nothing" reusability.

### 2.3 Systems Challenge: Cache underutilization and High memory usage 

To maximize reuse opportunities, fine-grained state checkpoints if required. However, increased checkpointing frequency inflates the number of cache entries generated per sequence, each of which is large and can quickly overwhelme the limited caceh budget. Worse, many recurrent states are never reused creating sparesely-hit entries and cache underutilization. The results in **large yet sparsely-hit cache entries**. As a result, we need a better cache management mechnaism for prefix caching in Hybrid LLM inference.

## 3. Marconi

### Key Insight: Cost Aware Cache Eviction

Marconi is a prior work done on prefix caching for hybrid model. It is motivated by the property that for attention layers, KV cache size grows linearly with sequence length, and so do compute savings from reusing it. So larger KV cache entries save proportionally more compute. However, a recurrent state is constant-size regardless of sequence length and thus the reuse value it delivers. 

Marconi proposes a **FLOP-aware cache eviction policy** for hybrid model architectures. The key idea is that eviction decisions should not rely solely on recency (as in LRU), but also consider the compute savings of each cached state.

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/flopeff.png" width="40%" />
</div>  


### FLOP-Efficiency score
Marconi propose a new metric **FLOP efficiency** to measure the compute savings per unit of memory achieved by reusing a prefix cache entry:

$$
\text{FLOP-efficiency} = \frac{\text{total FLOPs across layers}}{\text{memory consumption of all states}}
$$

### FLOP-Aware Eviction Policy
Marconi proposes a flop aware evictino policy that assess candidates for eviction based on both potential compute savings and recency. Marconi proposes a utility score S that account for Flop efficiency

$$
s(n) = \mathrm{recency}(n) + \alpha \cdot \mathrm{flop\_efficiency}(n)
$$

## 4. Integrating Marconi into SGLang

We want to evaluate Marconi by integrating it into a real serving engine sglang. Let's first understand how SGLang implements prefix caching for hybrid models.

### 4.1 SGLang's Hybrid State Management

SGLang implements prefix caching using a radix tree data structure. Each tree node stores shared token prefix segement together with pointers to the cached states associated with that segment. For hybrid models, SGLang extends the radix tree to jointly manage KV cache and recurrent state.

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/mambaradixcache.png" width="50%" />
</div>  

SGLang separate the memory pool into two parts: Mamba pool and KV cache pool, each of with its own allocation and eviction logic. 

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/cachepool.png" width="50%" />
</div>  

### 4.2 Resource capacity

Another practical property of the system is that Mamba capacity is usually more constrained than full KV capacity. Recurrent states are fewer in number and expensive enough that the Mamba side of the cache becomes the real bottleneck. As a result, prefix reuse behavior is often determined less by whether full KV entries exist and more by whether the relevant recurrent checkpoints still fit in the available Mamba slots. In other words, Mamba pressure tends to drive eviction.

| Cache pool | Slots | Total size |
| --- | ---: | ---: |
| Mamba cache pool | 151 | 7.29 GB |
| KV cache pool | 263,865 token slots | 8.06 GB |


### 4.3 Cache Eviction Behavior
SGLang also treats KV eviction and recurrent-state eviction differently:

- For Full KV eviction, the eviction candidacte consider leaf nodes only. It will free KV cache and SSM state, as well as deleting the node from the tree. 
- For Mamba eviction, eviction candidate can target both leaf and internal nodes. If eviction is targeting an internal node, only the SSM state is removed but the internal node will remain in the tree and the KV cache will also be there. It becomes a tombstone node meaning an internal node with KV cache but lost its mamba state. 

### 4.4 Prefix Matching

To realize prefix reusing for hybrid model, we need all prefix tokens’ KVs for each Attention layer and one SSM state that exactly matches all prefix tokens for each SSM layer. Reuse can proceed only to the deepest ancestor that still has a valid Mamba state.

This is an important difference from dense-attention-only caching. In an attention-only model, reuse can be thought of as continuing from any cached token boundary. In a hybrid model, reuse is anchored at nodes that carry valid recurrent checkpoints. While the KV prefix may extend through tombstone nodes, reuse stops at the last node for which recurrent state is present. If the traversal reaches a node whose recurrent state has been evicted, then all tokens below that point must be replayed even if some tree structure remains.

## 4.2 Challenges Integrating Marconi
We integrated Marconi into SGLang for Qwen/Qwen3.5-9B.

First, tombstone nodes complicates the FLOP efficiency scoring calculation. Marconi flop efficiency scoring calculate the marginal flops saved by this node relative to its parent assuming the parent node is not tombstoned. But due sglang allows tombstone nodes, the exact replay cost of a node can be much longer to the valid ancestor that stores the mamba state.

Second, exact FLOPs calculation is model-specific. A production serving system like SGLang supports many models and variants, and a fully faithful FLOPs-saving estimate would require architecture-specific logic. That adds engineering overhead, makes integration less clean, and complicates maintenance.

So the question became: can we preserve the key idea from Marconi, namely recomputation-aware eviction, without depending on model-specific FLOPs accounting?

## 5. SegLen Heuristic

The presence of tombstone node and model architecture dependency complicated the exact calculation of marconi flop efficiency score. Flop efficiency aturally favors retaining SSM state that represent longer prefix, so we propose `seglen` which uses the reply distance to the nearest parent node that is not tombstoned as a heuristic approximation to Marconi's FLOPs-efficiency score. Seglen cache eviction policies combine this replay distance with recency for the eviction candidate scoring. 

This preserves the most important signal from marconi for eviction decisions while significantly reduced integration complexity into sglang. 

Here is an example illustrating the different eviction pick across the three policies. 

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/cacheeviction.png" width="50%" />
</div>

| Eviction policy | Ranking (best → worst to keep) | Eviction pick |
|---|---|---|
| LRU | B (mru) → C → D → A (lru) | A |
| SegLen | A (seglen=1) → B (seglen=3) → C (seglen=4) → D (seglen=4+2=6) | D |
| Marconi | A (efficiency=0.12) → B (efficiency=0.31) → D (efficiency=0.45) → C (efficiency=0.67) | C |

When eviction is needed, LRU blindly picks A — the coldest node regardless of recomputation cost. SegLen picks D — its replay distance is the longest (4 tokens through tombstoned C + 2 of its own = 6). Marconi picks C — its 4-token segment yields the highest FLOP efficiency score.

The implementation of SegLen can be found [here](https://github.com/sgl-project/sglang/pull/22172).


## 6. Experiments

### 6.1 Setup

We evaluate `seglen` against `lru` on 1xH100 using SGLang on `Qwen/Qwen3.5-9B`. We studied how different cache policies behavr across different workloads and available cache memories. 

### 6.2 Across Workloads

We benchmark across three different workload categories:
  
- prefix-heavy dataset: a synthetic workload mixing various prefix groups with random noise

- SWE-bench traces: SWE-bench traces that capture realistic workload and re-use pattern

- ShareGPT with low prefix reuse**: a regression check in the low-reuse regime

Across all evaluated workloads, seglen reduced mean TTFT by 29.52% on prefix-heavy dataset, 26.06% on SWE-bench traces. On the ShareGPT regression benchmark, seglen remained slightly better, reducing mean TTFT by 3.38%.

The plot below summarizes mean TTFT across workloads. The y-axis uses a log scale so both the low-latency and high-latency regimes remain visible in one figure.

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/benchmark_ttft_subplots.png" width="70%" />
</div>

| Dataset | Prompts | LRU TTFT(ms) | SegLen TTFT(ms) | TTFT Dec. | LRU Hit Rate | SegLen Hit Rate | LRU Queue Depth | SegLen Queue Depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mixed_prefix_eviction_dataset_7k_10k_with_noise.json` | 1800 | 1369.94 | 1256.68 | 8.27% | 94.84 | 94.82 | 3.55 | 3.39 |
| `multi_group_shared_prefix_dataset.json` | 2760 | 253.20 | 178.46 | 29.52% | 85.95 | 85.80 | 0.07 | 0.02 |
| `ShareGPT_V3_unfiltered_cleaned_split.json` | 2000 | 83.31 | 80.49 | 3.38% | 0.31 | 1.36 | 0.00 | 0.00 |
| `swebench_sps=10_art=5_nums=100.jsonl` | 1200 | 40789.14 | 34292.94 | 15.93% | 21.23 | 25.94 | 144.87 | 124.89 |
| `swebench_sps=10_art=10_nums=100.jsonl` | 1200 | 41011.02 | 30324.60 | 26.06% | 19.87 | 26.99 | 147.36 | 110.19 |

Script for benchmark reproduction: 

- [benchmark_cache_eviction_multi.py](https://raw.githubusercontent.com/abdelfattah-lab/sglang/refs/heads/benchmark/benchmark_cache_eviction_multi.py)
- [benchmark_cache_eviction_trace_multi.py](https://raw.githubusercontent.com/abdelfattah-lab/sglang/refs/heads/benchmark/benchmark_cache_eviction_trace_multi.py)

bench serving changes to support swe-bench traces:
- [bench_serving](https://github.com/abdelfattah-lab/sglang/tree/benchmark)

Download datasets: https://huggingface.co/datasets/Isabella5/sglang-seglen-benchmark  

### Across cache memories

We also sweep available memory fractions on a H100 to evaluate the policies under increasing cache pressure.

The main trend is clear: as memory pressure increases, the advantage of `seglen` becomes more obvious.

This matches the systems intuition behind the policy. When memory is plentiful, the cache can retain many reusable checkpoints, so the exact eviction choice matters less. But as memory becomes tighter, each eviction decision has a larger downstream effect on replay cost. In that regime, a policy that is better aligned with the true replay boundary has a larger payoff.

This memory-sweep result is important because it shows that `seglen` is not just a better policy in the easy regime. Its advantage grows precisely when cache management becomes harder and more important.

The plots below show the memory-fraction sweep on H100 for `lru` and `seglen`. As the cache budget tightens, `seglen` improves TTFT and reduces queue depth more clearly. At the highest tested budget (`0.72`), `lru` slightly outperforms `seglen` on TTFT and hit rate, while `seglen` still maintains the lower queue depth.


## Conclusion

Hybrid LLMs make prefix caching fundamentally harder. Recurrent state is constant-size but large, cannot be rolled back like dense KV cache, and is only reusable at exact checkpoints. Fine-grained checkpointing increases reuse opportunities, but it also produces a cache full of large, sparsely-hit entries. In that setting, eviction policy matters a lot.

Marconi provides the right high-level answer: eviction should be aware of recomputation cost, not just recency. But integrating exact FLOP-aware scoring into a production system like SGLang is challenging because of tombstone nodes and the need for model-specific FLOPs estimation.

`seglen` is a practical middle ground. It approximates the same intuition using replay distance to the nearest reusable ancestor, which is cheap to compute, architecture-agnostic, and naturally compatible with SGLang's `MambaRadixCache` design.

The result is a simple systems heuristic that is easy to integrate and delivers strong TTFT improvements in practice.
