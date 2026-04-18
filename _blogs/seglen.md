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

Modern LLM applications are pushing toward longer context windows. Few-shot prompting, chain-of-thought reasoning, and retrieval-augmented generation all rely on feeding more context into the model. 

As context length grows, the cost of **prefill** phase quickly adds up, making inference significantly more expensive. 

Prefix caching has therefore become a core optimization in serving systems like vLLM and SGLang. The idea is simple: if two requests share the same prefix, we reuse previously computed states instead of recomputing them. 

For attention-based models, this works really well. KV cache is stored per token, so systems can reuse partial prefixes. Even if two requests only overlap halfway, you still get meaningful savings.

But Hybrid models make this much less straightforward. Modern model architectures are increasingly hybrid - they mix attention layers with recurrent components (like state space models) to reduce compute and memory cost.

 <div style="text-align:center;">
    <img src="/imgs/blog/seglen/hybridmodel.png" width="40%" />
  </div>  

| Property | Attention | SSM |
| --- | --- | --- |
| Computational Complexity | \(O(L^2)\) | \(O(L)\) |
| Inference-Time Memory | \(O(L)\) | \(O(1)\) |

These recurrent states behave very differently from KV caches. Instead of storing information per token, they compress the entire prefix into a single fixed-size state and update it in place, wich means you can't partially reuse a prefix. 

So the question now becomes:

> **How should prefix caching work for hybrid models?**

In this work, we take the FLOP-aware cache eviction idea from Marconi—a recent approach to recomputation-aware caching for hybrid models—and try to bring it into a real serving system SGLang. In doing so, we run into a number of practical considerations, which led us to propose a simpler heuristic approach: SegLen.

SegLen is a lightweight, model-agnostic heuristic that captures the core intuition behind Marconi, while being much easier to integrate into a real serving engine. We implement SegLen in SGLang and evaluate it across a range of workloads and cache settings. Our results show that SegLen delivers strong performance while significantly simplifying system integration.

## Problem

Prefix caching is well understood for attention-only models. But once we introduce recurrent components, the behavior of the cache changes in important ways.

### 2.1 Recurrent State vs KV Cache

Let's start with the key differences.

KV cache (attention):
- Stored per token 
- Grows linearly with sequence length
- Supports partial reuse

Recurrent state (SSM / Mamba):
- Fixed size regardless of sequence length
- Much larger than a single token’s KV
- Updated in-place

 <div style="text-align:center;">
    <img src="/imgs/blog/seglen/recurrentstate.png" width="60%" />
  </div>  

Because recurrent states are updated in place, you can’t roll them back to represent earlier prefixes.

### 2.2 Core Problem: All-or-Nothing Reuse
With KV cache, you can reuse any partial prefix. 

With recurrent state, reuse becomes **all-or-nothing**.

To reuse a recurrent state, you need a checkpoint that exactly matches the full prefix of the new request. 

In practice, this makes reuse much more sparse and much harder to exploit.

### 2.3 Systems Challenge: Big, Sparse, and Expensive Cache

To maximize reuse opportunities, one natural idea is to store more checkpoints so we have a better chance of hitting an exact prefix match.

But this quickly creates a new problem.
- Each recurrent state is large
- More checkpoints means more memory usage
- But most checkpoints are still rarely reused

As a result, the cache grows quickly, but the hit rate doesn’t improve much. You end up with a cache that is large, expensive, and sparsely utilized.

So the real challenge becomes:

> **how do we decide which cache entries are actually worth keeping?**


## 3. Marconi: Rethinking Cache Eviction

Marconi proposed a new prefix caching strategy for hybrid model. 

It starts from a simple observation. 

In attention layers, KV cache grows with sequence length. That means longer prefixes take more memory and also save more compute when reused. 

Recurrent states behave very differently. They are constant size regardless of sequence length. But the amount of compute they can save depends on how many tokens they represent. So two cache entries can take the same memory - but have very different reuse value.

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/flopeff.png" width="40%" />
</div>  

### Key Insight: Flop Aware Cache Eviction

Most systems today rely on simple cache eviction policies like LRU (least recently used), which evict entries based on recency. 

But recency alone doesn’t tell us how valuable an entry is.

Marconi proposes a different approach: make eviction decisions based on compute savings.

Each cache entry is scored based on:
- how much compute (FLOPs) it can save if reused
- how much memory it consumes

This leads to the notion of FLOP efficiency — how much compute you save per unit of memory:

$$
\text{FLOP-efficiency} = \frac{\text{total FLOPs across layers}}{\text{memory consumption of all states}}
$$

To combine this with recency, Marconi defines a utility score:

$$
s(n) = \mathrm{recency}(n) + \alpha \cdot \mathrm{flop\_efficiency}(n)
$$

This metric favors cache entries with higher recency, save more compute, and take less memory.

## 4. Integrating Marconi into SGLang

We want to bring Marconi into a real serving system by integrating it into SGLang. Before diving into the integration, it’s worth understanding how SGLang handles prefix caching for hybrid models. 

### 4.1 SGLang's Hybrid Cache Management

SGLang implements prefix caching using a radix tree.

Each node in the tree represents a shared prefix segment, and stores:
- the tokens for that segment
- pointers to the cached states

For hybrid models, SGLang extends this structure to manage both KV cache and recurrent state together.

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/mambaradixcache.png" width="50%" />
</div>  

Under the hood, memory is split into two separate pools:
- KV cache pool (attention)
- Mamba pool (recurrent state)
Each pool has its own allocation and eviction logic.

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/cachepool.png" width="50%" />
</div>  

### 4.2 Resource capacity

Another practical aspect is that Mamba pool is usually much more resource constrained than the KV Cache pool. Recurrent states are much larger in size than KV for a single token, which means the Mamba pool have much fewer slots, faces more eviction pressure and drives prefix reuse.

| Cache pool | Slots | Total size |
| --- | ---: | ---: |
| Mamba cache pool | 151 | 7.29 GB |
| KV cache pool | 263,865 token slots | 8.06 GB |

| Cache Pool | Slots | Total size |
|---|---:|---:|
| Mamba | 424 | 20.39 GB |
| KV | 740752 | 22.60 GB |


### 4.3 Cache Eviction Behavior
SGLang also treats KV cache and recurrent state very differently during eviction.

KV eviction
- Only applies to leaf nodes
- Removes both KV cache and recurrent state
- Deletes the node entirely

Mamba eviction
- Can target both leaf and internal nodes
- Removes only the recurrent state
- Leaves the internal node and its KV Cache intact 

This creates **tombstone nodes** - internal nodes that still have KV cache, but no recurrent state.

### 4.4 Prefix Matching

For hybrid models, reuse requires:
- KV cache for all prefix tokens
- and a recurrent state that exactly matches the prefix

Reuse can only proceed up to the deepest node that still has a valid recurrent state. Even KV prefix may extend through tombstone nodes, reuse stops once you hit a tombstone node. Everything beyond that point has to be recomputed.

## 4.2 Challenges Integrating Marconi
When we integrate Marconi into SGLang (using Qwen/Qwen3.5-9B), a few practical issues showed up.

First, tombstone nodes complicate the scoring logic.

Marconi’s FLOP-efficiency assumes parent nodes are valid. But in SGLang, that’s not always true due to tombstone nodes. Recompute distance may extend far beyond the immediate parent.

Second, FLOP estimation is model-specific.

Accurately computing FLOPs requires architecture-specific logic, which tightly couples the eviction policy to model details and makes the system harder to maintain.

## 5. SegLen: A Simpler Heuristic

The core idea from Marconi is simple: states that represent longer prefixes are more valuable to keep.

So instead of computing FLOPS exactly, we looked for a simpler signal that captures the core idea. 

We proposed a heuristic `seglen` that uses the reply distance to the nearest parent node that still has a valid recurrent state as a heuristic approximation to Marconi's FLOPs-efficiency score. 

Seglen favors keeping entries that represent longer prefixes, which naturally save more recomputation and is therefore more valuable to keep. 

To make eviction decisions, `seglen` combines replay distance with recency — similar to Marconi, but without requiring model-specific FLOP estimation.

Here's a simple example illustrating how different eviction policies behave:

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/cacheeviction.png" width="50%" />
</div>

| Eviction policy | Ranking (best → worst to keep) | Eviction pick |
|---|---|---|
| LRU | B (mru) → C → D → A (lru) | A |
| Marconi | A (efficiency=0.12) → B (efficiency=0.31) → D (efficiency=0.45) → C (efficiency=0.67) | C |
| SegLen | A (seglen=1) → B (seglen=3) → C (seglen=4) → D (seglen=4+2=6) | D |

When eviction is needed
- LRU picks A — the least recently used node
- Marconi picks C — its 4-token segment yields the highest FLOP efficiency score
- SegLen picks D — its replay distance is the longest (4 tokens through tombstoned C + 2 of its own = 6). 

The implementation of SegLen can be found [here](https://github.com/sgl-project/sglang/pull/22172).

## 6. Experiments

### 6.1 Setup

We evaluate `segLen` against `marconi` and `lru` in SGLang using Qwen/Qwen3.5-9B model on a single H100 GPU.

Our goal is to understand how different cache policies behavr under different workloads and memory constraints.

### 6.2 Across Workloads

We evaluate across two types of workloads:
  
<!-- - prefix-heavy dataset: synthetic workload mixing various prefix groups with random noise (multi_group_shared_prefix_dataset.json) -->
- SWE-bench traces: realistic workloads with meaningful prefix reuse
- ShareGPT with low prefix reuse: a regression check in the low-reuse regime

Across these workloads, SegLen consistently reduces mean TTFT compared to LRU, achieving over 50% reduction on SWE-bench traces, while remaining competitive in low-reuse settings.

<!-- ### mixed output
| Policy | Mean TTFT (ms) | Mean queue depth | Mean cache hit rate |
|---|---:|---:|---:|
| lru | 2274.69 | 19.0400 | 0.6529 |
| seglen | 2233.17 | 18.8548 | 0.6538 |
| marconi | 2325.26 | 20.9328 | 0.6548 |
| marconi-v2 | 2901.55 | 28.3761 | 0.6685 | -->

#### SWE-Bench
On swe-bench traces where each prefix is reused ~5 tmies on average, seglen reduced TTFT by 51.3% compared to lru, while also improving cache hit rate and reducing queue depth.

| Policy | Mean TTFT (ms) | Mean queue depth | Mean cache hit rate |
|---|---:|---:|---:|
| lru | 2107.68 | 10.3686 | 0.3261 |
| seglen | 1027.35 | 4.8386 | 0.4179 |
| marconi | 1178.64 | 5.6679 | 0.4065 |
| marconi-v2 | 1521.77 | 7.7896 | 0.4182 |

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/swebench_art5_ttft.png" width="70%" />
</div>


On swe-bench trace where each prefix is reused ~10 times, seglen reduced TTFT by 51.5% compared to lru.

| Policy | Mean TTFT (ms) | Mean queue depth | Mean cache hit rate |
|---|---:|---:|---:|
| lru | 2273.05 | 11.2514 | 0.2994 |
| seglen | 1101.67 | 5.9165 | 0.4353 |
| marconi | 1335.84 | 7.2620 | 0.4242 |
| marconi-v2 | 1883.44 | 10.4380 | 0.4214 |

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/swebench_art10_ttft.png" width="70%" />
</div>

result on swebench_sps=10_art=10_nums=100.jsonl

#### ShareGPT

On ShareGPT dataset, where prefix reuse is minimal, SegLen remains competitive and achieves a 0.12% reduction in TTFT compared to lru.

| Policy | Mean TTFT (ms) | Mean queue depth | Mean cache hit rate |
|---|---:|---:|---:|
| lru | 72.48 | 0.0004 | 0.0022 |
| seglen | 72.39 | 0.0000 | 0.0049 |
| marconi | 75.29 | 0.0004 | 0.0049 |
| marconi-v2 | 113.57 | 0.0013 | 0.0051 |

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/share_gpt_ttft.png" width="70%" />
</div>

result on ShareGPT_V3_unfiltered_cleaned_split.json

These results show that with meaninful prefix reuse, SegLen delivers significant performance gains. When reuse is low, it still remains competitive.

### Across Memory Budgets

Next, we vary available cache memory to understand how policies behave under different levels of memory pressure. 

| Mem fraction static | Policy | Mean TTFT (ms) | Mean queue depth | Mean cache hit rate |
|---|---|---:|---:|---:|
| 0.77 | lru | 1697.04 | 8.4729 | 0.3317 |
| 0.77 | seglen | 934.93 | 4.6355 | 0.4226 |
| 0.77 | marconi | 1110.74 | 5.4124 | 0.4088 |
| 0.77 | marconi-v2 | 1499.00 | 8.0385 | 0.4218 |
| 0.7 | lru | 2809.16 | 14.3618 | 0.2912 |
| 0.7 | seglen | 1704.70 | 9.0218 | 0.3775 |
| 0.7 | marconi | 1992.97 | 10.1597 | 0.3835 |
| 0.7 | marconi-v2 | 2155.35 | 11.3566 | 0.3858 |
| 0.5 | lru | 13283.69 | 58.0286 | 0.1290 |
| 0.5 | seglen | 9706.63 | 44.9238 | 0.2249 |
| 0.5 | marconi | 10194.86 | 46.8752 | 0.2238 |
| 0.5 | marconi-v2 | 9547.27 | 44.6585 | 0.2347 |

<div style="text-align:center;">
    <img src="/imgs/blog/seglen/ttft_vs_memory_fraction.png" width="70%" />
</div>

Results on swebench_sps=10_art=5_nums=100.jsonl dataset

The trend is clear: as memory pressure increases, the advantage of `seglen` becomes more pronounced.

Reproducibility

Scripts: 
- [benchmark_cache_eviction_multi.py](https://raw.githubusercontent.com/abdelfattah-lab/sglang/refs/heads/benchmark/benchmark_cache_eviction_multi.py)
- [benchmark_cache_eviction_trace_multi.py](https://raw.githubusercontent.com/abdelfattah-lab/sglang/refs/heads/benchmark/benchmark_cache_eviction_trace_multi.py)

Bench-serving changes for swe-bench traces support:
- [bench_serving](https://github.com/abdelfattah-lab/sglang/tree/benchmark)

Datasets
- https://huggingface.co/datasets/Isabella5/sglang-seglen-benchmark  

## Conclusion

Prefix caching works well for attention-only models, but hybrid architectures introduce a new challenge: recurrent states make reuse all-or-nothing, leading to large, sparsely-used cache entries.

Marconi offers an important insight: cache eviction should be guided by recomputation cost, not just recency. Building on this idea, we propose SegLen, a simple heuristic that preserves this core intuition while being much easier to integrate into a real serving system like SGLang.

Across our experiments, SegLen achieves over 50% reduction in TTFT on realistic workloads, remains competitive in low-reuse settings, and shows even larger gains under memory pressure.

In the end, SegLen demonstrates that a simple, system-friendly approximation of recomputation cost is enough to make effective cache eviction decisions in real-world serving systems.
