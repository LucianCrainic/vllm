# Experimental single-socket CPU HBM and DDR KV cache

This opt-in path targets a CPU socket with two local NUMA memory nodes: one HBM
and one DDR. Bind the inference threads to cores on that socket. Keep both
memory nodes available to the process. The CPU worker binds model loading to
HBM, then allocates each layer's KV cache with an HBM block prefix and a DDR
block suffix using Linux `mbind`. A block retains its tier for its lifetime;
there is no page migration in the attention path.

The block pool fills HBM first. Near its capacity limit, it uses smoothed
per-tier attention completion times to choose the tier for new blocks. Until
both tiers have produced enough measurements, it keeps allocating in HBM.
During causal decode, the attention
scheduler reads the block table, creates work for contiguous runs in each tier,
and gives HBM and DDR work to separate OpenMP thread groups. It merges partial
outputs with the existing split-KV softmax reduction. Prefill, sliding-window
attention, and batches without traffic in both tiers use the regular CPU
scheduler.

## Configuration

- `VLLM_CPU_HBM_NODE` and `VLLM_CPU_DDR_NODE`: distinct local NUMA node IDs.
- `VLLM_CPU_HBM_KV_CACHE_GIB`: HBM capacity reserved for KV, in GiB. It may
  cover the entire configured `--kv-cache-memory-bytes` pool.
- `VLLM_CPU_TIERED_ATTN_HBM_THREADS`: number of attention threads initially
  assigned to HBM; `0` adapts the split from measured tier completion times.
- `VLLM_CPU_TIERED_ATTN_SCHEDULE=0`: disable tier-aware scheduling while
  retaining upstream split-KV for a clean ablation.
- `VLLM_CPU_OMP_THREADS_BIND`: explicitly list cores on the selected socket.
- `VLLM_CPU_ATTN_SPLIT_KV=1`: enables concurrent tier scheduling during decode.

The feature is disabled when `VLLM_CPU_HBM_KV_CACHE_GIB=0`. It currently
requires one uniform full-attention cache group and a dense layer-outermost
KV layout. It is intended for a single tensor-parallel rank. The allocator
fails when either memory node is unavailable or when the HBM budget cannot
leave space for DDR blocks. Leave enough free HBM for weights, activations,
and runtime allocations.

## Measurement

First compare serial and simultaneous local HBM and DDR reads with the same
total core count and byte traffic. Next establish a single-socket vLLM
baseline with the same model, image, request mix, and core binding. Then run
the tiered path while varying the HBM KV budget, block fraction, and thread
split. Record decode tokens/s, latency by request length, actual NUMA page
residency, memory bandwidth per tier, and output agreement with the standard
CPU attention path. A useful balance minimizes the slower of the two tier
completion times, including AMX compute and softmax reduction costs.

The worker returns tier token counts, completion times, and thread split to
the scheduler after each dual-tier decode step. Debug logs include these
measurements plus actual HBM/DDR residency, the smoothed target, and the last
allocation tier. New blocks respond to feedback; active blocks do not move.
Fragmented prefix-cache blocks can create more tier runs and reduction work;
measure this case separately.

## Observed tradeoffs

On the Llama-3.1-8B-Instruct 4096-input/1024-output benchmark, HBM-first
tiering had no observed c48 penalty against all-HBM KV (196.4 versus 198.6 ms
mean time per output token, three repetitions each). At c80, HBM-first tiering
measured 304.2 ms versus 306.4 ms for the earlier fixed-fraction allocator
(two repetitions each). This small difference is within run variation. Putting
all KV in DDR measured 385.8 ms. Moving weights to DDR to fit the full KV pool
in HBM measured 321.6 ms, so the delivered implementation keeps weights in HBM.
Disabling tier-aware attention scheduling measured 303.5 ms; this result does
not establish a gain from adapting the thread split. The scheduler switch
remains available for continued measurement.
