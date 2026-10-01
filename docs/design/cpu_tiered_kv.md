# Experimental single-socket CPU HBM and DDR KV cache

This opt-in path targets a CPU socket with two local NUMA memory nodes: one HBM
and one DDR. Bind the inference threads to cores on that socket. Keep both
memory nodes available to the process. The CPU worker binds model loading to
HBM, then allocates each layer's KV cache with an HBM block prefix and a DDR
block suffix using Linux `mbind`. A block retains its tier for its lifetime;
there is no page migration in the attention path.

The block pool assigns new blocks in short tier stripes and aims for a chosen
HBM fraction among allocated blocks. During causal decode, the attention
scheduler reads the block table, creates work for contiguous runs in each tier,
and gives HBM and DDR work to separate OpenMP thread groups. It merges partial
outputs with the existing split-KV softmax reduction. Prefill, sliding-window
attention, and batches without traffic in both tiers use the regular CPU
scheduler.

## Configuration

- `VLLM_CPU_HBM_NODE` and `VLLM_CPU_DDR_NODE`: distinct local NUMA node IDs.
- `VLLM_CPU_HBM_KV_CACHE_GIB`: HBM capacity reserved for KV, in GiB. The
  configured total `--kv-cache-memory-bytes` must exceed this budget.
- `VLLM_CPU_TIERED_KV_HBM_FRACTION`: target fraction of allocated blocks in
  HBM; default `0.67`.
- `VLLM_CPU_TIERED_ATTN_HBM_THREADS`: number of attention threads initially
  assigned to HBM; `0` estimates a split from the current decode batch.
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

The first implementation uses active block counts as a placement proxy and
scans block tables each decode step. Fragmented prefix-cache blocks can create
more tier runs and reduction work; measure this case separately.
