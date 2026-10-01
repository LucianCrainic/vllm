# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration shared by the CPU KV allocator and attention scheduler."""

from dataclasses import dataclass

from vllm import envs
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)


@dataclass(frozen=True)
class CPUTieredKVConfig:
    hbm_node: int
    ddr_node: int
    hbm_blocks: int
    target_hbm_fraction: float
    hbm_attention_threads: int


def get_cpu_tiered_kv_config(
    kv_cache_config: KVCacheConfig,
) -> CPUTieredKVConfig | None:
    """Resolve the opt-in single-socket HBM/DDR KV layout.

    The first implementation supports one uniform full-attention cache. A
    block ID has the same tier across all layers, so attention can recover its
    placement from the block table without migrating pages.
    """
    hbm_gib = envs.VLLM_CPU_HBM_KV_CACHE_GIB
    if hbm_gib == 0:
        return None
    if hbm_gib < 0:
        raise ValueError("VLLM_CPU_HBM_KV_CACHE_GIB must be positive")

    hbm_node = envs.VLLM_CPU_HBM_NODE
    ddr_node = envs.VLLM_CPU_DDR_NODE
    if hbm_node < 0 or ddr_node < 0 or hbm_node == ddr_node:
        raise ValueError("Set distinct VLLM_CPU_HBM_NODE and VLLM_CPU_DDR_NODE")

    if (
        len(kv_cache_config.kv_cache_groups) != 1
        or len(kv_cache_config.kv_cache_tensors) != 1
    ):
        raise ValueError("Tiered CPU KV requires one uniform attention cache group")
    group = kv_cache_config.kv_cache_groups[0]
    spec = group.kv_cache_spec
    specs = list(
        spec.kv_cache_specs.values()
        if isinstance(spec, UniformTypeKVCacheSpecs)
        else (spec,)
    )
    if not all(isinstance(layer_spec, FullAttentionSpec) for layer_spec in specs):
        raise ValueError("Tiered CPU KV currently supports full attention only")

    tensor = kv_cache_config.kv_cache_tensors[0]
    num_blocks = kv_cache_config.num_blocks
    if (
        tensor.offset != 0
        or tensor.size != len(tensor.layers) * num_blocks * tensor.block_stride
        or len(tensor.layers) != len(group.layer_names)
        or any(
            layer_spec.page_size_bytes != tensor.block_stride for layer_spec in specs
        )
    ):
        raise ValueError("Tiered CPU KV requires a dense layer-outermost layout")
    bytes_per_block = len(tensor.layers) * tensor.block_stride
    hbm_blocks = min(num_blocks, hbm_gib * GiB_bytes // bytes_per_block)
    if not 1 < hbm_blocks < num_blocks:
        raise ValueError(
            "Tiered CPU KV requires at least two HBM blocks and DDR capacity"
        )

    fraction = envs.VLLM_CPU_TIERED_KV_HBM_FRACTION
    if not 0 < fraction < 1:
        raise ValueError("VLLM_CPU_TIERED_KV_HBM_FRACTION must be in (0, 1)")
    hbm_threads = envs.VLLM_CPU_TIERED_ATTN_HBM_THREADS
    if hbm_threads < 0:
        raise ValueError("VLLM_CPU_TIERED_ATTN_HBM_THREADS must be nonnegative")
    return CPUTieredKVConfig(
        hbm_node=hbm_node,
        ddr_node=ddr_node,
        hbm_blocks=hbm_blocks,
        target_hbm_fraction=fraction,
        hbm_attention_threads=hbm_threads,
    )
