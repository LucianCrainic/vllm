#ifndef VLLM_NUMA_DISABLED
  #include <numa.h>
  #include <numaif.h>
  #include <sys/mman.h>
  #include <unistd.h>
  #include <string>
  #include <sched.h>
#endif
#include <ATen/ops/from_blob.h>
#include <cerrno>
#include <cstring>
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
  #include <unistd.h>
  #include <sys/syscall.h>
  #define gettid() syscall(SYS_gettid)
#endif

#include "cpu/utils.hpp"

#ifdef VLLM_NUMA_DISABLED
void init_cpu_memory_env(std::vector<int64_t> node_ids) {}
#else
void init_cpu_memory_env(std::vector<int64_t> node_ids) {
  // Memory node binding
  if (numa_available() != -1) {
    // Concatenate all node_ids into a single comma-separated string
    if (!node_ids.empty()) {
      std::string node_ids_str;
      for (const int node_id : node_ids) {
        if (!node_ids_str.empty()) {
          node_ids_str += ",";
        }
        node_ids_str += std::to_string(node_id);
      }

      bitmask* mask = numa_parse_nodestring(node_ids_str.c_str());
      bitmask* src_mask = numa_get_mems_allowed();

      int pid = getpid();

      if (mask && src_mask) {
        // move all existing pages to the specified numa node.
        *(src_mask->maskp) = *(src_mask->maskp) ^ *(mask->maskp);
        int page_num = numa_migrate_pages(pid, src_mask, mask);
        if (page_num == -1) {
          TORCH_WARN("numa_migrate_pages failed. errno: " +
                     std::to_string(errno));
        }

        // Restrict memory allocation to the selected NUMA node(s).
        // Enhances memory locality for the threads bound to those NUMA CPUs.
        if (node_ids.size() > 1) {
          errno = 0;
          numa_set_interleave_mask(mask);
          if (errno != 0) {
            TORCH_WARN("numa_set_interleave_mask failed. errno: " +
                       std::to_string(errno));
          } else {
            TORCH_WARN(
                "NUMA binding: Using INTERLEAVE policy for memory "
                "allocation across multiple NUMA nodes (nodes: " +
                node_ids_str +
                "). Memory allocations will be "
                "interleaved across the specified NUMA nodes.");
          }
        } else {
          errno = 0;
          numa_set_membind(mask);
          if (errno != 0) {
            TORCH_WARN("numa_set_membind failed. errno: " +
                       std::to_string(errno));
          } else {
            TORCH_WARN(
                "NUMA binding: Using MEMBIND policy for memory "
                "allocation on the NUMA nodes (" +
                node_ids_str +
                "). Memory allocations will be "
                "strictly bound to these NUMA nodes.");
          }
        }

        numa_set_strict(1);

        numa_free_nodemask(mask);
        numa_free_nodemask(src_mask);
      } else {
        TORCH_WARN(
            "numa_parse_nodestring or numa_get_run_node_mask failed. errno: " +
            std::to_string(errno));
      }
    }
  }
}
#endif  // VLLM_NUMA_DISABLED

torch::Tensor allocate_tiered_cpu_kv_cache(int64_t bytes, int64_t num_layers,
                                           int64_t num_blocks,
                                           int64_t hbm_blocks,
                                           int64_t block_stride,
                                           int64_t hbm_node, int64_t ddr_node) {
#ifdef VLLM_NUMA_DISABLED
  TORCH_CHECK(false, "Tiered CPU KV cache requires Linux NUMA support");
#else
  const int64_t page_size = sysconf(_SC_PAGESIZE);
  TORCH_CHECK(bytes > 0 && num_layers > 0 && num_blocks > 0);
  TORCH_CHECK(hbm_blocks > 0 && hbm_blocks <= num_blocks);
  TORCH_CHECK(block_stride > 0 && block_stride % page_size == 0);
  TORCH_CHECK(bytes == num_layers * num_blocks * block_stride);
  TORCH_CHECK(hbm_node >= 0 && ddr_node >= 0 && hbm_node != ddr_node);
  TORCH_CHECK(numa_available() >= 0);

  bitmask* allowed = numa_get_mems_allowed();
  TORCH_CHECK(allowed != nullptr);
  const bool nodes_allowed = hbm_node < static_cast<int64_t>(allowed->size) &&
                             ddr_node < static_cast<int64_t>(allowed->size) &&
                             numa_bitmask_isbitset(allowed, hbm_node) &&
                             numa_bitmask_isbitset(allowed, ddr_node);
  numa_free_nodemask(allowed);
  TORCH_CHECK(nodes_allowed, "Both tiered KV nodes must be allowed");

  void* data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  TORCH_CHECK(data != MAP_FAILED, "KV mmap failed: ", std::strerror(errno));

  auto bind_range = [&](int64_t offset, int64_t length, int64_t node) {
    if (!length) return;
    bitmask* mask = numa_allocate_nodemask();
    TORCH_CHECK(mask != nullptr);
    numa_bitmask_setbit(mask, node);
    const int result = mbind(static_cast<char*>(data) + offset, length,
                             MPOL_BIND, mask->maskp, mask->size, 0);
    const int saved_errno = errno;
    numa_free_nodemask(mask);
    TORCH_CHECK(result == 0, "KV mbind to node ", node,
                " failed: ", std::strerror(saved_errno));
  };

  const int64_t layer_bytes = num_blocks * block_stride;
  const int64_t hbm_bytes = hbm_blocks * block_stride;
  try {
    for (int64_t layer = 0; layer < num_layers; ++layer) {
      bind_range(layer * layer_bytes, hbm_bytes, hbm_node);
      bind_range(layer * layer_bytes + hbm_bytes, layer_bytes - hbm_bytes,
                 ddr_node);
    }
  } catch (...) {
    munmap(data, bytes);
    throw;
  }

  return at::from_blob(
      data, {bytes}, [bytes](void* ptr) { munmap(ptr, bytes); },
      at::TensorOptions().dtype(at::kChar).device(at::kCPU));
#endif
}

namespace cpu_utils {
ScratchPadManager::ScratchPadManager() : size_(0), ptr_(nullptr) {
  this->realloc(allocation_unit * 128);
}

void ScratchPadManager::realloc(size_t new_size) {
  new_size = round(new_size);
  if (new_size > size_) {
    void* new_ptr = std::aligned_alloc(64, new_size);
    TORCH_CHECK(new_ptr != nullptr,
                "ScratchPadManager: aligned_alloc failed for size ", new_size);
    if (ptr_ != nullptr) {
      std::free(ptr_);
    }
    ptr_ = new_ptr;
    size_ = new_size;
  }
}

ScratchPadManager* ScratchPadManager::get_scratchpad_manager() {
  static ScratchPadManager manager;
  return &manager;
}
}  // namespace cpu_utils

void compute_slot_mapping_kernel_impl(const torch::Tensor query_start_loc,
                                      const torch::Tensor positions,
                                      const torch::Tensor block_table,
                                      torch::Tensor slot_mapping,
                                      const int64_t block_size) {
  const int32_t req_num = query_start_loc.size(0) - 1;
  const int64_t block_table_stride = block_table.stride(0);

  const int32_t* __restrict__ query_start_loc_ptr =
      query_start_loc.data_ptr<int32_t>();
  const int64_t* __restrict__ positions_ptr = positions.data_ptr<int64_t>();
  const int32_t* __restrict__ blocktable_ptr = block_table.data_ptr<int32_t>();
  int64_t* __restrict__ slot_mapping_ptr = slot_mapping.data_ptr<int64_t>();

#pragma omp parallel for
  for (int32_t req_idx = 0; req_idx < req_num; ++req_idx) {
    int32_t token_start_idx = query_start_loc_ptr[req_idx];
    int32_t token_end_idx = query_start_loc_ptr[req_idx + 1];
    int32_t token_num = token_end_idx - token_start_idx;
    const int64_t* __restrict__ curr_position_ptr =
        positions_ptr + token_start_idx;
    int64_t* __restrict__ curr_slot_mapping_ptr =
        slot_mapping_ptr + token_start_idx;
    const int32_t* __restrict__ curr_block_table_ptr =
        blocktable_ptr + req_idx * block_table_stride;

    for (int32_t token_idx = 0; token_idx < token_num; ++token_idx) {
      int64_t token_position = curr_position_ptr[token_idx];
      int64_t block_id = curr_block_table_ptr[token_position / block_size];
      curr_slot_mapping_ptr[token_idx] =
          block_id * block_size + token_position % block_size;
    }
  }
}
