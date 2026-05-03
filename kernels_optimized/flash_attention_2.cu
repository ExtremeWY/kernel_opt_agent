#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kHeadDim = 128;
constexpr int kThreadsPerWarp = 32;
constexpr float kLog2e = 1.4426950408889634f;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::BFloat16, #x " must be bfloat16")

__device__ inline int mma_accumulator_fragment_row(int lane_id, int frag_idx) {
  return (lane_id >> 2) + ((frag_idx & 2) ? 8 : 0);
}

__device__ inline int mma_accumulator_fragment_col(int lane_id, int frag_idx) {
  return ((lane_id & 3) * 2) + (frag_idx & 1) + ((frag_idx >= 4) ? 8 : 0);
}

__device__ inline uint32_t pack_bf16_pair(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

__device__ inline uint32_t shared_u32_addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ inline void ldmatrix_m8n8_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
      : "r"(addr));
}

__device__ inline void ldmatrix_m8n8_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
      : "r"(addr));
}

__device__ inline void cp_async_ca_16(uint32_t smem_addr, const void* gmem_addr) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
}

__device__ inline void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ inline void cp_async_wait_group_0() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ inline void mma_m16n8k16_bf16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ inline float fast_exp2_ftz(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

template <typename T>
__device__ inline T lane_group4_sum(T value) {
  value += __shfl_xor_sync(0xffffffff, value, 2, 4);
  value += __shfl_xor_sync(0xffffffff, value, 1, 4);
  return value;
}

template <typename T>
__device__ inline T lane_group4_max(T value) {
  value = max(value, __shfl_xor_sync(0xffffffff, value, 2, 4));
  value = max(value, __shfl_xor_sync(0xffffffff, value, 1, 4));
  return value;
}

template <int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int NUM_WARPS, bool CAUSAL>
__global__ void flash_attention_2_forward_regtc_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ o,
    int num_heads,
    int seq_len,
    float sm_scale_log2e) {
  static_assert(HEAD_DIM == 128, "Register Tensor Core kernel is specialized for head_dim=128");
  static_assert(BLOCK_KV == 16, "Register Tensor Core kernel currently computes one 16-column KV tile");
  static_assert(BLOCK_Q == NUM_WARPS * 16, "Each warp owns one 16-row query panel");
  constexpr int kSharedPitch = HEAD_DIM + 8;
  constexpr int kVecElemsPerAccess = 8;
  constexpr int kVecsPerRow = HEAD_DIM / kVecElemsPerAccess;
  constexpr int kValueTileN = 16;
  constexpr int kValueTiles = HEAD_DIM / kValueTileN;
  constexpr int kAccElemsPerMmaN16 = 8;
  constexpr int kOperandTileElems = BLOCK_KV * kSharedPitch;

  extern __shared__ __align__(32) unsigned char smem_raw[];
  auto* q_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  auto* operand_tile = q_tile;
  auto* k_tile0 = operand_tile;
  auto* v_tile0 = k_tile0 + kOperandTileElems;
  auto* k_tile1 = v_tile0 + kOperandTileElems;
  auto* v_tile1 = k_tile1 + kOperandTileElems;

  const int warp_id = threadIdx.x / kThreadsPerWarp;
  const int lane_id = threadIdx.x & (kThreadsPerWarp - 1);
  const int row_base = warp_id * 16;
  const int query_block_start = blockIdx.x * BLOCK_Q;
  const int block_last_query = min(query_block_start + BLOCK_Q, seq_len) - 1;
  const int head_idx = blockIdx.y;
  const int batch_idx = blockIdx.z;
  const int64_t bh_offset_elems =
      (static_cast<int64_t>(batch_idx) * num_heads + head_idx) * seq_len * HEAD_DIM;
  const uint4 zero_vec = make_uint4(0, 0, 0, 0);

  float acc_frag[kValueTiles][8];
#pragma unroll
  for (int value_tile = 0; value_tile < kValueTiles; ++value_tile) {
#pragma unroll
    for (int frag_idx = 0; frag_idx < 8; ++frag_idx) {
      acc_frag[value_tile][frag_idx] = 0.0f;
    }
  }

  float m_lo = -INFINITY;
  float m_hi = -INFINITY;
  float l_lo = 0.0f;
  float l_hi = 0.0f;

  for (int vec_linear = lane_id; vec_linear < 16 * kVecsPerRow; vec_linear += kThreadsPerWarp) {
    const int row = row_base + vec_linear / kVecsPerRow;
    const int vec_col = (vec_linear % kVecsPerRow) * kVecElemsPerAccess;
    const int q_idx = query_block_start + row;
    auto* q_dst = reinterpret_cast<uint4*>(q_tile + row * kSharedPitch + vec_col);
    if (q_idx < seq_len) {
      const auto* q_src = reinterpret_cast<const uint4*>(
          q + bh_offset_elems + static_cast<int64_t>(q_idx) * HEAD_DIM + vec_col);
      *q_dst = *q_src;
    } else {
      *q_dst = zero_vec;
    }
  }
  constexpr int kSharedPadElems = kSharedPitch - HEAD_DIM;
  if constexpr (kSharedPadElems > 0) {
    for (int row = threadIdx.x; row < BLOCK_Q; row += blockDim.x) {
      auto* q_pad = reinterpret_cast<uint4*>(q_tile + row * kSharedPitch + HEAD_DIM);
      *q_pad = zero_vec;
    }
  }
  __syncwarp();

  constexpr int kCachedQFrags = HEAD_DIM / 16;
  uint32_t q_frag_cache[kCachedQFrags][4];
#pragma unroll
  for (int k_step = 0; k_step < HEAD_DIM; k_step += 16) {
    const int a_addr_lane = lane_id & 7;
    const int a_addr_quad = lane_id >> 3;
    const int a_row = row_base + a_addr_lane + ((a_addr_quad & 1) ? 8 : 0);
    const int a_col = k_step + ((a_addr_quad >= 2) ? 8 : 0);
    ldmatrix_m8n8_x4(
        q_frag_cache[k_step / 16],
        shared_u32_addr(q_tile + a_row * kSharedPitch + a_col));
  }
  __syncthreads();

  for (int start_n = 0; start_n < seq_len; start_n += BLOCK_KV) {
    if constexpr (CAUSAL) {
      if (start_n > block_last_query) {
        break;
      }
    }

    const int tile_count = min(BLOCK_KV, seq_len - start_n);
    int active_rows = tile_count;
    if constexpr (CAUSAL) {
      active_rows = min(tile_count, block_last_query - start_n + 1);
    }

    const int tile_iter = start_n / BLOCK_KV;
    auto* k_tile = (tile_iter & 1) ? k_tile1 : k_tile0;
    auto* v_tile = (tile_iter & 1) ? v_tile1 : v_tile0;

    if (start_n == 0) {
      if (active_rows == BLOCK_KV) {
        for (int vec_linear = threadIdx.x; vec_linear < BLOCK_KV * kVecsPerRow; vec_linear += blockDim.x) {
          const int row = vec_linear / kVecsPerRow;
          const int vec_col = (vec_linear % kVecsPerRow) * kVecElemsPerAccess;
          const int key_idx = start_n + row;
          const auto* k_src = reinterpret_cast<const uint4*>(
              k + bh_offset_elems + static_cast<int64_t>(key_idx) * HEAD_DIM + vec_col);
          auto* k_dst = reinterpret_cast<uint4*>(k_tile + row * kSharedPitch + vec_col);
          cp_async_ca_16(shared_u32_addr(k_dst), k_src);
          const auto* v_src = reinterpret_cast<const uint4*>(
              v + bh_offset_elems + static_cast<int64_t>(key_idx) * HEAD_DIM + vec_col);
          auto* v_dst = reinterpret_cast<uint4*>(v_tile + row * kSharedPitch + vec_col);
          cp_async_ca_16(shared_u32_addr(v_dst), v_src);
        }
      } else {
        for (int vec_linear = threadIdx.x; vec_linear < BLOCK_KV * kVecsPerRow; vec_linear += blockDim.x) {
          const int row = vec_linear / kVecsPerRow;
          const int vec_col = (vec_linear % kVecsPerRow) * kVecElemsPerAccess;
          auto* k_dst = reinterpret_cast<uint4*>(k_tile + row * kSharedPitch + vec_col);
          auto* v_dst = reinterpret_cast<uint4*>(v_tile + row * kSharedPitch + vec_col);
          if (row < active_rows) {
            const int key_idx = start_n + row;
            const auto* k_src = reinterpret_cast<const uint4*>(
                k + bh_offset_elems + static_cast<int64_t>(key_idx) * HEAD_DIM + vec_col);
            cp_async_ca_16(shared_u32_addr(k_dst), k_src);
            const auto* v_src = reinterpret_cast<const uint4*>(
                v + bh_offset_elems + static_cast<int64_t>(key_idx) * HEAD_DIM + vec_col);
            cp_async_ca_16(shared_u32_addr(v_dst), v_src);
          } else {
            *k_dst = zero_vec;
            *v_dst = zero_vec;
          }
        }
      }
      if constexpr (kSharedPadElems > 0) {
        if (active_rows < BLOCK_KV) {
          for (int row = threadIdx.x; row < BLOCK_KV; row += blockDim.x) {
            auto* k_pad = reinterpret_cast<uint4*>(k_tile + row * kSharedPitch + HEAD_DIM);
            auto* v_pad = reinterpret_cast<uint4*>(v_tile + row * kSharedPitch + HEAD_DIM);
            *k_pad = zero_vec;
            *v_pad = zero_vec;
          }
        }
      }
      cp_async_commit_group();
      cp_async_wait_group_0();
    }
    if (start_n == 0) {
      __syncthreads();
    }

    const int next_start_n = start_n + BLOCK_KV;
    bool has_next_tile = next_start_n < seq_len;
    if constexpr (CAUSAL) {
      has_next_tile = has_next_tile && (next_start_n <= block_last_query);
    }
    if (has_next_tile) {
      const int next_tile_count = min(BLOCK_KV, seq_len - next_start_n);
      int next_active_rows = next_tile_count;
      if constexpr (CAUSAL) {
        next_active_rows = min(next_tile_count, block_last_query - next_start_n + 1);
      }
      auto* next_k_tile = (tile_iter & 1) ? k_tile0 : k_tile1;
      auto* next_v_tile = (tile_iter & 1) ? v_tile0 : v_tile1;
      if (next_active_rows == BLOCK_KV) {
        for (int vec_linear = threadIdx.x; vec_linear < BLOCK_KV * kVecsPerRow; vec_linear += blockDim.x) {
          const int row = vec_linear / kVecsPerRow;
          const int vec_col = (vec_linear % kVecsPerRow) * kVecElemsPerAccess;
          const int key_idx = next_start_n + row;
          const auto* k_src = reinterpret_cast<const uint4*>(
              k + bh_offset_elems + static_cast<int64_t>(key_idx) * HEAD_DIM + vec_col);
          auto* k_dst = reinterpret_cast<uint4*>(next_k_tile + row * kSharedPitch + vec_col);
          cp_async_ca_16(shared_u32_addr(k_dst), k_src);
          const auto* v_src = reinterpret_cast<const uint4*>(
              v + bh_offset_elems + static_cast<int64_t>(key_idx) * HEAD_DIM + vec_col);
          auto* v_dst = reinterpret_cast<uint4*>(next_v_tile + row * kSharedPitch + vec_col);
          cp_async_ca_16(shared_u32_addr(v_dst), v_src);
        }
      } else {
        for (int vec_linear = threadIdx.x; vec_linear < BLOCK_KV * kVecsPerRow; vec_linear += blockDim.x) {
          const int row = vec_linear / kVecsPerRow;
          const int vec_col = (vec_linear % kVecsPerRow) * kVecElemsPerAccess;
          auto* k_dst = reinterpret_cast<uint4*>(next_k_tile + row * kSharedPitch + vec_col);
          auto* v_dst = reinterpret_cast<uint4*>(next_v_tile + row * kSharedPitch + vec_col);
          if (row < next_active_rows) {
            const int key_idx = next_start_n + row;
            const auto* k_src = reinterpret_cast<const uint4*>(
                k + bh_offset_elems + static_cast<int64_t>(key_idx) * HEAD_DIM + vec_col);
            cp_async_ca_16(shared_u32_addr(k_dst), k_src);
            const auto* v_src = reinterpret_cast<const uint4*>(
                v + bh_offset_elems + static_cast<int64_t>(key_idx) * HEAD_DIM + vec_col);
            cp_async_ca_16(shared_u32_addr(v_dst), v_src);
          } else {
            *k_dst = zero_vec;
            *v_dst = zero_vec;
          }
        }
      }
      if constexpr (kSharedPadElems > 0) {
        if (next_active_rows < BLOCK_KV) {
          for (int row = threadIdx.x; row < BLOCK_KV; row += blockDim.x) {
            auto* k_pad = reinterpret_cast<uint4*>(next_k_tile + row * kSharedPitch + HEAD_DIM);
            auto* v_pad = reinterpret_cast<uint4*>(next_v_tile + row * kSharedPitch + HEAD_DIM);
            *k_pad = zero_vec;
            *v_pad = zero_vec;
          }
        }
      }
      cp_async_commit_group();
    }

    float score_half0[4];
    float score_half1[4];
#pragma unroll
    for (int frag_idx = 0; frag_idx < 4; ++frag_idx) {
      score_half0[frag_idx] = 0.0f;
      score_half1[frag_idx] = 0.0f;
    }
#pragma unroll
    for (int k_step = 0; k_step < HEAD_DIM; k_step += 16) {
      uint32_t k_regs[4];
      const int b_addr_lane = lane_id & 7;
      const int b_addr_phase = (lane_id >> 3) & 1;
      const int b_row = k_step + b_addr_phase * 8;
      const int b_col = b_addr_lane + ((lane_id >= 16) ? 8 : 0);
      ldmatrix_m8n8_x4(k_regs, shared_u32_addr(k_tile + b_col * kSharedPitch + b_row));
      mma_m16n8k16_bf16(score_half0, q_frag_cache[k_step / 16], k_regs);
      mma_m16n8k16_bf16(score_half1, q_frag_cache[k_step / 16], k_regs + 2);
    }

    float local_max_lo = -INFINITY;
    float local_max_hi = -INFINITY;
#pragma unroll
    for (int frag_idx = 0; frag_idx < 4; ++frag_idx) {
      const int local_row = mma_accumulator_fragment_row(lane_id, frag_idx);
      const int local_col = mma_accumulator_fragment_col(lane_id, frag_idx);
      const int global_row = query_block_start + row_base + local_row;
      const bool valid_key =
          global_row < seq_len && local_col < active_rows &&
          (!CAUSAL || start_n + local_col <= global_row);
      float score = valid_key ? score_half0[frag_idx] * sm_scale_log2e : -INFINITY;
      score_half0[frag_idx] = score;
      if (local_row < 8) {
        local_max_lo = fmaxf(local_max_lo, score);
      } else {
        local_max_hi = fmaxf(local_max_hi, score);
      }

      const int acc_idx = frag_idx + 4;
      const int local_row_hi = mma_accumulator_fragment_row(lane_id, acc_idx);
      const int local_col_hi = mma_accumulator_fragment_col(lane_id, acc_idx);
      const int global_row_hi = query_block_start + row_base + local_row_hi;
      const bool valid_key_hi =
          global_row_hi < seq_len && local_col_hi < active_rows &&
          (!CAUSAL || start_n + local_col_hi <= global_row_hi);
      float score_hi =
          valid_key_hi ? score_half1[frag_idx] * sm_scale_log2e : -INFINITY;
      score_half1[frag_idx] = score_hi;
      if (local_row_hi < 8) {
        local_max_lo = fmaxf(local_max_lo, score_hi);
      } else {
        local_max_hi = fmaxf(local_max_hi, score_hi);
      }
    }
    const float tile_m_lo = lane_group4_max(local_max_lo);
    const float tile_m_hi = lane_group4_max(local_max_hi);
    const float new_m_lo = fmaxf(m_lo, tile_m_lo);
    const float new_m_hi = fmaxf(m_hi, tile_m_hi);
    const float alpha_lo = isinf(m_lo) ? 0.0f : exp2f(m_lo - new_m_lo);
    const float alpha_hi = isinf(m_hi) ? 0.0f : exp2f(m_hi - new_m_hi);

    uint32_t p_regs[4];
    float local_sum_lo = 0.0f;
    float local_sum_hi = 0.0f;
    __nv_bfloat16 p_vals[kAccElemsPerMmaN16];
    const float p0 = fast_exp2_ftz(score_half0[0] - new_m_lo);
    const float p1 = fast_exp2_ftz(score_half0[1] - new_m_lo);
    const float p2 = fast_exp2_ftz(score_half0[2] - new_m_hi);
    const float p3 = fast_exp2_ftz(score_half0[3] - new_m_hi);
    const float p4 = fast_exp2_ftz(score_half1[0] - new_m_lo);
    const float p5 = fast_exp2_ftz(score_half1[1] - new_m_lo);
    const float p6 = fast_exp2_ftz(score_half1[2] - new_m_hi);
    const float p7 = fast_exp2_ftz(score_half1[3] - new_m_hi);
    p_vals[0] = __float2bfloat16(p0);
    p_vals[1] = __float2bfloat16(p1);
    p_vals[2] = __float2bfloat16(p2);
    p_vals[3] = __float2bfloat16(p3);
    p_vals[4] = __float2bfloat16(p4);
    p_vals[5] = __float2bfloat16(p5);
    p_vals[6] = __float2bfloat16(p6);
    p_vals[7] = __float2bfloat16(p7);
    local_sum_lo += (p0 + p1) + (p4 + p5);
    local_sum_hi += (p2 + p3) + (p6 + p7);
    p_regs[0] = pack_bf16_pair(p_vals[0], p_vals[1]);
    p_regs[1] = pack_bf16_pair(p_vals[2], p_vals[3]);
    p_regs[2] = pack_bf16_pair(p_vals[4], p_vals[5]);
    p_regs[3] = pack_bf16_pair(p_vals[6], p_vals[7]);
    const float new_l_lo = l_lo * alpha_lo + lane_group4_sum(local_sum_lo);
    const float new_l_hi = l_hi * alpha_hi + lane_group4_sum(local_sum_hi);

#pragma unroll
    for (int value_tile = 0; value_tile < kValueTiles; ++value_tile) {
      float o_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float o_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      uint32_t v_regs[4];
      const int v_addr_lane = lane_id & 7;
      const int v_addr_phase = (lane_id >> 3) & 1;
      const int v_row = v_addr_lane + v_addr_phase * 8;
      const int v_col = value_tile * kValueTileN + ((lane_id >= 16) ? 8 : 0);
      ldmatrix_m8n8_x4_trans(v_regs, shared_u32_addr(v_tile + v_row * kSharedPitch + v_col));
      mma_m16n8k16_bf16(o_lo, p_regs, v_regs);
      mma_m16n8k16_bf16(o_hi, p_regs, v_regs + 2);
#pragma unroll
      for (int frag_idx = 0; frag_idx < 4; ++frag_idx) {
        const int local_row = mma_accumulator_fragment_row(lane_id, frag_idx);
        const float alpha = local_row < 8 ? alpha_lo : alpha_hi;
        acc_frag[value_tile][frag_idx] = acc_frag[value_tile][frag_idx] * alpha + o_lo[frag_idx];
      }
#pragma unroll
      for (int frag_idx = 0; frag_idx < 4; ++frag_idx) {
        const int acc_idx = frag_idx + 4;
        const int local_row = mma_accumulator_fragment_row(lane_id, acc_idx);
        const float alpha = local_row < 8 ? alpha_lo : alpha_hi;
        acc_frag[value_tile][acc_idx] =
            acc_frag[value_tile][acc_idx] * alpha + o_hi[frag_idx];
      }
    }
    m_lo = new_m_lo;
    m_hi = new_m_hi;
    l_lo = new_l_lo;
    l_hi = new_l_hi;
    cp_async_wait_group_0();
    if (has_next_tile) {
      __syncthreads();
    }
  }

#pragma unroll
  for (int value_tile = 0; value_tile < kValueTiles; ++value_tile) {
    const float inv_l_lo = l_lo > 0.0f ? 1.0f / l_lo : 0.0f;
    const float inv_l_hi = l_hi > 0.0f ? 1.0f / l_hi : 0.0f;
    const uint32_t row_lo_cols0 = pack_bf16_pair(
        __float2bfloat16(acc_frag[value_tile][0] * inv_l_lo),
        __float2bfloat16(acc_frag[value_tile][1] * inv_l_lo));
    const uint32_t row_hi_cols0 = pack_bf16_pair(
        __float2bfloat16(acc_frag[value_tile][2] * inv_l_hi),
        __float2bfloat16(acc_frag[value_tile][3] * inv_l_hi));
    const uint32_t row_lo_cols8 = pack_bf16_pair(
        __float2bfloat16(acc_frag[value_tile][4] * inv_l_lo),
        __float2bfloat16(acc_frag[value_tile][5] * inv_l_lo));
    const uint32_t row_hi_cols8 = pack_bf16_pair(
        __float2bfloat16(acc_frag[value_tile][6] * inv_l_hi),
        __float2bfloat16(acc_frag[value_tile][7] * inv_l_hi));

#define STORE_COALESCED_ROW_GROUP(ROW_GROUP_BASE, PACK0, PACK8, SRC_ROW_OFFSET)                  \
    do {                                                                                        \
      const int dst_local_row = (ROW_GROUP_BASE) + (lane_id >> 3);                               \
      const int dst_pair = lane_id & 3;                                                         \
      const int src_lane = ((dst_local_row - (SRC_ROW_OFFSET)) << 2) + dst_pair;                \
      const uint32_t packed0 = __shfl_sync(0xffffffff, (PACK0), src_lane);                      \
      const uint32_t packed8 = __shfl_sync(0xffffffff, (PACK8), src_lane);                      \
      const uint32_t packed = (lane_id & 4) ? packed8 : packed0;                                \
      const int global_row = query_block_start + row_base + dst_local_row;                      \
      if (global_row < seq_len) {                                                              \
        const int out_col = value_tile * kValueTileN + ((lane_id & 7) << 1);                    \
        const int64_t out_offset =                                                             \
            bh_offset_elems + static_cast<int64_t>(global_row) * HEAD_DIM + out_col;           \
        *reinterpret_cast<uint32_t*>(o + out_offset) = packed;                                 \
      }                                                                                         \
    } while (0)

    STORE_COALESCED_ROW_GROUP(0, row_lo_cols0, row_lo_cols8, 0);
    STORE_COALESCED_ROW_GROUP(4, row_lo_cols0, row_lo_cols8, 0);
    STORE_COALESCED_ROW_GROUP(8, row_hi_cols0, row_hi_cols8, 8);
    STORE_COALESCED_ROW_GROUP(12, row_hi_cols0, row_hi_cols8, 8);
#undef STORE_COALESCED_ROW_GROUP
  }
}

template <int STRIDE>
__device__ inline uint32_t manual_swizzle(uint32_t index) {
  if constexpr (STRIDE == 16) {
    return index;
  }
  constexpr int kDivisor = (64 / STRIDE) > 1 ? (64 / STRIDE) : 1;
  const uint32_t row_idx = (index / STRIDE) & 7;
  return index ^ ((row_idx / kDivisor) << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void manual_global_to_shared_swizzle(
    uint32_t dst,
    const __nv_bfloat16* src,
    int src_stride,
    int tid) {
  constexpr int kElemsPerAccess = 16 / sizeof(__nv_bfloat16);
  constexpr int kIters = HEIGHT * WIDTH / (TB_SIZE * kElemsPerAccess);
#pragma unroll
  for (int iter = 0; iter < kIters; ++iter) {
    const int idx = (iter * TB_SIZE + tid) * kElemsPerAccess;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    const uint32_t dst_addr =
        manual_swizzle<WIDTH * sizeof(__nv_bfloat16)>(
            dst + (row * WIDTH + col) * sizeof(__nv_bfloat16));
    const __nv_bfloat16* src_addr = src + row * src_stride + col;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_addr), "l"(src_addr));
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void manual_global_to_shared_swizzle_guarded(
    uint32_t dst,
    const __nv_bfloat16* src,
    int src_stride,
    int valid_rows,
    int tid) {
  constexpr int kElemsPerAccess = 16 / sizeof(__nv_bfloat16);
  constexpr int kIters = HEIGHT * WIDTH / (TB_SIZE * kElemsPerAccess);
#pragma unroll
  for (int iter = 0; iter < kIters; ++iter) {
    const int idx = (iter * TB_SIZE + tid) * kElemsPerAccess;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    const uint32_t dst_addr =
        manual_swizzle<WIDTH * sizeof(__nv_bfloat16)>(
            dst + (row * WIDTH + col) * sizeof(__nv_bfloat16));
    if (row < valid_rows) {
      const __nv_bfloat16* src_addr = src + row * src_stride + col;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_addr), "l"(src_addr));
    } else {
      asm volatile(
          "st.shared.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "r"(dst_addr), "r"(0));
    }
  }
}

template <int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int NUM_WARPS, bool CAUSAL>
__launch_bounds__(NUM_WARPS * kThreadsPerWarp)
__global__ void flash_attention_2_forward_v4_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ o,
    int seq_len) {
  static_assert(BLOCK_Q == 64, "manual-v4 fast path expects BLOCK_Q=64");
  static_assert(BLOCK_KV == 32, "manual-v4 fast path expects BLOCK_KV=32");
  static_assert(HEAD_DIM == 128, "manual-v4 fast path expects head_dim=128");
  static_assert(NUM_WARPS == 4, "manual-v4 fast path expects 4 warps");

  constexpr int kThreads = NUM_WARPS * kThreadsPerWarp;
  constexpr int kWarpQ = BLOCK_Q / NUM_WARPS;
  constexpr int kMmaM = 16;
  constexpr int kMmaN = 8;
  constexpr int kMmaK = 16;

  const int tid = threadIdx.x;
  const int warp_id = tid / kThreadsPerWarp;
  const int lane_id = tid & (kThreadsPerWarp - 1);
  const int num_q_blocks = seq_len / BLOCK_Q;
  const int bh_idx = blockIdx.x / num_q_blocks;
  const int q_block_idx = blockIdx.x - bh_idx * num_q_blocks;
  const int q_block_start = q_block_idx * BLOCK_Q;

  q += (static_cast<int64_t>(bh_idx) * seq_len + q_block_start) * HEAD_DIM;
  k += static_cast<int64_t>(bh_idx) * seq_len * HEAD_DIM;
  v += static_cast<int64_t>(bh_idx) * seq_len * HEAD_DIM;
  o += (static_cast<int64_t>(bh_idx) * seq_len + q_block_start) * HEAD_DIM;

  extern __shared__ __align__(32) unsigned char smem_raw[];
  auto* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  const uint32_t q_smem = shared_u32_addr(smem);
  const uint32_t k_smem = q_smem;
  const uint32_t v_smem = k_smem + BLOCK_KV * HEAD_DIM * sizeof(__nv_bfloat16);

  uint32_t q_frag[kWarpQ / kMmaM][HEAD_DIM / kMmaK][4];
  uint32_t k_frag[BLOCK_KV / kMmaN][HEAD_DIM / kMmaK][2];
  uint32_t p_frag[kWarpQ / kMmaM][BLOCK_KV / kMmaK][4];
  uint32_t v_frag[BLOCK_KV / kMmaK][HEAD_DIM / kMmaN][2];
  float out_frag[kWarpQ / kMmaM][HEAD_DIM / kMmaN][4] = {};

  uint32_t q_smem_thread;
  uint32_t k_smem_thread;
  uint32_t v_smem_thread;
  {
    const int row_off = warp_id * kWarpQ + (lane_id & 15);
    const int col_off = (lane_id >> 4) * 8;
    q_smem_thread = manual_swizzle<HEAD_DIM * sizeof(__nv_bfloat16)>(
        q_smem + (row_off * HEAD_DIM + col_off) * sizeof(__nv_bfloat16));
  }
  {
    const int row_off = lane_id & 7;
    const int col_off = (lane_id >> 3) * 8;
    k_smem_thread = manual_swizzle<HEAD_DIM * sizeof(__nv_bfloat16)>(
        k_smem + (row_off * HEAD_DIM + col_off) * sizeof(__nv_bfloat16));
  }
  {
    const int row_off = lane_id & 15;
    const int col_off = (lane_id >> 4) * 8;
    v_smem_thread = manual_swizzle<HEAD_DIM * sizeof(__nv_bfloat16)>(
        v_smem + (row_off * HEAD_DIM + col_off) * sizeof(__nv_bfloat16));
  }

  float row_max[kWarpQ / kMmaM][2];
  float row_sum[kWarpQ / kMmaM][2] = {};
#pragma unroll
  for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
    row_max[q_panel][0] = -FLT_MAX;
    row_max[q_panel][1] = -FLT_MAX;
  }

  manual_global_to_shared_swizzle<BLOCK_Q, HEAD_DIM, kThreads>(q_smem, q, HEAD_DIM, tid);
  cp_async_commit_group();
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();

#pragma unroll
  for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
#pragma unroll
    for (int d_panel = 0; d_panel < HEAD_DIM / kMmaK; ++d_panel) {
      uint32_t addr = q_smem_thread;
      addr += q_panel * kMmaM * HEAD_DIM * sizeof(__nv_bfloat16);
      addr ^= d_panel * kMmaK * sizeof(__nv_bfloat16);
      ldmatrix_m8n8_x4(q_frag[q_panel][d_panel], addr);
    }
  }
  __syncthreads();

  const int num_kv_iters =
      CAUSAL ? ((q_block_start + BLOCK_Q) / BLOCK_KV) : (seq_len / BLOCK_KV);
  auto load_k = [&](int kv_iter) {
    if (kv_iter < num_kv_iters) {
      const uint32_t dst =
          k_smem + (kv_iter & 1) * (2 * BLOCK_KV * HEAD_DIM * sizeof(__nv_bfloat16));
      manual_global_to_shared_swizzle<BLOCK_KV, HEAD_DIM, kThreads>(dst, k, HEAD_DIM, tid);
      k += BLOCK_KV * HEAD_DIM;
    }
    cp_async_commit_group();
  };
  auto load_v = [&](int kv_iter) {
    if (kv_iter < num_kv_iters) {
      const uint32_t dst =
          v_smem + (kv_iter & 1) * (2 * BLOCK_KV * HEAD_DIM * sizeof(__nv_bfloat16));
      manual_global_to_shared_swizzle<BLOCK_KV, HEAD_DIM, kThreads>(dst, v, HEAD_DIM, tid);
      v += BLOCK_KV * HEAD_DIM;
    }
    cp_async_commit_group();
  };

  load_k(0);
  load_v(0);

  for (int kv_iter = 0; kv_iter < num_kv_iters; ++kv_iter) {
    float score_frag[kWarpQ / kMmaM][BLOCK_KV / kMmaN][4] = {};
    const int kv_start = kv_iter * BLOCK_KV;
    const bool needs_causal_mask = CAUSAL && (kv_start >= q_block_start);

    load_k(kv_iter + 1);
    asm volatile("cp.async.wait_group 2;\n" ::);
    __syncthreads();

#pragma unroll
    for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaN; ++kv_panel) {
#pragma unroll
      for (int d_panel = 0; d_panel < HEAD_DIM / kMmaK; d_panel += 2) {
        uint32_t addr =
            k_smem_thread + (kv_iter & 1) * (2 * BLOCK_KV * HEAD_DIM * sizeof(__nv_bfloat16));
        addr += kv_panel * kMmaN * HEAD_DIM * sizeof(__nv_bfloat16);
        addr ^= d_panel * kMmaK * sizeof(__nv_bfloat16);
        ldmatrix_m8n8_x4(k_frag[kv_panel][d_panel], addr);
      }
    }

#pragma unroll
    for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
#pragma unroll
      for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaN; ++kv_panel) {
#pragma unroll
        for (int d_panel = 0; d_panel < HEAD_DIM / kMmaK; ++d_panel) {
          mma_m16n8k16_bf16(
              score_frag[q_panel][kv_panel],
              q_frag[q_panel][d_panel],
              k_frag[kv_panel][d_panel]);
        }
      }
    }

    load_v(kv_iter + 1);

#pragma unroll
    for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
#pragma unroll
      for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaN; ++kv_panel) {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; ++reg_id) {
          float score = score_frag[q_panel][kv_panel][reg_id] * rsqrtf(static_cast<float>(HEAD_DIM));
          if (needs_causal_mask) {
            const int local_row =
                warp_id * kWarpQ + q_panel * kMmaM + mma_accumulator_fragment_row(lane_id, reg_id);
            const int local_col =
                kv_panel * kMmaN + mma_accumulator_fragment_col(lane_id, reg_id);
            score = (kv_start + local_col <= q_block_start + local_row) ? score : -FLT_MAX;
          }
          score_frag[q_panel][kv_panel][reg_id] = score;
        }
      }

      float tile_max[2];
#pragma unroll
      for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaN; ++kv_panel) {
        float* regs = score_frag[q_panel][kv_panel];
        if (kv_panel == 0) {
          tile_max[0] = fmaxf(regs[0], regs[1]);
          tile_max[1] = fmaxf(regs[2], regs[3]);
        } else {
          tile_max[0] = fmaxf(tile_max[0], fmaxf(regs[0], regs[1]));
          tile_max[1] = fmaxf(tile_max[1], fmaxf(regs[2], regs[3]));
        }
      }
      tile_max[0] = lane_group4_max(tile_max[0]);
      tile_max[1] = lane_group4_max(tile_max[1]);
      tile_max[0] = fmaxf(tile_max[0], row_max[q_panel][0]);
      tile_max[1] = fmaxf(tile_max[1], row_max[q_panel][1]);

      const float rescale0 = expf(row_max[q_panel][0] - tile_max[0]);
      const float rescale1 = expf(row_max[q_panel][1] - tile_max[1]);
#pragma unroll
      for (int d_panel = 0; d_panel < HEAD_DIM / kMmaN; ++d_panel) {
        out_frag[q_panel][d_panel][0] *= rescale0;
        out_frag[q_panel][d_panel][1] *= rescale0;
        out_frag[q_panel][d_panel][2] *= rescale1;
        out_frag[q_panel][d_panel][3] *= rescale1;
      }
      row_max[q_panel][0] = tile_max[0];
      row_max[q_panel][1] = tile_max[1];

      float tile_sum[2];
#pragma unroll
      for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaN; ++kv_panel) {
        float* regs = score_frag[q_panel][kv_panel];
        regs[0] = expf(regs[0] - row_max[q_panel][0]);
        regs[1] = expf(regs[1] - row_max[q_panel][0]);
        regs[2] = expf(regs[2] - row_max[q_panel][1]);
        regs[3] = expf(regs[3] - row_max[q_panel][1]);
        if (kv_panel == 0) {
          tile_sum[0] = regs[0] + regs[1];
          tile_sum[1] = regs[2] + regs[3];
        } else {
          tile_sum[0] += regs[0] + regs[1];
          tile_sum[1] += regs[2] + regs[3];
        }

        auto* p_vals = reinterpret_cast<__nv_bfloat162*>(p_frag[q_panel][kv_panel / 2]);
        p_vals[(kv_panel & 1) * 2] =
            __float22bfloat162_rn(make_float2(regs[0], regs[1]));
        p_vals[(kv_panel & 1) * 2 + 1] =
            __float22bfloat162_rn(make_float2(regs[2], regs[3]));
      }
      tile_sum[0] = lane_group4_sum(tile_sum[0]);
      tile_sum[1] = lane_group4_sum(tile_sum[1]);
      row_sum[q_panel][0] = row_sum[q_panel][0] * rescale0 + tile_sum[0];
      row_sum[q_panel][1] = row_sum[q_panel][1] * rescale1 + tile_sum[1];
    }

    asm volatile("cp.async.wait_group 2;\n" ::);
    __syncthreads();

#pragma unroll
    for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaK; ++kv_panel) {
#pragma unroll
      for (int d_panel = 0; d_panel < HEAD_DIM / kMmaN; d_panel += 2) {
        uint32_t addr =
            v_smem_thread + (kv_iter & 1) * (2 * BLOCK_KV * HEAD_DIM * sizeof(__nv_bfloat16));
        addr += kv_panel * kMmaK * HEAD_DIM * sizeof(__nv_bfloat16);
        addr ^= d_panel * kMmaN * sizeof(__nv_bfloat16);
        ldmatrix_m8n8_x4_trans(v_frag[kv_panel][d_panel], addr);
      }
    }

#pragma unroll
    for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
#pragma unroll
      for (int d_panel = 0; d_panel < HEAD_DIM / kMmaN; ++d_panel) {
#pragma unroll
        for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaK; ++kv_panel) {
          mma_m16n8k16_bf16(out_frag[q_panel][d_panel], p_frag[q_panel][kv_panel], v_frag[kv_panel][d_panel]);
        }
      }
    }
  }

#pragma unroll
  for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
#pragma unroll
    for (int d_panel = 0; d_panel < HEAD_DIM / kMmaN; ++d_panel) {
      const int row = warp_id * kWarpQ + q_panel * kMmaM + (lane_id >> 2);
      const int col = d_panel * kMmaN + ((lane_id & 3) << 1);
      float* regs = out_frag[q_panel][d_panel];
      regs[0] /= row_sum[q_panel][0];
      regs[1] /= row_sum[q_panel][0];
      regs[2] /= row_sum[q_panel][1];
      regs[3] /= row_sum[q_panel][1];
      *reinterpret_cast<__nv_bfloat162*>(o + row * HEAD_DIM + col) =
          __float22bfloat162_rn(make_float2(regs[0], regs[1]));
      *reinterpret_cast<__nv_bfloat162*>(o + (row + 8) * HEAD_DIM + col) =
          __float22bfloat162_rn(make_float2(regs[2], regs[3]));
    }
  }
}

template <int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int NUM_WARPS, bool CAUSAL>
__launch_bounds__(NUM_WARPS * kThreadsPerWarp, 3)
__global__ void flash_attention_2_forward_v4_tail_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ o,
    int seq_len) {
  static_assert(BLOCK_Q == 64, "manual-v4 fast path expects BLOCK_Q=64");
  static_assert(BLOCK_KV == 32, "manual-v4 fast path expects BLOCK_KV=32");
  static_assert(HEAD_DIM == 128, "manual-v4 fast path expects head_dim=128");
  static_assert(NUM_WARPS == 4, "manual-v4 fast path expects 4 warps");

  constexpr int kThreads = NUM_WARPS * kThreadsPerWarp;
  constexpr int kWarpQ = BLOCK_Q / NUM_WARPS;
  constexpr int kMmaM = 16;
  constexpr int kMmaN = 8;
  constexpr int kMmaK = 16;

  const int tid = threadIdx.x;
  const int warp_id = tid / kThreadsPerWarp;
  const int lane_id = tid & (kThreadsPerWarp - 1);
  const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
  const int bh_idx = blockIdx.x / num_q_blocks;
  const int q_block_idx = blockIdx.x - bh_idx * num_q_blocks;
  const int q_block_start = q_block_idx * BLOCK_Q;
  const int q_valid_rows = min(BLOCK_Q, seq_len - q_block_start);
  const int block_last_query = q_block_start + q_valid_rows - 1;

  q += (static_cast<int64_t>(bh_idx) * seq_len + q_block_start) * HEAD_DIM;
  k += static_cast<int64_t>(bh_idx) * seq_len * HEAD_DIM;
  v += static_cast<int64_t>(bh_idx) * seq_len * HEAD_DIM;
  o += (static_cast<int64_t>(bh_idx) * seq_len + q_block_start) * HEAD_DIM;

  extern __shared__ __align__(32) unsigned char smem_raw[];
  auto* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  const uint32_t q_smem = shared_u32_addr(smem);
  const uint32_t k_smem = q_smem;
  const uint32_t v_smem = k_smem + BLOCK_KV * HEAD_DIM * sizeof(__nv_bfloat16);

  uint32_t q_frag[kWarpQ / kMmaM][HEAD_DIM / kMmaK][4];
  uint32_t k_frag[BLOCK_KV / kMmaN][HEAD_DIM / kMmaK][2];
  uint32_t p_frag[kWarpQ / kMmaM][BLOCK_KV / kMmaK][4];
  uint32_t v_frag[BLOCK_KV / kMmaK][HEAD_DIM / kMmaN][2];
  float out_frag[kWarpQ / kMmaM][HEAD_DIM / kMmaN][4] = {};

  uint32_t q_smem_thread;
  uint32_t k_smem_thread;
  uint32_t v_smem_thread;
  {
    const int row_off = warp_id * kWarpQ + (lane_id & 15);
    const int col_off = (lane_id >> 4) * 8;
    q_smem_thread = manual_swizzle<HEAD_DIM * sizeof(__nv_bfloat16)>(
        q_smem + (row_off * HEAD_DIM + col_off) * sizeof(__nv_bfloat16));
  }
  {
    const int row_off = lane_id & 7;
    const int col_off = (lane_id >> 3) * 8;
    k_smem_thread = manual_swizzle<HEAD_DIM * sizeof(__nv_bfloat16)>(
        k_smem + (row_off * HEAD_DIM + col_off) * sizeof(__nv_bfloat16));
  }
  {
    const int row_off = lane_id & 15;
    const int col_off = (lane_id >> 4) * 8;
    v_smem_thread = manual_swizzle<HEAD_DIM * sizeof(__nv_bfloat16)>(
        v_smem + (row_off * HEAD_DIM + col_off) * sizeof(__nv_bfloat16));
  }

  float row_max[kWarpQ / kMmaM][2];
  float row_sum[kWarpQ / kMmaM][2] = {};
#pragma unroll
  for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
    row_max[q_panel][0] = -FLT_MAX;
    row_max[q_panel][1] = -FLT_MAX;
  }

  if (q_valid_rows == BLOCK_Q) {
    manual_global_to_shared_swizzle<BLOCK_Q, HEAD_DIM, kThreads>(q_smem, q, HEAD_DIM, tid);
  } else {
    manual_global_to_shared_swizzle_guarded<BLOCK_Q, HEAD_DIM, kThreads>(
        q_smem, q, HEAD_DIM, q_valid_rows, tid);
  }
  cp_async_commit_group();
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();

#pragma unroll
  for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
#pragma unroll
    for (int d_panel = 0; d_panel < HEAD_DIM / kMmaK; ++d_panel) {
      uint32_t addr = q_smem_thread;
      addr += q_panel * kMmaM * HEAD_DIM * sizeof(__nv_bfloat16);
      addr ^= d_panel * kMmaK * sizeof(__nv_bfloat16);
      ldmatrix_m8n8_x4(q_frag[q_panel][d_panel], addr);
    }
  }
  __syncthreads();

  const int num_kv_iters =
      CAUSAL ? ((block_last_query + BLOCK_KV) / BLOCK_KV) : ((seq_len + BLOCK_KV - 1) / BLOCK_KV);
  auto load_k = [&](int kv_iter) {
    if (kv_iter < num_kv_iters) {
      const int kv_start = kv_iter * BLOCK_KV;
      const int valid_rows = min(BLOCK_KV, seq_len - kv_start);
      const uint32_t dst =
          k_smem + (kv_iter & 1) * (2 * BLOCK_KV * HEAD_DIM * sizeof(__nv_bfloat16));
      if (valid_rows == BLOCK_KV) {
        manual_global_to_shared_swizzle<BLOCK_KV, HEAD_DIM, kThreads>(dst, k, HEAD_DIM, tid);
      } else {
        manual_global_to_shared_swizzle_guarded<BLOCK_KV, HEAD_DIM, kThreads>(
            dst, k, HEAD_DIM, valid_rows, tid);
      }
      k += BLOCK_KV * HEAD_DIM;
    }
    cp_async_commit_group();
  };
  auto load_v = [&](int kv_iter) {
    if (kv_iter < num_kv_iters) {
      const int kv_start = kv_iter * BLOCK_KV;
      const int valid_rows = min(BLOCK_KV, seq_len - kv_start);
      const uint32_t dst =
          v_smem + (kv_iter & 1) * (2 * BLOCK_KV * HEAD_DIM * sizeof(__nv_bfloat16));
      if (valid_rows == BLOCK_KV) {
        manual_global_to_shared_swizzle<BLOCK_KV, HEAD_DIM, kThreads>(dst, v, HEAD_DIM, tid);
      } else {
        manual_global_to_shared_swizzle_guarded<BLOCK_KV, HEAD_DIM, kThreads>(
            dst, v, HEAD_DIM, valid_rows, tid);
      }
      v += BLOCK_KV * HEAD_DIM;
    }
    cp_async_commit_group();
  };

  load_k(0);
  load_v(0);

  for (int kv_iter = 0; kv_iter < num_kv_iters; ++kv_iter) {
    float score_frag[kWarpQ / kMmaM][BLOCK_KV / kMmaN][4] = {};
    const int kv_start = kv_iter * BLOCK_KV;
    const int kv_valid_rows = min(BLOCK_KV, seq_len - kv_start);
    const bool needs_bounds_mask = q_valid_rows < BLOCK_Q || kv_valid_rows < BLOCK_KV;
    const bool needs_causal_mask = CAUSAL && (kv_start >= q_block_start);
    const bool needs_mask = needs_bounds_mask || needs_causal_mask;

    load_k(kv_iter + 1);
    asm volatile("cp.async.wait_group 2;\n" ::);
    __syncthreads();

#pragma unroll
    for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaN; ++kv_panel) {
#pragma unroll
      for (int d_panel = 0; d_panel < HEAD_DIM / kMmaK; d_panel += 2) {
        uint32_t addr =
            k_smem_thread + (kv_iter & 1) * (2 * BLOCK_KV * HEAD_DIM * sizeof(__nv_bfloat16));
        addr += kv_panel * kMmaN * HEAD_DIM * sizeof(__nv_bfloat16);
        addr ^= d_panel * kMmaK * sizeof(__nv_bfloat16);
        ldmatrix_m8n8_x4(k_frag[kv_panel][d_panel], addr);
      }
    }

#pragma unroll
    for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
#pragma unroll
      for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaN; ++kv_panel) {
#pragma unroll
        for (int d_panel = 0; d_panel < HEAD_DIM / kMmaK; ++d_panel) {
          mma_m16n8k16_bf16(
              score_frag[q_panel][kv_panel],
              q_frag[q_panel][d_panel],
              k_frag[kv_panel][d_panel]);
        }
      }
    }

    load_v(kv_iter + 1);

#pragma unroll
    for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
#pragma unroll
      for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaN; ++kv_panel) {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; ++reg_id) {
          float score = score_frag[q_panel][kv_panel][reg_id] * rsqrtf(static_cast<float>(HEAD_DIM));
          if (needs_mask) {
            const int local_row =
                warp_id * kWarpQ + q_panel * kMmaM + mma_accumulator_fragment_row(lane_id, reg_id);
            const int local_col =
                kv_panel * kMmaN + mma_accumulator_fragment_col(lane_id, reg_id);
            const bool valid =
                local_row < q_valid_rows &&
                local_col < kv_valid_rows &&
                (!CAUSAL || kv_start + local_col <= q_block_start + local_row);
            score = valid ? score : -FLT_MAX;
          }
          score_frag[q_panel][kv_panel][reg_id] = score;
        }
      }

      float tile_max[2];
#pragma unroll
      for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaN; ++kv_panel) {
        float* regs = score_frag[q_panel][kv_panel];
        if (kv_panel == 0) {
          tile_max[0] = fmaxf(regs[0], regs[1]);
          tile_max[1] = fmaxf(regs[2], regs[3]);
        } else {
          tile_max[0] = fmaxf(tile_max[0], fmaxf(regs[0], regs[1]));
          tile_max[1] = fmaxf(tile_max[1], fmaxf(regs[2], regs[3]));
        }
      }
      tile_max[0] = lane_group4_max(tile_max[0]);
      tile_max[1] = lane_group4_max(tile_max[1]);
      tile_max[0] = fmaxf(tile_max[0], row_max[q_panel][0]);
      tile_max[1] = fmaxf(tile_max[1], row_max[q_panel][1]);

      const float rescale0 = expf(row_max[q_panel][0] - tile_max[0]);
      const float rescale1 = expf(row_max[q_panel][1] - tile_max[1]);
#pragma unroll
      for (int d_panel = 0; d_panel < HEAD_DIM / kMmaN; ++d_panel) {
        out_frag[q_panel][d_panel][0] *= rescale0;
        out_frag[q_panel][d_panel][1] *= rescale0;
        out_frag[q_panel][d_panel][2] *= rescale1;
        out_frag[q_panel][d_panel][3] *= rescale1;
      }
      row_max[q_panel][0] = tile_max[0];
      row_max[q_panel][1] = tile_max[1];

      float tile_sum[2];
#pragma unroll
      for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaN; ++kv_panel) {
        float* regs = score_frag[q_panel][kv_panel];
        regs[0] = expf(regs[0] - row_max[q_panel][0]);
        regs[1] = expf(regs[1] - row_max[q_panel][0]);
        regs[2] = expf(regs[2] - row_max[q_panel][1]);
        regs[3] = expf(regs[3] - row_max[q_panel][1]);
        if (kv_panel == 0) {
          tile_sum[0] = regs[0] + regs[1];
          tile_sum[1] = regs[2] + regs[3];
        } else {
          tile_sum[0] += regs[0] + regs[1];
          tile_sum[1] += regs[2] + regs[3];
        }

        auto* p_vals = reinterpret_cast<__nv_bfloat162*>(p_frag[q_panel][kv_panel / 2]);
        p_vals[(kv_panel & 1) * 2] =
            __float22bfloat162_rn(make_float2(regs[0], regs[1]));
        p_vals[(kv_panel & 1) * 2 + 1] =
            __float22bfloat162_rn(make_float2(regs[2], regs[3]));
      }
      tile_sum[0] = lane_group4_sum(tile_sum[0]);
      tile_sum[1] = lane_group4_sum(tile_sum[1]);
      row_sum[q_panel][0] = row_sum[q_panel][0] * rescale0 + tile_sum[0];
      row_sum[q_panel][1] = row_sum[q_panel][1] * rescale1 + tile_sum[1];
    }

    asm volatile("cp.async.wait_group 2;\n" ::);
    __syncthreads();

#pragma unroll
    for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaK; ++kv_panel) {
#pragma unroll
      for (int d_panel = 0; d_panel < HEAD_DIM / kMmaN; d_panel += 2) {
        uint32_t addr =
            v_smem_thread + (kv_iter & 1) * (2 * BLOCK_KV * HEAD_DIM * sizeof(__nv_bfloat16));
        addr += kv_panel * kMmaK * HEAD_DIM * sizeof(__nv_bfloat16);
        addr ^= d_panel * kMmaN * sizeof(__nv_bfloat16);
        ldmatrix_m8n8_x4_trans(v_frag[kv_panel][d_panel], addr);
      }
    }

#pragma unroll
    for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
#pragma unroll
      for (int d_panel = 0; d_panel < HEAD_DIM / kMmaN; ++d_panel) {
#pragma unroll
        for (int kv_panel = 0; kv_panel < BLOCK_KV / kMmaK; ++kv_panel) {
          mma_m16n8k16_bf16(out_frag[q_panel][d_panel], p_frag[q_panel][kv_panel], v_frag[kv_panel][d_panel]);
        }
      }
    }
  }

#pragma unroll
  for (int q_panel = 0; q_panel < kWarpQ / kMmaM; ++q_panel) {
#pragma unroll
    for (int d_panel = 0; d_panel < HEAD_DIM / kMmaN; ++d_panel) {
      const int row = warp_id * kWarpQ + q_panel * kMmaM + (lane_id >> 2);
      const int col = d_panel * kMmaN + ((lane_id & 3) << 1);
      float* regs = out_frag[q_panel][d_panel];
      regs[0] /= row_sum[q_panel][0];
      regs[1] /= row_sum[q_panel][0];
      regs[2] /= row_sum[q_panel][1];
      regs[3] /= row_sum[q_panel][1];
      if (row < q_valid_rows) {
        *reinterpret_cast<__nv_bfloat162*>(o + row * HEAD_DIM + col) =
            __float22bfloat162_rn(make_float2(regs[0], regs[1]));
      }
      if (row + 8 < q_valid_rows) {
        *reinterpret_cast<__nv_bfloat162*>(o + (row + 8) * HEAD_DIM + col) =
            __float22bfloat162_rn(make_float2(regs[2], regs[3]));
      }
    }
  }
}

}  // namespace

torch::Tensor flash_attention_2_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double sm_scale,
    bool causal) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(v);
  CHECK_BF16(q);
  CHECK_BF16(k);
  CHECK_BF16(v);

  TORCH_CHECK(q.dim() == 4, "q must have shape [batch, heads, seq_len, head_dim]");
  TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape");
  TORCH_CHECK(v.sizes() == q.sizes(), "v must match q shape");
  TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 128");

  const auto batch_size = static_cast<int>(q.size(0));
  const auto num_heads = static_cast<int>(q.size(1));
  const auto seq_len = static_cast<int>(q.size(2));
  auto output = torch::empty_like(q);

  const auto stream = at::cuda::getCurrentCUDAStream();
  const float sm_scale_log2e = static_cast<float>(sm_scale) * kLog2e;

  auto launch_regtc = [&](auto causal_tag) {
    constexpr bool is_causal = decltype(causal_tag)::value;
    constexpr int block_q = 64;
    constexpr int block_kv = 16;
    constexpr int head_dim = kHeadDim;
    constexpr int warps = 4;
    constexpr int shared_pitch = head_dim + 8;
    constexpr int threads = warps * kThreadsPerWarp;
    constexpr int operand_tiles = 4;
    static_assert(block_q == operand_tiles * block_kv, "Q shared tile is aliased with K/V operand tiles");
    const size_t shared_mem_bytes = operand_tiles * block_kv * shared_pitch * sizeof(__nv_bfloat16);
    const dim3 grid((seq_len + block_q - 1) / block_q, num_heads, batch_size);
    if constexpr (is_causal) {
      flash_attention_2_forward_regtc_kernel<block_q, block_kv, head_dim, warps, true>
          <<<grid, threads, shared_mem_bytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          num_heads,
          seq_len,
          sm_scale_log2e);
    } else {
      flash_attention_2_forward_regtc_kernel<block_q, block_kv, head_dim, warps, false>
          <<<grid, threads, shared_mem_bytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          num_heads,
          seq_len,
          sm_scale_log2e);
    }
  };

  auto launch_v4 = [&](auto causal_tag) {
    constexpr bool is_causal = decltype(causal_tag)::value;
    constexpr int block_q = 64;
    constexpr int block_kv = 32;
    constexpr int head_dim = kHeadDim;
    constexpr int warps = 4;
    constexpr int threads = warps * kThreadsPerWarp;
    constexpr int shared_rows = block_kv * 4;
    const size_t shared_mem_bytes = shared_rows * head_dim * sizeof(__nv_bfloat16);
    const dim3 grid(batch_size * num_heads * (seq_len / block_q));
    if constexpr (is_causal) {
      flash_attention_2_forward_v4_kernel<block_q, block_kv, head_dim, warps, true>
          <<<grid, threads, shared_mem_bytes, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
              seq_len);
    } else {
      flash_attention_2_forward_v4_kernel<block_q, block_kv, head_dim, warps, false>
          <<<grid, threads, shared_mem_bytes, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
              seq_len);
    }
  };

  auto launch_v4_tail = [&](auto causal_tag) {
    constexpr bool is_causal = decltype(causal_tag)::value;
    constexpr int block_q = 64;
    constexpr int block_kv = 32;
    constexpr int head_dim = kHeadDim;
    constexpr int warps = 4;
    constexpr int threads = warps * kThreadsPerWarp;
    constexpr int shared_rows = block_kv * 4;
    const size_t shared_mem_bytes = shared_rows * head_dim * sizeof(__nv_bfloat16);
    const dim3 grid(batch_size * num_heads * ((seq_len + block_q - 1) / block_q));
    if constexpr (is_causal) {
      flash_attention_2_forward_v4_tail_kernel<block_q, block_kv, head_dim, warps, true>
          <<<grid, threads, shared_mem_bytes, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
              seq_len);
    } else {
      flash_attention_2_forward_v4_tail_kernel<block_q, block_kv, head_dim, warps, false>
          <<<grid, threads, shared_mem_bytes, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
              seq_len);
    }
  };

  if (!causal && seq_len > 0 && (seq_len % 64) == 0) {
    launch_v4(std::false_type{});
  } else if (!causal && seq_len > 0) {
    launch_v4_tail(std::false_type{});
  } else if (causal && seq_len > 0) {
    launch_v4_tail(std::true_type{});
  } else if (causal) {
    launch_regtc(std::true_type{});
  } else {
    launch_regtc(std::false_type{});
  }

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "flash_attention_2_forward launch failed: ", cudaGetErrorString(err));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "flash_attention_2_forward",
      &flash_attention_2_forward,
      "FlashAttention-2 style forward kernel (CUDA)");
}
