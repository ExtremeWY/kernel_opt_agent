#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kHeadDim = 128;
constexpr int kThreadsPerWarp = 32;
constexpr int kElementsPerLane = kHeadDim / kThreadsPerWarp;
constexpr int kPairsPerLane = kElementsPerLane / 2;
constexpr int kHeadPairs = kHeadDim / 2;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::BFloat16, #x " must be bfloat16")

template <typename T>
__device__ inline T warp_sum(T value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return __shfl_sync(0xffffffff, value, 0);
}

template <int BLOCK_N, int WARPS_PER_BLOCK>
__global__ void flash_attention_2_forward_kernel(
    const __nv_bfloat162* __restrict__ q,
    const __nv_bfloat162* __restrict__ k,
    const __nv_bfloat162* __restrict__ v,
    __nv_bfloat162* __restrict__ o,
    int batch_size,
    int num_heads,
    int seq_len,
    float sm_scale,
    int causal) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* shared = reinterpret_cast<__nv_bfloat162*>(smem_raw);
  __nv_bfloat162* k_tile = shared;
  __nv_bfloat162* v_tile = shared + BLOCK_N * kHeadPairs;

  const int warp_id = threadIdx.x / kThreadsPerWarp;
  const int lane_id = threadIdx.x % kThreadsPerWarp;
  const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int head_idx = blockIdx.y;
  const int batch_idx = blockIdx.z;

  const int64_t bh_offset =
      (static_cast<int64_t>(batch_idx) * num_heads + head_idx) * seq_len * kHeadPairs;

  if (query_idx >= seq_len) {
    return;
  }

  float2 q_frag[kPairsPerLane];
  float2 acc[kPairsPerLane];
#pragma unroll
  for (int i = 0; i < kPairsPerLane; ++i) {
    const int pair_idx = lane_id * kPairsPerLane + i;
    q_frag[i] = __bfloat1622float2(q[bh_offset + static_cast<int64_t>(query_idx) * kHeadPairs + pair_idx]);
    acc[i] = make_float2(0.0f, 0.0f);
  }

  float m_i = -INFINITY;
  float l_i = 0.0f;

  for (int start_n = 0; start_n < seq_len; start_n += BLOCK_N) {
    const int tile_count = min(BLOCK_N, seq_len - start_n);

    for (int linear = threadIdx.x; linear < BLOCK_N * kHeadPairs; linear += blockDim.x) {
      const int row = linear / kHeadPairs;
      const int col = linear % kHeadPairs;
      const int key_idx = start_n + row;
      const int64_t global_idx = bh_offset + static_cast<int64_t>(key_idx) * kHeadPairs + col;
      if (row < tile_count) {
        k_tile[linear] = k[global_idx];
        v_tile[linear] = v[global_idx];
      } else {
        const __nv_bfloat162 zero = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
        k_tile[linear] = zero;
        v_tile[linear] = zero;
      }
    }
    __syncthreads();

    for (int local_n = 0; local_n < tile_count; ++local_n) {
      float score_partial = 0.0f;
#pragma unroll
      for (int i = 0; i < kPairsPerLane; ++i) {
        const int pair_idx = lane_id * kPairsPerLane + i;
        const float2 k_val = __bfloat1622float2(k_tile[local_n * kHeadPairs + pair_idx]);
        score_partial += q_frag[i].x * k_val.x + q_frag[i].y * k_val.y;
      }
      float score = warp_sum(score_partial);
      const int key_idx = start_n + local_n;
      if (causal && key_idx > query_idx) {
        score = -INFINITY;
      } else {
        score *= sm_scale;
      }

      const float new_m = fmaxf(m_i, score);
      const float alpha = isinf(m_i) ? 0.0f : __expf(m_i - new_m);
      const float p = isinf(score) ? 0.0f : __expf(score - new_m);
      const float new_l = l_i * alpha + p;

#pragma unroll
      for (int i = 0; i < kPairsPerLane; ++i) {
        const int pair_idx = lane_id * kPairsPerLane + i;
        const float2 v_val = __bfloat1622float2(v_tile[local_n * kHeadPairs + pair_idx]);
        acc[i].x = acc[i].x * alpha + p * v_val.x;
        acc[i].y = acc[i].y * alpha + p * v_val.y;
      }
      m_i = new_m;
      l_i = new_l;
    }
    __syncthreads();
  }

  const float inv_l = l_i > 0.0f ? 1.0f / l_i : 0.0f;
#pragma unroll
  for (int i = 0; i < kPairsPerLane; ++i) {
    const int pair_idx = lane_id * kPairsPerLane + i;
    const float2 out_val =
        make_float2(acc[i].x * inv_l, acc[i].y * inv_l);
    o[bh_offset + static_cast<int64_t>(query_idx) * kHeadPairs + pair_idx] =
        __float22bfloat162_rn(out_val);
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

  constexpr int block_n = 32;
  constexpr int warps_per_block = 4;
  constexpr int threads = warps_per_block * kThreadsPerWarp;
  const dim3 grid((seq_len + warps_per_block - 1) / warps_per_block, num_heads, batch_size);
  const size_t shared_mem_bytes = 2 * block_n * kHeadPairs * sizeof(__nv_bfloat162);

  const auto stream = at::cuda::getCurrentCUDAStream();
  flash_attention_2_forward_kernel<block_n, warps_per_block><<<grid, threads, shared_mem_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat162*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat162*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat162*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat162*>(output.data_ptr<at::BFloat16>()),
      batch_size,
      num_heads,
      seq_len,
      static_cast<float>(sm_scale),
      causal ? 1 : 0);

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
