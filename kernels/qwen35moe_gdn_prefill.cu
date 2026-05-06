#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int D = 128;
constexpr int H_K = 16;
constexpr int H_V = 32;
constexpr int WARP = 32;
constexpr int COLS_PER_BLOCK = 4;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")

template <int width>
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int mask = width / 2; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, width);
  }
  return x;
}

__global__ __launch_bounds__(WARP * COLS_PER_BLOCK, 2)
void gdn_ref_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ state,
    float* __restrict__ out,
    float* __restrict__ final_state,
    int batch,
    int tokens,
    float scale) {
  const int h = blockIdx.x;
  const int seq = blockIdx.y;
  const int lane = threadIdx.x;
  const int col = blockIdx.z * blockDim.y + threadIdx.y;
  const int kh = h % H_K;

  (void) batch;

  constexpr int rows_per_lane = D / WARP;
  float s_shard[rows_per_lane];

#pragma unroll
  for (int r = 0; r < rows_per_lane; ++r) {
    const int i = r * WARP + lane;
    s_shard[r] = state[((static_cast<int64_t>(seq) * H_V + h) * D + col) * D + i];
  }

  for (int t = 0; t < tokens; ++t) {
    const int64_t q_base = ((static_cast<int64_t>(seq) * tokens + t) * H_K + kh) * D;
    const int64_t v_base = ((static_cast<int64_t>(seq) * tokens + t) * H_V + h) * D;
    const int64_t gb_idx = (static_cast<int64_t>(seq) * tokens + t) * H_V + h;
    const float beta_val = beta[gb_idx];
    const float g_val = expf(g[gb_idx]);

    float kv_shard = 0.0f;
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int i = r * WARP + lane;
      k_reg[r] = k[q_base + i];
      q_reg[r] = q[q_base + i];
      kv_shard += s_shard[r] * k_reg[r];
    }

    const float kv_col = warp_reduce_sum<WARP>(kv_shard);
    const float delta_col = (v[v_base + col] - g_val * kv_col) * beta_val;

    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
      attn_partial += s_shard[r] * q_reg[r];
    }

    const float attn_col = warp_reduce_sum<WARP>(attn_partial);
    if (lane == 0) {
      out[v_base + col] = attn_col * scale;
    }
  }

#pragma unroll
  for (int r = 0; r < rows_per_lane; ++r) {
    const int i = r * WARP + lane;
    final_state[((static_cast<int64_t>(seq) * H_V + h) * D + col) * D + i] = s_shard[r];
  }
}

void check_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const torch::Tensor& state) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_CUDA(g);
  CHECK_CUDA(beta);
  CHECK_CUDA(state);
  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(v);
  CHECK_CONTIGUOUS(g);
  CHECK_CONTIGUOUS(beta);
  CHECK_CONTIGUOUS(state);
  CHECK_FLOAT(q);
  CHECK_FLOAT(k);
  CHECK_FLOAT(v);
  CHECK_FLOAT(g);
  CHECK_FLOAT(beta);
  CHECK_FLOAT(state);
  TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes(), "q/k must be [B,T,16,128]");
  TORCH_CHECK(v.dim() == 4, "v must be [B,T,32,128]");
  TORCH_CHECK(g.dim() == 3 && beta.sizes() == g.sizes(), "g/beta must be [B,T,32]");
  TORCH_CHECK(state.dim() == 4, "state must be [B,32,128,128]");
  TORCH_CHECK(q.size(2) == H_K && q.size(3) == D, "q/k shape must be [B,T,16,128]");
  TORCH_CHECK(v.size(0) == q.size(0) && v.size(1) == q.size(1) && v.size(2) == H_V && v.size(3) == D,
              "v shape must be [B,T,32,128]");
  TORCH_CHECK(g.size(0) == q.size(0) && g.size(1) == q.size(1) && g.size(2) == H_V,
              "g/beta shape must be [B,T,32]");
  TORCH_CHECK(state.size(0) == q.size(0) && state.size(1) == H_V && state.size(2) == D && state.size(3) == D,
              "state shape must be [B,32,128,128]");
}

}  // namespace

std::vector<torch::Tensor> qwen35moe_gdn_prefill_ref(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor state,
    double scale) {
  check_inputs(q, k, v, g, beta, state);

  const int batch = static_cast<int>(q.size(0));
  const int tokens = static_cast<int>(q.size(1));
  auto out = torch::empty_like(v);
  auto final_state = torch::empty_like(state);

  const dim3 grid(H_V, batch, D / COLS_PER_BLOCK);
  const dim3 block(WARP, COLS_PER_BLOCK, 1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  gdn_ref_kernel<<<grid, block, 0, stream>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      g.data_ptr<float>(),
      beta.data_ptr<float>(),
      state.data_ptr<float>(),
      out.data_ptr<float>(),
      final_state.data_ptr<float>(),
      batch,
      tokens,
      static_cast<float>(scale));

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "qwen35moe_gdn_prefill_ref launch failed: ", cudaGetErrorString(err));
  return {out, final_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qwen35moe_gdn_prefill_ref", &qwen35moe_gdn_prefill_ref, "Qwen3.5 MoE GDN FP32 scalar prefill reference");
}
