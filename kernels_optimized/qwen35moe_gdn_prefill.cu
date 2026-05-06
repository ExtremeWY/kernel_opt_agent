#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using namespace nvcuda;

using bf16_t = __nv_bfloat16;

constexpr int D = 128;
constexpr int H_K = 16;
constexpr int H_V = 32;
constexpr int CHUNK = 32;
constexpr int BLOCK_DV = 32;
constexpr int WARPS = 16;
constexpr int THREADS = WARPS * 32;
constexpr int PRE_WARPS = 6;
constexpr int PRE_THREADS = PRE_WARPS * 32;
constexpr float LOG2E = 1.4426950408889634f;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")

static constexpr size_t align_up(size_t x, size_t a) {
  return (x + a - 1) / a * a;
}

static constexpr size_t TC_QH_OFF      = 0;
static constexpr size_t TC_KH_OFF      = TC_QH_OFF      + size_t(CHUNK) * D     * sizeof(bf16_t);
static constexpr size_t TC_KH_T_OFF    = TC_KH_OFF      + size_t(CHUNK) * D     * sizeof(bf16_t);
static constexpr size_t TC_SH_OFF      = TC_KH_T_OFF;
static constexpr size_t TC_KREST_T_OFF = TC_SH_OFF      + size_t(D)     * BLOCK_DV * sizeof(bf16_t);
static constexpr size_t TC_UH_OFF      = TC_KREST_T_OFF + size_t(D)     * CHUNK * sizeof(bf16_t);
static constexpr size_t TC_PH_OFF      = TC_UH_OFF      + size_t(CHUNK) * BLOCK_DV * sizeof(bf16_t);
static constexpr size_t TC_MAT_OFF     = TC_PH_OFF      + size_t(CHUNK) * CHUNK * sizeof(bf16_t);
static constexpr size_t TC_M_OFF       = TC_MAT_OFF     + size_t(CHUNK) * CHUNK * sizeof(float);
static constexpr size_t TC_U_OFF       = TC_M_OFF       + size_t(CHUNK) * BLOCK_DV * sizeof(float);
static constexpr size_t TC_UPD_OFF     = TC_U_OFF       + size_t(CHUNK) * BLOCK_DV * sizeof(float);
static constexpr size_t TC_BYTES_RAW   = TC_UPD_OFF     + size_t(D)     * BLOCK_DV * sizeof(float);

static_assert(TC_QH_OFF      % 256 == 0, "workspace alignment");
static_assert(TC_KH_OFF      % 256 == 0, "workspace alignment");
static_assert(TC_KH_T_OFF    % 256 == 0, "workspace alignment");
static_assert(TC_SH_OFF      % 256 == 0, "workspace alignment");
static_assert(TC_KREST_T_OFF % 256 == 0, "workspace alignment");
static_assert(TC_UH_OFF      % 256 == 0, "workspace alignment");
static_assert(TC_PH_OFF      % 256 == 0, "workspace alignment");
static_assert(TC_MAT_OFF     % 256 == 0, "workspace alignment");
static_assert(TC_M_OFF       % 256 == 0, "workspace alignment");
static_assert(TC_U_OFF       % 256 == 0, "workspace alignment");
static_assert(TC_UPD_OFF     % 256 == 0, "workspace alignment");

static size_t bytes_per_head() {
  return align_up(TC_BYTES_RAW, 256);
}

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ inline int mma_accumulator_fragment_row(int lane_id, int frag_idx) {
  return (lane_id >> 2) + ((frag_idx & 2) ? 8 : 0);
}

__device__ inline int mma_accumulator_fragment_col(int lane_id, int frag_idx) {
  return ((lane_id & 3) * 2) + (frag_idx & 1) + ((frag_idx >= 4) ? 8 : 0);
}

__device__ inline uint32_t shared_u32_addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
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

template <int WIDTH>
__device__ inline int swizzled_bf16_index(int row, int col) {
  const uint32_t byte_index = static_cast<uint32_t>((row * WIDTH + col) * sizeof(bf16_t));
  return static_cast<int>(manual_swizzle<WIDTH * sizeof(bf16_t)>(byte_index) / sizeof(bf16_t));
}

template <int WIDTH>
__device__ inline uint32_t swizzled_bf16_addr(const bf16_t* base, int row, int col) {
  const uint32_t byte_index = static_cast<uint32_t>((row * WIDTH + col) * sizeof(bf16_t));
  return shared_u32_addr(base) + manual_swizzle<WIDTH * sizeof(bf16_t)>(byte_index);
}

__device__ inline void ldmatrix_m8n8_x2(uint32_t regs[2], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(regs[0]), "=r"(regs[1])
      : "r"(addr));
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

__device__ inline uint32_t pack_bf16_pair(bf16_t lo, bf16_t hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

template <int M>
__device__ inline bf16_t ktrans_swizzled_a_value(const bf16_t* K_swz, int logical_row, int logical_col) {
  return K_swz[swizzled_bf16_index<M>(logical_col, logical_row)];
}

template <int M>
__device__ inline void load_ktrans_a_manual_from_swizzled(uint32_t a_frag[4], const bf16_t* K_swz, int mt, int kt, int lane_id) {
  const int group_id = lane_id >> 2;
  const int thread_in_group = lane_id & 3;
  bf16_t a[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = ((i < 2) || (i >= 4 && i < 6)) ? group_id : group_id + 8;
    const int col = (thread_in_group * 2) + (i & 1) + ((i >= 4) ? 8 : 0);
    a[i] = ktrans_swizzled_a_value<M>(K_swz, mt * 16 + row, kt * 16 + col);
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    a_frag[r] = pack_bf16_pair(a[2 * r], a[2 * r + 1]);
  }
}

template <int M, int N, int K, int LDA, int LDB, int LDC>
__device__ void mma_gemm_bf16_bf16_f32_rm_smem_a_swz_ld_b_ld_c_ld(
    const bf16_t* A,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;
  static_assert(M % 16 == 0, "M must be a multiple of 16");
  static_assert(N % 16 == 0, "N must be a multiple of 16");
  static_assert(K % 16 == 0, "K must be a multiple of 16");

  const int lane_id = threadIdx.x & 31;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    float c_frag0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float c_frag1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      uint32_t a_frag[4];
      const int a_row = mt * 16 + (lane_id & 15);
      const int a_col = kt * 16 + ((lane_id >> 4) * 8);
      ldmatrix_m8n8_x4(a_frag, swizzled_bf16_addr<LDA>(A, a_row, a_col));

      uint32_t b_frag[4];
      const int b_row = kt * 16 + (lane_id & 7) + (((lane_id >> 3) & 1) * 8);
      const int b_col = nt * 16 + ((lane_id >= 16) ? 8 : 0);
      ldmatrix_m8n8_x4_trans(b_frag, shared_u32_addr(B + b_row * LDB + b_col));

      mma_m16n8k16_bf16(c_frag0, a_frag, b_frag);
      mma_m16n8k16_bf16(c_frag1, a_frag, b_frag + 2);
    }

#pragma unroll
    for (int frag_idx = 0; frag_idx < 4; ++frag_idx) {
      const int row = mt * 16 + mma_accumulator_fragment_row(lane_id, frag_idx);
      const int col0 = nt * 16 + mma_accumulator_fragment_col(lane_id, frag_idx);
      const int col1 = col0 + 8;
      C[row * LDC + col0] = c_frag0[frag_idx];
      C[row * LDC + col1] = c_frag1[frag_idx];
    }
  }
}


template <int M, int KHLD>
__device__ inline void load_ktrans_a_manual_from_padded(uint32_t a_frag[4], const bf16_t* K_pad, int mt, int kt, int lane_id) {
  const int group_id = lane_id >> 2;
  const int thread_in_group = lane_id & 3;
  bf16_t v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = ((i < 2) || (i >= 4 && i < 6)) ? group_id : group_id + 8;
    const int col = (thread_in_group * 2) + (i & 1) + ((i >= 4) ? 8 : 0);
    const int logical_row = mt * 16 + row;
    const int logical_col = kt * 16 + col;
    v[i] = K_pad[logical_col * KHLD + logical_row];
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    a_frag[r] = pack_bf16_pair(v[2 * r], v[2 * r + 1]);
  }
}

template <int M, int N, int K, int LDB, int STATE_LD, int KHLD>
__device__ void mma_ktrans_swz_update_state_bf16(
    const bf16_t* K_swz,
    const bf16_t* B,
    bf16_t* state_s,
    float g_last,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;
  static_assert(M % 16 == 0, "M must be a multiple of 16");
  static_assert(N % 16 == 0, "N must be a multiple of 16");
  static_assert(K % 16 == 0, "K must be a multiple of 16");

  const int lane_id = threadIdx.x & 31;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    float c_frag0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float c_frag1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      uint32_t a_frag[4];
      load_ktrans_a_manual_from_padded<M, KHLD>(a_frag, K_swz, mt, kt, lane_id);

      uint32_t b_frag[4];
      const int b_row = kt * 16 + (lane_id & 7) + (((lane_id >> 3) & 1) * 8);
      const int b_col = nt * 16 + ((lane_id >= 16) ? 8 : 0);
      ldmatrix_m8n8_x4_trans(b_frag, shared_u32_addr(B + b_row * LDB + b_col));

      mma_m16n8k16_bf16(c_frag0, a_frag, b_frag);
      mma_m16n8k16_bf16(c_frag1, a_frag, b_frag + 2);
    }

#pragma unroll
    for (int frag_idx = 0; frag_idx < 4; ++frag_idx) {
      const int row = mt * 16 + mma_accumulator_fragment_row(lane_id, frag_idx);
      const int col0 = nt * 16 + mma_accumulator_fragment_col(lane_id, frag_idx);
      const int col1 = col0 + 8;
      const float old0 = __bfloat162float(state_s[row * STATE_LD + col0]);
      const float old1 = __bfloat162float(state_s[row * STATE_LD + col1]);
      state_s[row * STATE_LD + col0] = __float2bfloat16_rn(g_last * old0 + c_frag0[frag_idx]);
      state_s[row * STATE_LD + col1] = __float2bfloat16_rn(g_last * old1 + c_frag1[frag_idx]);
    }
  }
}

template <int M, int N, int K>
__device__ void mma_gemm_bf16_bf16_f32_rm_b_col_swz(
    const bf16_t* A,
    const bf16_t* B_col_major,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;
  static_assert(K % 16 == 0, "K must be a multiple of 16");
  static_assert(N % 16 == 0, "N must be a multiple of 16");

  const int lane_id = threadIdx.x & 31;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

#pragma unroll
    for (int n8 = 0; n8 < 2; ++n8) {
      float c_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
      for (int kt = 0; kt < KT; ++kt) {
        uint32_t a_frag[4];
        uint32_t b_frag[2];

        const int a_row = mt * 16 + (lane_id & 15);
        const int a_col = kt * 16 + ((lane_id >> 4) * 8);
        ldmatrix_m8n8_x4(a_frag, swizzled_bf16_addr<K>(A, a_row, a_col));

        const int b_row = nt * 16 + n8 * 8 + (lane_id & 7);
        const int b_col = kt * 16 + ((lane_id >> 3) * 8);
        ldmatrix_m8n8_x2(b_frag, swizzled_bf16_addr<K>(B_col_major, b_row, b_col));

        mma_m16n8k16_bf16(c_frag, a_frag, b_frag);
      }

#pragma unroll
      for (int frag_idx = 0; frag_idx < 4; ++frag_idx) {
        const int row = mt * 16 + mma_accumulator_fragment_row(lane_id, frag_idx);
        const int col = nt * 16 + n8 * 8 + mma_accumulator_fragment_col(lane_id, frag_idx);
        C[row * N + col] = c_frag[frag_idx];
      }
    }
  }
}

template <int M, int N, int K, int LDB, int LDC>
__device__ void wmma_gemm_bf16_bf16_f32_rm_b_ld_c_ld(
    const bf16_t* A,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A + (mt * 16) * K + kt * 16, K);
      wmma::load_matrix_sync(b_frag, B + (kt * 16) * LDB + nt * 16, LDB);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * LDC + nt * 16, c_frag, LDC, wmma::mem_row_major);
  }
}

template <int M, int N, int K, int LDA, int LDC>
__device__ void wmma_gemm_bf16_bf16_f32_rm_a_ld_c_ld(
    const bf16_t* A,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A + (mt * 16) * LDA + kt * 16, LDA);
      wmma::load_matrix_sync(b_frag, B + (kt * 16) * N + nt * 16, N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * LDC + nt * 16, c_frag, LDC, wmma::mem_row_major);
  }
}

template <int M, int N, int K, int LDA, int LDB, int LDC>
__device__ void wmma_gemm_bf16_bf16_f32_rm_a_ld_b_ld_c_ld(
    const bf16_t* A,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A + (mt * 16) * LDA + kt * 16, LDA);
      wmma::load_matrix_sync(b_frag, B + (kt * 16) * LDB + nt * 16, LDB);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * LDC + nt * 16, c_frag, LDC, wmma::mem_row_major);
  }
}

template <int M, int N, int K, int LDB>
__device__ void wmma_gemm_bf16_bf16_f32_rm_b_ld(
    const bf16_t* A,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A + (mt * 16) * K + kt * 16, K);
      wmma::load_matrix_sync(b_frag, B + (kt * 16) * LDB + nt * 16, LDB);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * N + nt * 16, c_frag, N, wmma::mem_row_major);
  }
}

template <int M, int N, int K>
__device__ void wmma_gemm_bf16_bf16_f32_rm_a_col(
    const bf16_t* A_col_major,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A_col_major + (kt * 16) * M + mt * 16, M);
      wmma::load_matrix_sync(b_frag, B + (kt * 16) * N + nt * 16, N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * N + nt * 16, c_frag, N, wmma::mem_row_major);
  }
}

template <int M, int N, int K, int LDC>
__device__ void wmma_gemm_bf16_bf16_f32_rm_a_col_ldc(
    const bf16_t* A_col_major,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A_col_major + (kt * 16) * M + mt * 16, M);
      wmma::load_matrix_sync(b_frag, B + (kt * 16) * N + nt * 16, N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * LDC + nt * 16, c_frag, LDC, wmma::mem_row_major);
  }
}

template <int M, int N, int K, int LDA, int LDC>
__device__ void wmma_gemm_bf16_bf16_f32_rm_a_col_ld_a_ldc(
    const bf16_t* A_col_major,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A_col_major + (kt * 16) * LDA + mt * 16, LDA);
      wmma::load_matrix_sync(b_frag, B + (kt * 16) * N + nt * 16, N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * LDC + nt * 16, c_frag, LDC, wmma::mem_row_major);
  }
}

template <int M, int N, int K>
__device__ void wmma_gemm_bf16_bf16_f32_rm(
    const bf16_t* A,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile % NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A + (mt * 16) * K + kt * 16, K);
      wmma::load_matrix_sync(b_frag, B + (kt * 16) * N + nt * 16, N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * N + nt * 16, c_frag, N, wmma::mem_row_major);
  }
}

template <int M, int N, int K>
__device__ void wmma_gemm_bf16_bf16_f32_rm_a_ld(
    const bf16_t* A,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A + (mt * 16) * K + kt * 16, K);
      wmma::load_matrix_sync(b_frag, B + (kt * 16) * N + nt * 16, N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * N + nt * 16, c_frag, N, wmma::mem_row_major);
  }
}

template <int M, int N, int K, int LDA>
__device__ void wmma_gemm_bf16_bf16_f32_rm_a_ld(
    const bf16_t* A,
    const bf16_t* B,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile - mt * NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A + (mt * 16) * LDA + kt * 16, LDA);
      wmma::load_matrix_sync(b_frag, B + (kt * 16) * N + nt * 16, N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * N + nt * 16, c_frag, N, wmma::mem_row_major);
  }
}

template <int M, int N, int K>
__device__ void wmma_gemm_bf16_bf16_f32_rm_b_col(
    const bf16_t* A,
    const bf16_t* B_col_major,
    float* C,
    int warp_id,
    int num_warps) {
  constexpr int MT = M / 16;
  constexpr int NT = N / 16;
  constexpr int KT = K / 16;

  for (int tile = warp_id; tile < MT * NT; tile += num_warps) {
    const int mt = tile / NT;
    const int nt = tile % NT;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      wmma::load_matrix_sync(a_frag, A + (mt * 16) * K + kt * 16, K);
      wmma::load_matrix_sync(b_frag, B_col_major + (nt * 16) * K + kt * 16, K);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (mt * 16) * N + nt * 16, c_frag, N, wmma::mem_row_major);
  }
}

__global__ __launch_bounds__(PRE_THREADS, 1)
void gdn_precompute_a_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ beta,
    bf16_t* __restrict__ a_solve,
    bf16_t* __restrict__ qk_cache,
    int tokens,
    int n_chunks) {
  const int kh = blockIdx.x;
  const int seq = blockIdx.y;
  const int chunk_idx = blockIdx.z;
  const int chunk0 = chunk_idx * CHUNK;
  const int h0 = kh;
  const int h1 = kh + H_K;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int num_warps = blockDim.x >> 5;

  __shared__ __align__(16) bf16_t kh_s[CHUNK * D];
  __shared__ __align__(16) bf16_t qh_pre_s[CHUNK * D];
  __shared__ __align__(16) float kkt_s[CHUNK * CHUNK];
  __shared__ __align__(16) float a0_s[CHUNK * CHUNK];
  __shared__ __align__(16) float a1_s[CHUNK * CHUNK];
  __shared__ float beta0_s[CHUNK];
  __shared__ float beta1_s[CHUNK];

  for (int idx = tid; idx < CHUNK * (D / 2); idx += blockDim.x) {
    const int row = idx / (D / 2);
    const int col2 = (idx - row * (D / 2)) * 2;
    const int t = chunk0 + row;
    float2 kv = {0.0f, 0.0f};
    float2 qv = {0.0f, 0.0f};
    if (t < tokens) {
      const int64_t qk_off = ((static_cast<int64_t>(seq) * tokens + t) * H_K + kh) * D + col2;
      kv = *reinterpret_cast<const float2*>(k + qk_off);
      qv = *reinterpret_cast<const float2*>(q + qk_off);
    }
    *reinterpret_cast<__nv_bfloat162*>(kh_s + swizzled_bf16_index<D>(row, col2)) =
        __floats2bfloat162_rn(kv.x, kv.y);
    *reinterpret_cast<__nv_bfloat162*>(qh_pre_s + swizzled_bf16_index<D>(row, col2)) =
        __floats2bfloat162_rn(qv.x, qv.y);
  }
  if (tid < CHUNK) {
    const int t = chunk0 + tid;
    beta0_s[tid] = t < tokens ? beta[(static_cast<int64_t>(seq) * tokens + t) * H_V + h0] : 0.0f;
    beta1_s[tid] = t < tokens ? beta[(static_cast<int64_t>(seq) * tokens + t) * H_V + h1] : 0.0f;
  }
  __syncthreads();

  mma_gemm_bf16_bf16_f32_rm_b_col_swz<CHUNK, CHUNK, D>(kh_s, kh_s, kkt_s, warp_id, num_warps);
  __syncthreads();

  for (int idx = tid; idx < 2 * CHUNK * CHUNK; idx += blockDim.x) {
    const int local = idx % (CHUNK * CHUNK);
    const int row = local / CHUNK;
    const int col = local - row * CHUNK;
    const float init = row == col ? 1.0f : 0.0f;
    if (idx < CHUNK * CHUNK) {
      a0_s[local] = init;
    } else {
      a1_s[local] = init;
    }
  }
  __syncthreads();

#pragma unroll
  for (int row = 1; row < CHUNK; ++row) {
    for (int col = tid; col < row; col += blockDim.x) {
      float acc0 = 0.0f;
      float acc1 = 0.0f;
      const float b0 = beta0_s[row];
      const float b1 = beta1_s[row];
      for (int j = 0; j < row; ++j) {
        const float kkt = kkt_s[row * CHUNK + j];
        acc0 += (kkt * b0) * a0_s[j * CHUNK + col];
        acc1 += (kkt * b1) * a1_s[j * CHUNK + col];
      }
      a0_s[row * CHUNK + col] = -acc0;
      a1_s[row * CHUNK + col] = -acc1;
    }
    __syncthreads();
  }

  for (int idx = tid; idx < 2 * CHUNK * CHUNK; idx += blockDim.x) {
    const int plane = idx / (CHUNK * CHUNK);
    const int local = idx - plane * CHUNK * CHUNK;
    const int row = local / CHUNK;
    const int col = local - row * CHUNK;
    const int t = chunk0 + row;
    if (t < tokens) {
      const int h = plane == 0 ? h0 : h1;
      const float* a_s = plane == 0 ? a0_s : a1_s;
      const float bcol = plane == 0 ? beta0_s[col] : beta1_s[col];
      const float av = col <= row ? a_s[local] * bcol : 0.0f;
      a_solve[((static_cast<int64_t>(seq) * tokens + t) * H_V + h) * CHUNK + col] =
          __float2bfloat16_rn(av);
    }
  }

  __syncthreads();
  mma_gemm_bf16_bf16_f32_rm_b_col_swz<CHUNK, CHUNK, D>(qh_pre_s, kh_s, kkt_s, warp_id, num_warps);
  __syncthreads();
  const size_t qk_off = ((size_t(seq) * H_K + kh) * n_chunks + chunk_idx) * CHUNK * CHUNK;
  for (int idx = tid; idx < CHUNK * (CHUNK / 2); idx += blockDim.x) {
    const int row = idx / (CHUNK / 2);
    const int col2 = (idx - row * (CHUNK / 2)) * 2;
    *reinterpret_cast<__nv_bfloat162*>(qk_cache + qk_off + row * CHUNK + col2) =
        __floats2bfloat162_rn(kkt_s[row * CHUNK + col2], kkt_s[row * CHUNK + col2 + 1]);
  }
}

__global__ __launch_bounds__(THREADS, 2)
void gdn_bf16tc_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ state,
    const bf16_t* __restrict__ a_solve,
    const bf16_t* __restrict__ qk_cache,
    int n_chunks,
    float* __restrict__ out,
    float* __restrict__ final_state,
    char* __restrict__ workspace,
    int tokens,
    size_t bytes_per_head,
    float scale) {
  const int h = blockIdx.x;
  const int seq = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int num_warps = blockDim.x >> 5;
  const int kh = h % H_K;
  const int v0 = blockIdx.z * BLOCK_DV;

  char* ws = workspace + ((size_t(seq) * H_V + h) * (D / BLOCK_DV) + blockIdx.z) * bytes_per_head;

  bf16_t* qh      = reinterpret_cast<bf16_t*>(ws + TC_QH_OFF);
  bf16_t* uh      = reinterpret_cast<bf16_t*>(ws + TC_UH_OFF);

  float* state_out = final_state + (static_cast<int64_t>(seq) * H_V + h) * D * D;
  const float* state_in = state + (static_cast<int64_t>(seq) * H_V + h) * D * D;

  constexpr int STATE_LD = 40;
  __shared__ __align__(16) bf16_t state_s[D * STATE_LD];
  constexpr int KH_LD = 136;
  __shared__ __align__(16) bf16_t kh_s_main[CHUNK * KH_LD];
  __shared__ float g_prefix[CHUNK];
  __shared__ float g_inv[CHUNK];
  __shared__ float g_scaled[CHUNK];
  __shared__ float tail_beta;
  __shared__ float tail_qk;
  constexpr int PH_LD = 40;
  __shared__ __align__(16) bf16_t ph_s[CHUNK * PH_LD];
  constexpr int M_LD = 48;
  __shared__ __align__(16) float m_s[CHUNK * M_LD];
  constexpr int UPD_LD = 40;
  __shared__ __align__(16) float upd_s[D * UPD_LD];
  bf16_t* ph = ph_s;
  float* m = m_s;
  float* upd = upd_s;
  float* base_out = upd_s;
  bf16_t* sh = state_s;
  bf16_t* khm_swz = reinterpret_cast<bf16_t*>(upd_s);
  bf16_t* khm = kh_s_main;

  for (int idx = tid; idx < D * BLOCK_DV; idx += blockDim.x) {
    const int row = idx / BLOCK_DV;
    const int col = idx - row * BLOCK_DV;
    state_s[row * STATE_LD + col] = __float2bfloat16_rn(state_in[(v0 + col) * D + row]);
  }
  __syncthreads();

  for (int chunk0 = 0; chunk0 < tokens; chunk0 += CHUNK) {
    const int remaining = tokens - chunk0;
    const int actual = remaining < CHUNK ? remaining : CHUNK;
    const int chunk_idx = chunk0 / CHUNK;
    const size_t qk_cache_off = ((size_t(seq) * H_K + kh) * n_chunks + chunk_idx) * CHUNK * CHUNK;

    if (actual == 1) {
      if (tid < D) {
        const int64_t base_off = ((static_cast<int64_t>(seq) * tokens + chunk0) * H_K + kh) * D + tid;
        ph[tid] = __float2bfloat16_rn(q[base_off]);
        khm[tid] = __float2bfloat16_rn(k[base_off]);
      }
      if (tid == 0) {
        const float prefix = g[(static_cast<int64_t>(seq) * tokens + chunk0) * H_V + h];
        const float gp = exp2f(prefix * LOG2E);
        g_prefix[0] = gp;
        g_scaled[0] = scale * gp;
        tail_beta = beta[(static_cast<int64_t>(seq) * tokens + chunk0) * H_V + h];
      }
      __syncthreads();

      if (warp_id == 0) {
        float qk_acc = 0.0f;
        const int lane = tid & 31;
        for (int d = lane; d < D; d += 32) {
          qk_acc += __bfloat162float(ph[d]) * __bfloat162float(khm[d]);
        }
        qk_acc = warp_sum(qk_acc);
        if (lane == 0) {
          tail_qk = qk_acc;
        }
      }

#pragma unroll
      for (int pass = 0; pass < 2; ++pass) {
        const int col = warp_id + pass * WARPS;
        if (col < BLOCK_DV) {
          const int lane = tid & 31;
          float k_acc = 0.0f;
          float q_acc = 0.0f;
          for (int d = lane; d < D; d += 32) {
            const float sv = __bfloat162float(state_s[d * STATE_LD + col]);
            k_acc += __bfloat162float(khm[d]) * sv;
            q_acc += __bfloat162float(ph[d]) * sv;
          }
          k_acc = warp_sum(k_acc);
          q_acc = warp_sum(q_acc);
          if (lane == 0) {
            m[col] = k_acc;
            base_out[col] = g_scaled[0] * q_acc;
          }
        }
      }
      __syncthreads();

      for (int col = tid; col < BLOCK_DV; col += blockDim.x) {
        const int64_t v_off = ((static_cast<int64_t>(seq) * tokens + chunk0) * H_V + h) * D + v0 + col;
        const float vd = tail_beta * (v[v_off] - g_prefix[0] * m[col]);
        uh[col] = __float2bfloat16_rn(vd);
        out[((static_cast<int64_t>(seq) * tokens + chunk0) * H_V + h) * D + v0 + col] =
            base_out[col] + (scale * tail_qk) * vd;
      }
      __syncthreads();

      for (int idx = tid; idx < D * BLOCK_DV; idx += blockDim.x) {
        const int row = idx / BLOCK_DV;
        const int col = idx - row * BLOCK_DV;
        const float oldv = __bfloat162float(state_s[row * STATE_LD + col]);
        const float kval = __bfloat162float(khm[row]);
        const float vd = __bfloat162float(uh[col]);
        state_s[row * STATE_LD + col] = __float2bfloat16_rn(g_prefix[0] * oldv + kval * vd);
      }
      __syncthreads();
      continue;
    }

    if (actual == CHUNK) {
      for (int idx = tid; idx < CHUNK * (D / 2); idx += blockDim.x) {
        const int row = idx / (D / 2);
        const int col2 = (idx - row * (D / 2)) * 2;
        const int t = chunk0 + row;
        const int64_t base_off = ((static_cast<int64_t>(seq) * tokens + t) * H_K + kh) * D + col2;
        const float2 qv = *reinterpret_cast<const float2*>(q + base_off);
        const float2 kv = *reinterpret_cast<const float2*>(k + base_off);
        const __nv_bfloat162 q_bf16 = __floats2bfloat162_rn(qv.x, qv.y);
        const __nv_bfloat162 k_bf16 = __floats2bfloat162_rn(kv.x, kv.y);
        *reinterpret_cast<__nv_bfloat162*>(qh + row * D + col2) = q_bf16;
        *reinterpret_cast<__nv_bfloat162*>(khm + row * KH_LD + col2) = k_bf16;
        *reinterpret_cast<__nv_bfloat162*>(khm_swz + swizzled_bf16_index<D>(row, col2)) = k_bf16;
      }
      if (tid < CHUNK) {
        float prefix = g[(static_cast<int64_t>(seq) * tokens + chunk0 + tid) * H_V + h];
        const int lane = tid & 31;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
          const float y = __shfl_up_sync(0xffffffff, prefix, offset);
          if (lane >= offset) {
            prefix += y;
          }
        }
        const float gp = exp2f(prefix * LOG2E);
        g_prefix[tid] = gp;
        g_inv[tid] = 1.0f / gp;
        g_scaled[tid] = scale * gp;
      }
    } else {
    for (int idx = tid; idx < CHUNK * (D / 2); idx += blockDim.x) {
      const int row = idx / (D / 2);
      const int col2 = (idx - row * (D / 2)) * 2;
      const int t = chunk0 + row;
      float q0 = 0.0f;
      float q1 = 0.0f;
      float k0 = 0.0f;
      float k1 = 0.0f;
      if (row < actual) {
        const int64_t base_off = ((static_cast<int64_t>(seq) * tokens + t) * H_K + kh) * D + col2;
        const float2 qv = *reinterpret_cast<const float2*>(q + base_off);
        const float2 kv = *reinterpret_cast<const float2*>(k + base_off);
        q0 = qv.x;
        q1 = qv.y;
        k0 = kv.x;
        k1 = kv.y;
      }
      const __nv_bfloat162 q_bf16 = __floats2bfloat162_rn(q0, q1);
      const __nv_bfloat162 k_bf16 = __floats2bfloat162_rn(k0, k1);
      *reinterpret_cast<__nv_bfloat162*>(qh + row * D + col2) = q_bf16;
      *reinterpret_cast<__nv_bfloat162*>(khm + row * KH_LD + col2) = k_bf16;
      *reinterpret_cast<__nv_bfloat162*>(khm_swz + swizzled_bf16_index<D>(row, col2)) = k_bf16;
    }

    if (tid < CHUNK) {
      float prefix = 0.0f;
      if (tid < actual) {
        prefix = g[(static_cast<int64_t>(seq) * tokens + chunk0 + tid) * H_V + h];
      }
      const int lane = tid & 31;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const float y = __shfl_up_sync(0xffffffff, prefix, offset);
        if (lane >= offset) {
          prefix += y;
        }
      }
      const float gp = exp2f(prefix * LOG2E);
      g_prefix[tid] = gp;
      g_inv[tid] = 1.0f / gp;
      g_scaled[tid] = scale * gp;
    }
    }
    __syncthreads();

    // W = V - G_t * (K @ S0).
    mma_gemm_bf16_bf16_f32_rm_smem_a_swz_ld_b_ld_c_ld<CHUNK, BLOCK_DV, D, D, STATE_LD, M_LD>(khm_swz, sh, m, warp_id, num_warps);
    __syncthreads();

    if (actual == CHUNK) {
      for (int idx = tid; idx < CHUNK * (BLOCK_DV / 2); idx += blockDim.x) {
        const int row = idx / (BLOCK_DV / 2);
        const int col2 = (idx - row * (BLOCK_DV / 2)) * 2;
        const int64_t v_off = ((static_cast<int64_t>(seq) * tokens + chunk0 + row) * H_V + h) * D + v0 + col2;
        const float2 vv = *reinterpret_cast<const float2*>(v + v_off);
        const float gp = g_prefix[row];
        const float w0 = vv.x - gp * m[row * M_LD + col2];
        const float w1 = vv.y - gp * m[row * M_LD + col2 + 1];
        *reinterpret_cast<__nv_bfloat162*>(uh + row * BLOCK_DV + col2) = __floats2bfloat162_rn(w0, w1);
      }
    } else {
    for (int idx = tid; idx < CHUNK * (BLOCK_DV / 2); idx += blockDim.x) {
      const int row = idx / (BLOCK_DV / 2);
      const int col2 = (idx - row * (BLOCK_DV / 2)) * 2;
      float w0 = 0.0f;
      float w1 = 0.0f;
      if (row < actual) {
        const int64_t v_off = ((static_cast<int64_t>(seq) * tokens + chunk0 + row) * H_V + h) * D + v0 + col2;
        const float2 vv = *reinterpret_cast<const float2*>(v + v_off);
        const float gp = g_prefix[row];
        w0 = vv.x - gp * m[row * M_LD + col2];
        w1 = vv.y - gp * m[row * M_LD + col2 + 1];
      }
      *reinterpret_cast<__nv_bfloat162*>(uh + row * BLOCK_DV + col2) = __floats2bfloat162_rn(w0, w1);
    }
    }
    __syncthreads();

    // Vd = (A * G_i/G_j * beta_j) @ W.
    if (actual == CHUNK) {
      for (int idx = tid; idx < CHUNK * CHUNK; idx += blockDim.x) {
        const int row = idx / CHUNK;
        const int col = idx - row * CHUNK;
        float av = 0.0f;
        if (col <= row) {
          const int t = chunk0 + row;
          av = __bfloat162float(a_solve[((static_cast<int64_t>(seq) * tokens + t) * H_V + h) * CHUNK + col]);
          av *= g_prefix[row] * g_inv[col];
        }
        ph[row * PH_LD + col] = __float2bfloat16_rn(av);
      }
    } else {
    for (int idx = tid; idx < CHUNK * CHUNK; idx += blockDim.x) {
      const int row = idx / CHUNK;
      const int col = idx - row * CHUNK;
      float av = 0.0f;
      if (row < actual && col <= row) {
        const int t = chunk0 + row;
        av = __bfloat162float(a_solve[((static_cast<int64_t>(seq) * tokens + t) * H_V + h) * CHUNK + col]);
        av *= g_prefix[row] * g_inv[col];
      }
      ph[row * PH_LD + col] = __float2bfloat16_rn(av);
    }
    }
    __syncthreads();

    wmma_gemm_bf16_bf16_f32_rm_a_ld_c_ld<CHUNK, BLOCK_DV, CHUNK, PH_LD, M_LD>(ph, uh, m, warp_id, num_warps);
    __syncthreads();

    if (actual == CHUNK) {
      for (int idx = tid; idx < CHUNK * BLOCK_DV; idx += blockDim.x) {
        const int row = idx / BLOCK_DV;
        const int col = idx - row * BLOCK_DV;
        uh[idx] = __float2bfloat16_rn(m[row * M_LD + col]);
      }
    } else {
    for (int idx = tid; idx < CHUNK * BLOCK_DV; idx += blockDim.x) {
      const int row = idx / BLOCK_DV;
      const int col = idx - row * BLOCK_DV;
      const float vd = row < actual ? m[row * M_LD + col] : 0.0f;
      uh[idx] = __float2bfloat16_rn(vd);
    }
    }
    __syncthreads();

    // Base output: scale * G_t * (Q @ S0).
    wmma_gemm_bf16_bf16_f32_rm_b_ld_c_ld<CHUNK, BLOCK_DV, D, STATE_LD, M_LD>(qh, sh, m, warp_id, num_warps);
    __syncthreads();
    for (int idx = tid; idx < actual * (BLOCK_DV / 2); idx += blockDim.x) {
      const int row = idx / (BLOCK_DV / 2);
      const int col2 = (idx - row * (BLOCK_DV / 2)) * 2;
      const float gs = g_scaled[row];
      float2 bv;
      bv.x = gs * m[row * M_LD + col2];
      bv.y = gs * m[row * M_LD + col2 + 1];
      *reinterpret_cast<float2*>(base_out + row * BLOCK_DV + col2) = bv;
    }

    // P = tril(scale * G_i/G_j * QK_ij), with QK precomputed once per K head/chunk.
    if (actual == CHUNK) {
      for (int idx = tid; idx < CHUNK * (CHUNK / 2); idx += blockDim.x) {
        const int row = idx / (CHUNK / 2);
        const int col2 = (idx - row * (CHUNK / 2)) * 2;
        float p0 = 0.0f;
        float p1 = 0.0f;
        if (col2 <= row) {
          const __nv_bfloat162 qkv =
              *reinterpret_cast<const __nv_bfloat162*>(qk_cache + qk_cache_off + row * CHUNK + col2);
          const float2 qkf = __bfloat1622float2(qkv);
          const float gs = g_scaled[row];
          p0 = gs * g_inv[col2] * qkf.x;
          if (col2 + 1 <= row) {
            p1 = gs * g_inv[col2 + 1] * qkf.y;
          }
        }
        *reinterpret_cast<__nv_bfloat162*>(ph + row * PH_LD + col2) = __floats2bfloat162_rn(p0, p1);
      }
    } else {
    for (int idx = tid; idx < CHUNK * (CHUNK / 2); idx += blockDim.x) {
      const int row = idx / (CHUNK / 2);
      const int col2 = (idx - row * (CHUNK / 2)) * 2;
      float p0 = 0.0f;
      float p1 = 0.0f;
      if (row < actual && col2 <= row) {
        const __nv_bfloat162 qkv =
            *reinterpret_cast<const __nv_bfloat162*>(qk_cache + qk_cache_off + row * CHUNK + col2);
        const float2 qkf = __bfloat1622float2(qkv);
        const float gs = g_scaled[row];
        p0 = gs * g_inv[col2] * qkf.x;
        if (col2 + 1 <= row) {
          p1 = gs * g_inv[col2 + 1] * qkf.y;
        }
      }
      *reinterpret_cast<__nv_bfloat162*>(ph + row * PH_LD + col2) = __floats2bfloat162_rn(p0, p1);
    }
    }
    __syncthreads();

    // Add P @ U to output.
    wmma_gemm_bf16_bf16_f32_rm_a_ld_c_ld<CHUNK, BLOCK_DV, CHUNK, PH_LD, M_LD>(ph, uh, m, warp_id, num_warps);
    __syncthreads();
    for (int idx = tid; idx < actual * (BLOCK_DV / 2); idx += blockDim.x) {
      const int row = idx / (BLOCK_DV / 2);
      const int col2 = (idx - row * (BLOCK_DV / 2)) * 2;
      float2 ov;
      ov.x = base_out[row * BLOCK_DV + col2] + m[row * M_LD + col2];
      ov.y = base_out[row * BLOCK_DV + col2 + 1] + m[row * M_LD + col2 + 1];
      *reinterpret_cast<float2*>(out + ((static_cast<int64_t>(seq) * tokens + chunk0 + row) * H_V + h) * D + v0 + col2) = ov;
    }

    // S = G_last*S0 + K_restored^T @ U.
    const float g_last = actual == CHUNK ? g_prefix[CHUNK - 1] : (actual > 0 ? g_prefix[actual - 1] : 1.0f);
    if (actual == CHUNK) {
      for (int idx = tid; idx < CHUNK * BLOCK_DV; idx += blockDim.x) {
        const int row = idx / BLOCK_DV;
        const int col = idx - row * BLOCK_DV;
        const float vd = __bfloat162float(uh[idx]);
        ph[row * BLOCK_DV + col] = __float2bfloat16_rn((g_last * g_inv[row]) * vd);
      }
    } else {
      for (int idx = tid; idx < CHUNK * BLOCK_DV; idx += blockDim.x) {
        const int row = idx / BLOCK_DV;
        const int col = idx - row * BLOCK_DV;
        const float vd = row < actual ? __bfloat162float(uh[idx]) : 0.0f;
        ph[row * BLOCK_DV + col] = __float2bfloat16_rn(row < actual ? (g_last * g_inv[row]) * vd : 0.0f);
      }
    }
    __syncthreads();

    mma_ktrans_swz_update_state_bf16<D, BLOCK_DV, CHUNK, BLOCK_DV, STATE_LD, KH_LD>(
        khm, ph, state_s, g_last, warp_id, num_warps);
    __syncthreads();
  }

  for (int idx = tid; idx < BLOCK_DV * D; idx += blockDim.x) {
    const int col = idx / D;
    const int row = idx - col * D;
    state_out[(v0 + col) * D + row] = __bfloat162float(state_s[row * STATE_LD + col]);
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

std::vector<torch::Tensor> qwen35moe_gdn_prefill_bf16tc(
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
  auto workspace = torch::empty({static_cast<int64_t>(bytes_per_head() * H_V * (D / BLOCK_DV) * batch)},
                                q.options().dtype(torch::kUInt8));
  const int n_chunks = (tokens + CHUNK - 1) / CHUNK;
  auto a_solve = torch::empty(
      {batch, tokens, H_V, CHUNK},
      q.options().dtype(torch::kBFloat16));
  auto qk_cache = torch::empty(
      {batch, H_K, n_chunks, CHUNK, CHUNK},
      q.options().dtype(torch::kBFloat16));

  const auto stream = at::cuda::getCurrentCUDAStream();

  const dim3 pre_block(PRE_THREADS, 1, 1);
  const dim3 pre_grid(H_K, batch, n_chunks);
  gdn_precompute_a_kernel<<<pre_grid, pre_block, 0, stream>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      beta.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(a_solve.data_ptr<at::BFloat16>()),
      reinterpret_cast<bf16_t*>(qk_cache.data_ptr<at::BFloat16>()),
      tokens,
      n_chunks);
  const cudaError_t pre_err = cudaGetLastError();
  TORCH_CHECK(pre_err == cudaSuccess, "qwen35moe_gdn_prefill A precompute launch failed: ", cudaGetErrorString(pre_err));

  const dim3 block(THREADS, 1, 1);
  const dim3 grid(H_V, batch, D / BLOCK_DV);
  gdn_bf16tc_kernel<<<grid, block, 0, stream>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      g.data_ptr<float>(),
      beta.data_ptr<float>(),
      state.data_ptr<float>(),
      reinterpret_cast<const bf16_t*>(a_solve.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16_t*>(qk_cache.data_ptr<at::BFloat16>()),
      n_chunks,
      out.data_ptr<float>(),
      final_state.data_ptr<float>(),
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      tokens,
      bytes_per_head(),
      static_cast<float>(scale));

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "qwen35moe_gdn_prefill_bf16tc launch failed: ", cudaGetErrorString(err));
  return {out, final_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qwen35moe_gdn_prefill_bf16tc", &qwen35moe_gdn_prefill_bf16tc,
        "Qwen3.5 MoE GDN chunked BF16 Tensor Core prefill");
}
