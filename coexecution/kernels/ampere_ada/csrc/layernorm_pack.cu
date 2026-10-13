// Native Welford statistics adapted from PyTorch v2.7.1's
// aten/src/ATen/native/cuda/layer_norm_kernel.cu (BSD-3-Clause).
// Copyright (c) 2016-present, Facebook Inc. and PyTorch contributors.
// Preserve its iteration, merge expressions/order and default NVCC FMA
// behavior. See THIRD_PARTY_NOTICES.md for the pinned source and full upstream
// license. Integer packing implements the shared symmetric-row-v1 activation
// format.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

#ifndef PIR2_NORM_SOURCE_ID
#error "norm requires a verified source identity"
#endif

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarps = 4;
constexpr int kVectorSize = 4;

template <typename T, int Size> struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
};

struct WelfordDataLN {
  float mean;
  float sigma2;
  float count;
  __host__ __device__ WelfordDataLN() : mean(0.f), sigma2(0.f), count(0.f) {}
  __host__ __device__ WelfordDataLN(float mean, float sigma2, float count)
      : mean(mean), sigma2(sigma2), count(count) {}
};

template <typename U>
__device__ WelfordDataLN cuWelfordOnlineSum(const U val,
                                            const WelfordDataLN &curr_sum) {
  U delta = val - curr_sum.mean;
  U new_count = curr_sum.count + 1.f;
  U new_mean = curr_sum.mean + delta * (1.f / new_count);
  return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
}

__device__ WelfordDataLN cuWelfordCombine(const WelfordDataLN dataB,
                                          const WelfordDataLN dataA) {
  using U = decltype(dataB.count);
  U delta = dataB.mean - dataA.mean;
  U count = dataA.count + dataB.count;
  U mean, sigma2;
  if (count > decltype(dataB.count){0}) {
    auto coef = 1.f / count;
    auto nA = dataA.count * coef;
    auto nB = dataB.count * coef;
    mean = nA * dataA.mean + nB * dataB.mean;
    sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
  } else {
    mean = U(0);
    sigma2 = U(0);
  }
  return {mean, sigma2, count};
}

// The following reduction deliberately retains PyTorch's argument order:
// cuWelfordCombine(wd, wdB) passes the current value as dataB, not dataA.
__device__ WelfordDataLN compute_stats(const at::BFloat16 *__restrict__ x,
                                       const int width, float *buf) {
  using Vector = AlignedVector<at::BFloat16, kVectorSize>;
  const Vector *x_vec = reinterpret_cast<const Vector *>(x);
  const int threads = blockDim.x * blockDim.y;
  const int thread = threadIdx.x + threadIdx.y * blockDim.x;
  const int vectors = width / kVectorSize;
  WelfordDataLN wd(0.f, 0.f, 0.f);
  for (int i = thread; i < vectors; i += threads) {
    Vector data = x_vec[i];
#pragma unroll
    for (int j = 0; j < kVectorSize; j++) {
      wd = cuWelfordOnlineSum(static_cast<float>(data.val[j]), wd);
    }
  }
  for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
    WelfordDataLN wdB{__shfl_down_sync(0xffffffff, wd.mean, offset),
                      __shfl_down_sync(0xffffffff, wd.sigma2, offset),
                      __shfl_down_sync(0xffffffff, wd.count, offset)};
    wd = cuWelfordCombine(wd, wdB);
  }
  float *meansigmabuf = buf;
  float *countbuf = buf + blockDim.y;
  for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
    if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
      const int write_y = threadIdx.y - offset;
      meansigmabuf[2 * write_y] = wd.mean;
      meansigmabuf[2 * write_y + 1] = wd.sigma2;
      countbuf[write_y] = wd.count;
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y < offset) {
      WelfordDataLN wdB{meansigmabuf[2 * threadIdx.y],
                        meansigmabuf[2 * threadIdx.y + 1],
                        countbuf[threadIdx.y]};
      wd = cuWelfordCombine(wd, wdB);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    meansigmabuf[0] = wd.mean;
    meansigmabuf[1] = wd.sigma2 / float(width);
  }
  __syncthreads();
  return WelfordDataLN{meansigmabuf[0], meansigmabuf[1], 0.f};
}

__device__ float round_bf16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

template <int Bits, bool Modulated>
__global__ void
layernorm_pack_kernel(const at::BFloat16 *__restrict__ x,
                      const at::BFloat16 *__restrict__ condition_scale,
                      const at::BFloat16 *__restrict__ condition_shift,
                      unsigned char *__restrict__ q, float *__restrict__ scales,
                      int width, int pad, float eps, int64_t tokens,
                      bool shared_conditions, int64_t scale_batch_stride,
                      int64_t scale_token_stride, int64_t shift_batch_stride,
                      int64_t shift_token_stride) {
  extern __shared__ float shared[];
  // Keep statistics separate: every thread must read them before packing
  // writes.
  float *values = shared + 8;
  float *maxima = values + pad;
  const int64_t row = blockIdx.x;
  const int thread = threadIdx.x + threadIdx.y * blockDim.x;
  const int threads = blockDim.x * blockDim.y;
  const at::BFloat16 *block_row = x + row * width;
  WelfordDataLN wd = compute_stats(block_row, width, shared);
  const float rstd = c10::cuda::compat::rsqrt(wd.sigma2 + eps);
  int64_t scale_offset = 0, shift_offset = 0;
  if constexpr (Modulated) {
    const int64_t batch = row / tokens;
    const int64_t token = shared_conditions ? 0 : row % tokens;
    scale_offset = batch * scale_batch_stride + token * scale_token_stride;
    shift_offset = batch * shift_batch_stride + token * shift_token_stride;
  }
  using Vector = AlignedVector<at::BFloat16, kVectorSize>;
  const Vector *x_vec = reinterpret_cast<const Vector *>(block_row);
  float maximum = 0.f;
  for (int i = thread; i < width / kVectorSize; i += threads) {
    Vector data = x_vec[i];
#pragma unroll
    for (int j = 0; j < kVectorSize; j++) {
      const int feature = i * kVectorSize + j;
      float value =
          round_bf16(rstd * (static_cast<float>(data.val[j]) - wd.mean));
      if constexpr (Modulated) {
        const float scale =
            static_cast<float>(condition_scale[scale_offset + feature]);
        const float shift =
            static_cast<float>(condition_shift[shift_offset + feature]);
        // RN intrinsics prevent FMA across the distinct upstream BF16
        // boundaries.
        const float factor = round_bf16(__fadd_rn(1.f, scale));
        const float product = round_bf16(__fmul_rn(value, factor));
        value = round_bf16(__fadd_rn(product, shift));
      }
      values[feature] = value;
      maximum = fmaxf(maximum, fabsf(value));
    }
  }
  for (int i = width + thread; i < pad; i += threads)
    values[i] = 0.f;
  for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
  if (threadIdx.x == 0)
    maxima[threadIdx.y] = maximum;
  __syncthreads();
  constexpr int limit = (1 << (Bits - 1)) - 1;
  if (threadIdx.y == 0) {
    maximum = threadIdx.x < kWarps ? maxima[threadIdx.x] : 0.f;
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    if (threadIdx.x == 0) {
      float quant_scale = maximum > 0.f ? maximum * (1.f / limit) : 1.f;
      scales[row] = quant_scale;
      maxima[0] = quant_scale;
    }
  }
  __syncthreads();
  const float quant_scale = maxima[0];
  if constexpr (Bits == 8) {
    for (int i = thread; i < pad; i += threads) {
      int value = __float2int_rn(__fdiv_rn(values[i], quant_scale));
      q[row * pad + i] =
          static_cast<unsigned char>(max(-limit, min(limit, value)));
    }
  } else {
    for (int i = thread; i < pad / 2; i += threads) {
      int lo = max(-limit, min(limit, __float2int_rn(__fdiv_rn(values[2 * i],
                                                               quant_scale))));
      int hi = max(-limit, min(limit, __float2int_rn(__fdiv_rn(
                                          values[2 * i + 1], quant_scale))));
      q[row * (pad / 2) + i] = (lo & 15) | ((hi & 15) << 4);
    }
  }
}

std::tuple<at::Tensor, at::Tensor>
prepare_layernorm(const at::Tensor &x, int64_t bits, double eps,
                  const c10::optional<at::Tensor> &condition_scale,
                  const c10::optional<at::Tensor> &condition_shift) {
  TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == at::kBFloat16 &&
          (x.dim() == 2 || x.dim() == 3) && !x.requires_grad(),
      "LayerNorm producer requires CUDA BF16 2D/3D rows without autograd");
  TORCH_CHECK(bits == 8 || bits == 4, "Invalid LayerNorm packing precision");
  TORCH_CHECK(std::isfinite(eps) && eps > 0, "Invalid LayerNorm epsilon");
  TORCH_CHECK(
      x.is_contiguous() && x.size(-1) >= 128 && x.size(-1) <= 8192 &&
          x.size(-1) % kVectorSize == 0 &&
          reinterpret_cast<uintptr_t>(x.data_ptr()) %
                  (kVectorSize * sizeof(at::BFloat16)) ==
              0,
      "LayerNorm producer requires aligned contiguous native-vector rows");
  TORCH_CHECK(condition_scale.has_value() == condition_shift.has_value(),
              "LayerNorm modulation requires both scale and shift");
  if (condition_scale) {
    TORCH_CHECK(x.dim() == 3, "LayerNorm modulation requires B,T,D input");
    for (const auto *condition : {&*condition_scale, &*condition_shift}) {
      TORCH_CHECK(
          condition->is_cuda() && condition->device() == x.device() &&
              condition->scalar_type() == at::kBFloat16 &&
              condition->dim() == 3 && !condition->requires_grad() &&
              condition->size(0) == x.size(0) &&
              (condition->size(1) == 1 || condition->size(1) == x.size(1)) &&
              condition->size(2) == x.size(2) && condition->stride(2) == 1,
          "Invalid LayerNorm modulation condition buffers");
    }
    TORCH_CHECK(condition_scale->sizes() == condition_shift->sizes(),
                "LayerNorm modulation scale and shift shapes must match");
  }
  c10::cuda::CUDAGuard guard(x.device());
  const int width = x.size(-1);
  const int64_t rows = x.numel() / width;
  const int pad = (width + 127) / 128 * 128;
  auto data = at::empty({rows, bits == 8 ? pad : pad / 2},
                        x.options().dtype(bits == 8 ? at::kChar : at::kByte));
  auto scales = at::empty({rows}, x.options().dtype(at::kFloat));
  if (rows == 0)
    return {data, scales};
  TORCH_CHECK(rows <= at::cuda::getCurrentDeviceProperties()->maxGridSize[0],
              "LayerNorm row count exceeds the CUDA grid limit");
  const auto *scale_ptr =
      condition_scale ? condition_scale->data_ptr<at::BFloat16>() : nullptr;
  const auto *shift_ptr =
      condition_shift ? condition_shift->data_ptr<at::BFloat16>() : nullptr;
  const int64_t tokens = x.dim() == 3 ? x.size(1) : 1;
  const bool shared_conditions =
      condition_scale && condition_scale->size(1) == 1;
  const int64_t scale_batch_stride =
      condition_scale ? condition_scale->stride(0) : 0;
  const int64_t scale_token_stride =
      condition_scale ? condition_scale->stride(1) : 0;
  const int64_t shift_batch_stride =
      condition_shift ? condition_shift->stride(0) : 0;
  const int64_t shift_token_stride =
      condition_shift ? condition_shift->stride(1) : 0;
  const dim3 threads(kWarpSize, kWarps);
  const dim3 blocks(static_cast<unsigned int>(rows));
  const int shared_bytes = (8 + pad + kWarps) * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  auto launch = [&](auto bits_tag, auto modulation_tag) {
    constexpr int precision = decltype(bits_tag)::value;
    constexpr bool modulated = decltype(modulation_tag)::value;
    layernorm_pack_kernel<precision, modulated>
        <<<blocks, threads, shared_bytes, stream>>>(
            x.data_ptr<at::BFloat16>(), scale_ptr, shift_ptr,
            reinterpret_cast<unsigned char *>(data.data_ptr()),
            scales.data_ptr<float>(), width, pad, static_cast<float>(eps),
            tokens, shared_conditions, scale_batch_stride, scale_token_stride,
            shift_batch_stride, shift_token_stride);
  };
  if (bits == 8) {
    if (condition_scale)
      launch(std::integral_constant<int, 8>{}, std::true_type{});
    else
      launch(std::integral_constant<int, 8>{}, std::false_type{});
  } else {
    if (condition_scale)
      launch(std::integral_constant<int, 4>{}, std::true_type{});
    else
      launch(std::integral_constant<int, 4>{}, std::false_type{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {data, scales};
}

} // namespace

TORCH_LIBRARY(pir2_layernorm_pack, m) {
  m.def("source_identity() -> str",
        []() { return std::string(PIR2_NORM_SOURCE_ID); });
  m.def("prepare(Tensor x, int bits, float eps, Tensor? scale=None, "
        "Tensor? shift=None) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(pir2_layernorm_pack, CUDA, m) {
  m.impl("prepare", &prepare_layernorm);
}
