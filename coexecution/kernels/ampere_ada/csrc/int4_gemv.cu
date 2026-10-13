// Exact single-row packed INT4 GEMV. Each warp computes one output channel,
// retaining integer dot products until the reference FP32/BF16 epilogue.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>

#ifndef PIR2_GEMV_SOURCE_ID
#error "gemv requires a verified source identity"
#endif

namespace {

constexpr int WarpsPerBlock = 8;
constexpr int WarpSize = 32;

__device__ __forceinline__ int signed_nibbles(uint32_t packed, int shift) {
  uint32_t nibbles = (packed >> shift) & 0x0f0f0f0fu;
  // Independent byte subtraction is essential: a scalar word subtraction
  // would propagate a borrow into neighboring signed INT4 values.
  return static_cast<int>(__vsub4(nibbles ^ 0x08080808u, 0x08080808u));
}

template <bool AddBias>
__global__ void int4_gemv(const uint32_t *__restrict__ activation,
                          const float *__restrict__ activation_scale,
                          const uint32_t *__restrict__ weight,
                          const float *__restrict__ weight_scale,
                          const __nv_bfloat16 *__restrict__ bias,
                          __nv_bfloat16 *__restrict__ output, int64_t n,
                          int words) {
  int lane = threadIdx.x % WarpSize;
  int64_t channel =
      int64_t(blockIdx.x) * WarpsPerBlock + threadIdx.x / WarpSize;
  if (channel >= n)
    return;
  int accumulator = 0;
  for (int index = lane; index < words; index += WarpSize) {
    uint32_t a = activation[index];
    uint32_t b = weight[channel * words + index];
    accumulator =
        __dp4a(signed_nibbles(a, 0), signed_nibbles(b, 0), accumulator);
    accumulator =
        __dp4a(signed_nibbles(a, 4), signed_nibbles(b, 4), accumulator);
  }
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2)
    accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
  if (lane == 0) {
    float value = __fmul_rn(__int2float_rn(accumulator), activation_scale[0]);
    value = __fmul_rn(value, weight_scale[channel]);
    if constexpr (AddBias)
      value = __fadd_rn(value, bias ? __bfloat162float(bias[channel]) : 0.0f);
    output[channel] = __float2bfloat16_rn(value);
  }
}

at::Tensor packed_int4_gemv(const at::Tensor &a, const at::Tensor &sa,
                            const at::Tensor &b, const at::Tensor &sb,
                            const c10::optional<at::Tensor> &bias, int64_t bits,
                            bool biasless) {
  TORCH_CHECK(bits == 4, "INT4 GEMV requires bits=4");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && sa.is_cuda() && sb.is_cuda(),
              "CUDA tensors required");
  TORCH_CHECK(a.device() == b.device() && a.device() == sa.device() &&
                  a.device() == sb.device(),
              "Devices must match");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) == 1 &&
                  a.size(1) == b.size(1),
              "Invalid packed single-row matrix shapes");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && sa.is_contiguous() &&
                  sb.is_contiguous(),
              "Contiguous packed buffers required");
  TORCH_CHECK(a.scalar_type() == at::kByte && b.scalar_type() == at::kByte,
              "INT4 GEMV requires UINT8 packed storage");
  TORCH_CHECK(sa.scalar_type() == at::kFloat &&
                  sb.scalar_type() == at::kFloat && sa.dim() == 1 &&
                  sb.dim() == 1 && sa.size(0) == 1 && sb.size(0) == b.size(0),
              "Invalid scale buffers");
  int64_t k = a.size(1) * 2;
  int64_t n = b.size(0);
  TORCH_CHECK(n > 0 && n % 8 == 0 && k >= 128 && k <= 8192 && k % 128 == 0,
              "INT4 GEMV requires N%8=0 and 128<=K<=8192 with K%128=0");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(b.data_ptr()) % 4 == 0,
              "Packed matrices require 4-byte pointer alignment");
  if (bias)
    TORCH_CHECK(
        bias->device() == a.device() && bias->scalar_type() == at::kBFloat16 &&
            bias->is_contiguous() && bias->dim() == 1 && bias->size(0) == n,
        "Invalid BF16 bias");

  c10::cuda::CUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  for (const auto *tensor : {&a, &sa, &b, &sb})
    c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(),
                                                  stream);
  if (bias)
    c10::cuda::CUDACachingAllocator::recordStream(bias->storage().data_ptr(),
                                                  stream);
  auto output = at::empty({1, n}, a.options().dtype(at::kBFloat16));
  const auto *activation = reinterpret_cast<const uint32_t *>(a.data_ptr());
  const auto *weight = reinterpret_cast<const uint32_t *>(b.data_ptr());
  const auto *bias_ptr =
      bias ? reinterpret_cast<const __nv_bfloat16 *>(bias->data_ptr())
           : nullptr;
  auto *output_ptr = reinterpret_cast<__nv_bfloat16 *>(output.data_ptr());
  dim3 grid(n / WarpsPerBlock), block(WarpsPerBlock * WarpSize);
  if (bias || !biasless)
    int4_gemv<true><<<grid, block, 0, stream>>>(
        activation, sa.data_ptr<float>(), weight, sb.data_ptr<float>(),
        bias_ptr, output_ptr, n, k / 8);
  else
    int4_gemv<false><<<grid, block, 0, stream>>>(
        activation, sa.data_ptr<float>(), weight, sb.data_ptr<float>(),
        bias_ptr, output_ptr, n, k / 8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace

TORCH_LIBRARY(pir2_int4_gemv, m) {
  m.def("source_identity() -> str",
        []() { return std::string(PIR2_GEMV_SOURCE_ID); });
  m.def("gemv(Tensor a, Tensor sa, Tensor b, Tensor sb, Tensor? bias, "
        "int bits, bool biasless) -> Tensor");
}
TORCH_LIBRARY_IMPL(pir2_int4_gemv, CUDA, m) {
  m.impl("gemv", &packed_int4_gemv);
}
