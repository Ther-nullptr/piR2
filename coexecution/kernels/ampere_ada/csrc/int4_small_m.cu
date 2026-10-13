// Small-M INT4 tile dispatch over the SHA256-pinned repository-owned CUTLASS
// run template. The loader creates this include only in Torch's local build
// cache; it contains no backend registrations or external tactic dispatch.
#include "pir2_int4_small_m_backend.cuh"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>

#include <cstdint>
#include <string>

namespace {

std::string small_m_source_identity() { return PIR2_SMALL_M_SOURCE_ID; }

template <bool WithBias>
at::Tensor dispatch_small_m_tile(const at::Tensor &a, const at::Tensor &sa,
                                 const at::Tensor &b, const at::Tensor &sb,
                                 const c10::optional<at::Tensor> &bias,
                                 int64_t tactic) {
  switch (tactic) {
  case 0:
    return run<WithBias, 4, 16, 64, 64, 16, 32, 3>(a, sa, b, sb, bias);
  case 1:
    return run<WithBias, 4, 16, 128, 64, 16, 64, 3>(a, sa, b, sb, bias);
  case 2:
    return run<WithBias, 4, 32, 64, 64, 16, 32, 3>(a, sa, b, sb, bias);
  case 3:
    return run<WithBias, 4, 32, 128, 64, 16, 64, 3>(a, sa, b, sb, bias);
  case 4:
    return run<WithBias, 4, 64, 64, 64, 32, 32, 3,
               cutlass::gemm::threadblock::ThreadblockSwizzleStreamK>(a, sa, b,
                                                                      sb, bias);
  case 5:
    return run<WithBias, 4, 16, 64, 64, 16, 32, 3,
               cutlass::gemm::threadblock::ThreadblockSwizzleStreamK>(a, sa, b,
                                                                      sb, bias);
  default:
    TORCH_CHECK(false, "Unknown small-M integer GEMM tactic");
  }
}

at::Tensor packed_int4_small_m(const at::Tensor &a, const at::Tensor &sa,
                               const at::Tensor &b, const at::Tensor &sb,
                               const c10::optional<at::Tensor> &bias,
                               int64_t bits, int64_t tactic, bool biasless) {
  TORCH_CHECK(bits == 4, "Small-M GEMM requires bits=4");
  TORCH_CHECK(tactic >= 0 && tactic <= 5,
              "Small-M GEMM tactic must be in 0..5");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && sa.is_cuda() && sb.is_cuda(),
              "CUDA tensors required");
  TORCH_CHECK(a.device() == b.device() && a.device() == sa.device() &&
                  a.device() == sb.device(),
              "Devices must match");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1),
              "Invalid packed matrix shapes");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && sa.is_contiguous() &&
                  sb.is_contiguous(),
              "Contiguous packed buffers required");
  TORCH_CHECK(a.scalar_type() == at::kByte && b.scalar_type() == at::kByte,
              "INT4 requires UINT8 packed storage");
  TORCH_CHECK(sa.scalar_type() == at::kFloat &&
                  sb.scalar_type() == at::kFloat && sa.dim() == 1 &&
                  sb.dim() == 1 && sa.size(0) == a.size(0) &&
                  sb.size(0) == b.size(0),
              "Invalid scale buffers");
  TORCH_CHECK(a.size(0) > 0 && a.size(0) <= 64 && b.size(0) > 0 &&
                  b.size(0) <= 8192 && b.size(0) % 8 == 0 && a.size(1) >= 64 &&
                  a.size(1) <= 4096 && a.size(1) % 64 == 0,
              "Small-M GEMM requires 1<=M<=64, padded 128<=K<=8192 with "
              "K%128=0 and padded 8<=N<=8192 with N%8=0");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(sb.data_ptr()) % 16 == 0,
              "Packed matrices and weight scales require 16-byte alignment");
  if (bias)
    TORCH_CHECK(bias->device() == a.device() &&
                    bias->scalar_type() == at::kBFloat16 &&
                    bias->is_contiguous() && bias->dim() == 1 &&
                    bias->size(0) == b.size(0) &&
                    reinterpret_cast<uintptr_t>(bias->data_ptr()) % 16 == 0,
                "Invalid aligned BF16 bias");

  c10::cuda::CUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  for (const auto *tensor : {&a, &sa, &b, &sb})
    c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(),
                                                  stream);
  if (bias)
    c10::cuda::CUDACachingAllocator::recordStream(bias->storage().data_ptr(),
                                                  stream);
  // This is the reference callback selection, including its implicit +0
  // operation when bias is absent and biasless specialization is disabled.
  auto output = bias || !biasless
                    ? dispatch_small_m_tile<true>(a, sa, b, sb, bias, tactic)
                    : dispatch_small_m_tile<false>(a, sa, b, sb, bias, tactic);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace

TORCH_LIBRARY(pir2_int4_small_m, m) {
  m.def("source_identity() -> str", &small_m_source_identity);
  m.def("gemm(Tensor a, Tensor sa, Tensor b, Tensor sb, Tensor? bias, "
        "int bits, int tactic, bool biasless) -> Tensor");
}
TORCH_LIBRARY_IMPL(pir2_int4_small_m, CUDA, m) {
  m.impl("gemm", &packed_int4_small_m);
}
