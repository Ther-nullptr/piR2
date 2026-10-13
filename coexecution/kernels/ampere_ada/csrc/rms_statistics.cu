// Instantiate the installed PyTorch v2.7.1 Reduce.cuh instead of copying its
// reduction tree. BF16 input and FP32 output are real, distinct tensors.
// Square and accumulator addition must remain separate FP32 rounding steps.
#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <string>

#ifndef PIR2_RMS_SOURCE_ID
#error "RMS statistics require a verified source identity"
#endif

namespace {

struct SquareMeanOps {
  float factor;

  inline C10_DEVICE float reduce(float accumulator, at::BFloat16 input,
                                 int64_t /*index*/) const {
    const float value = static_cast<float>(input);
    const float squared = __fmul_rn(value, value);
    return __fadd_rn(accumulator, squared);
  }

  inline C10_DEVICE float combine(float left, float right) const {
    return __fadd_rn(left, right);
  }

  inline C10_DEVICE float project(float accumulator) const {
    return __fmul_rn(accumulator, factor);
  }

  static C10_DEVICE float translate_idx(float accumulator,
                                        int64_t /*base_index*/) {
    return accumulator;
  }

  inline C10_DEVICE float warp_shfl_down(float value, int offset) const {
    return WARP_SHFL_DOWN(value, offset);
  }
};

at::Tensor mean_square(const at::Tensor &x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() >= 1 &&
                  !x.requires_grad(),
              "RMS statistics require CUDA BF16 rows without autograd");
  const int64_t width = x.size(-1);
  TORCH_CHECK(x.is_contiguous() && width >= 128 && width <= 8192 &&
                  width % 4 == 0 && x.numel() > 0 &&
                  x.numel() <= std::numeric_limits<int32_t>::max() / 4 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0,
              "RMS statistics require aligned, bounded contiguous BF16 rows");

  c10::cuda::CUDAGuard guard(x.device());
  const auto *properties = at::cuda::getDeviceProperties(x.get_device());
  TORCH_CHECK(properties->major == 8 && properties->minor == 9,
              "Native RMS statistics require CUDA capability 8.9 (Ada)");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  c10::cuda::CUDACachingAllocator::recordStream(x.storage().data_ptr(), stream);
  auto shape = x.sizes().vec();
  shape.back() = 1;
  auto variance = at::empty(shape, x.options().dtype(at::kFloat));
  auto iterator = at::TensorIterator::reduce_op(variance, x);
  TORCH_CHECK(iterator.can_use_32bit_indexing(),
              "RMS statistics require unsplit iterator indexing");
  // Match MeanKernel's FP32 host factor expression, rather than dividing the
  // reduced sum by width. The template explicitly retains native vt0=4 and
  // input_vec_size=4 for scalar and vectorized reduction geometries.
  const float factor = float(iterator.num_output_elements()) / iterator.numel();
  at::native::gpu_reduce_kernel<at::BFloat16, float, 4, 4>(
      iterator, SquareMeanOps{factor}, 0.0f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return variance;
}

std::string source_identity() { return PIR2_RMS_SOURCE_ID; }

} // namespace

TORCH_LIBRARY(pir2_rms_statistics, m) {
  m.def("source_identity() -> str", &source_identity);
  m.def("mean_square(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(pir2_rms_statistics, CUDA, m) {
  m.impl("mean_square", &mean_square);
}
