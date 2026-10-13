// Optional split-K over the pinned backend's packaged CUTLASS mainloop.
// Workspace slices and their reduction stay INT32 until the reference scaling
// order, optional bias, and the original round-to-nearest BF16 conversion.
// Optional residual addition follows that boundary, then rounds to BF16 again.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/epilogue/thread/conversion_op.h>
#include <cutlass/gemm/kernel/default_gemm_splitk_parallel.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <type_traits>

namespace {

std::string split_k_source_identity() { return PIR2_SPLIT_K_SOURCE_ID; }

template <int Splits, bool AddBias, bool AddResidual>
__global__ void reduce_scale(const int32_t *__restrict__ partials,
                             const float *__restrict__ activation_scale,
                             const float *__restrict__ weight_scale,
                             const __nv_bfloat16 *__restrict__ bias,
                             const __nv_bfloat16 *__restrict__ residual,
                             __nv_bfloat16 *__restrict__ output, int n,
                             int elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  int32_t accumulator = 0;
#pragma unroll
  for (int split = 0; split < Splits; ++split)
    accumulator += partials[split * elements + index];
  float value =
      __fmul_rn(__int2float_rn(accumulator), activation_scale[index / n]);
  value = __fmul_rn(value, weight_scale[index % n]);
  if constexpr (AddBias)
    value = __fadd_rn(value, bias ? __bfloat162float(bias[index % n]) : 0.0f);
  auto rounded = __float2bfloat16_rn(value);
  if constexpr (AddResidual)
    rounded = __float2bfloat16_rn(__fadd_rn(__bfloat162float(rounded),
                                            __bfloat162float(residual[index])));
  output[index] = rounded;
}

template <int Splits, bool AddResidual>
void launch_reduction_impl(const at::Tensor &partials, const at::Tensor &sa,
                           const at::Tensor &sb,
                           const c10::optional<at::Tensor> &bias,
                           const at::Tensor &output, bool biasless,
                           cudaStream_t stream,
                           const c10::optional<at::Tensor> &residual) {
  int n = output.size(1), elements = output.numel();
  const auto *bias_ptr =
      bias ? reinterpret_cast<const __nv_bfloat16 *>(bias->data_ptr())
           : nullptr;
  auto *output_ptr = reinterpret_cast<__nv_bfloat16 *>(output.data_ptr());
  const auto *residual_ptr =
      residual ? reinterpret_cast<const __nv_bfloat16 *>(residual->data_ptr())
               : nullptr;
  dim3 grid((elements + 255) / 256), block(256);
  if (bias || !biasless)
    reduce_scale<Splits, true, AddResidual><<<grid, block, 0, stream>>>(
        partials.data_ptr<int32_t>(), sa.data_ptr<float>(),
        sb.data_ptr<float>(), bias_ptr, residual_ptr, output_ptr, n, elements);
  else
    reduce_scale<Splits, false, AddResidual><<<grid, block, 0, stream>>>(
        partials.data_ptr<int32_t>(), sa.data_ptr<float>(),
        sb.data_ptr<float>(), bias_ptr, residual_ptr, output_ptr, n, elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Splits>
void launch_reduction(const at::Tensor &partials, const at::Tensor &sa,
                      const at::Tensor &sb,
                      const c10::optional<at::Tensor> &bias,
                      const at::Tensor &output, bool biasless,
                      cudaStream_t stream,
                      const c10::optional<at::Tensor> &residual) {
  if (residual)
    launch_reduction_impl<Splits, true>(partials, sa, sb, bias, output,
                                        biasless, stream, residual);
  else
    launch_reduction_impl<Splits, false>(partials, sa, sb, bias, output,
                                         biasless, stream, residual);
}

template <int Bits, int TileM = 64, int TileN = 128, int K8 = 64,
          int WarpM = 32, int WarpN = 64, int Stages = 3>
at::Tensor run(const at::Tensor &a, const at::Tensor &sa, const at::Tensor &b,
               const at::Tensor &sb, const c10::optional<at::Tensor> &bias,
               int splits, bool biasless, cudaStream_t stream,
               const c10::optional<at::Tensor> &residual) {
  using Element = std::conditional_t<Bits == 4, cutlass::int4b_t, int8_t>;
  constexpr int TileK = Bits == 4 ? K8 * 2 : K8;
  constexpr int Alignment = 128 / Bits;
  using Tile = cutlass::gemm::GemmShape<TileM, TileN, TileK>;
  using Warp = cutlass::gemm::GemmShape<WarpM, WarpN, TileK>;
  using Instruction = cutlass::gemm::GemmShape<16, 8, Bits == 4 ? 64 : 32>;
  using Convert = cutlass::epilogue::thread::Convert<int32_t, 4, int32_t>;
  using Swizzle =
      cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmSplitKParallel<
      Element, cutlass::layout::RowMajor, Alignment, Element,
      cutlass::layout::ColumnMajor, Alignment, int32_t,
      cutlass::layout::RowMajor, int32_t, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, Tile, Warp, Instruction, Convert, Swizzle, Stages,
      cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
  using RefA = typename Kernel::Mma::IteratorA::TensorRef;
  using RefB = typename Kernel::Mma::IteratorB::TensorRef;
  using RefD = typename Kernel::Epilogue::OutputTileIterator::TensorRef;

  int m = a.size(0), n = b.size(0), k = a.size(1) * (Bits == 4 ? 2 : 1);
  int64_t elements = int64_t(m) * n;
  // Both allocations belong to this device's current stream. The partials can
  // be freed on return because their reduction was enqueued on the same stream.
  auto partials = at::empty({splits, m, n}, a.options().dtype(at::kInt));
  auto output = at::empty({m, n}, a.options().dtype(at::kBFloat16));
  cutlass::gemm::GemmCoord problem(m, n, k);
  auto tiled = Swizzle::get_tiled_shape(problem, {TileM, TileN, TileK}, splits);
  typename Kernel::Params params(
      problem, tiled,
      RefA(reinterpret_cast<Element *>(a.data_ptr()),
           cutlass::layout::RowMajor(k)),
      RefB(reinterpret_cast<Element *>(b.data_ptr()),
           cutlass::layout::ColumnMajor(k)),
      RefD(partials.data_ptr<int32_t>(), cutlass::layout::RowMajor(n)),
      typename Convert::Params{}, elements);
  constexpr int shared_bytes = sizeof(typename Kernel::SharedStorage);
  if constexpr (shared_bytes >= (48 << 10))
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes));
  cutlass::Kernel<Kernel>
      <<<Swizzle::get_grid_shape(tiled), Kernel::kThreadCount, shared_bytes,
         stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  switch (splits) {
  case 2:
    launch_reduction<2>(partials, sa, sb, bias, output, biasless, stream,
                        residual);
    break;
  case 4:
    launch_reduction<4>(partials, sa, sb, bias, output, biasless, stream,
                        residual);
    break;
  case 8:
    launch_reduction<8>(partials, sa, sb, bias, output, biasless, stream,
                        residual);
    break;
  }
  return output;
}

template <int Bits>
at::Tensor dispatch_tile(const at::Tensor &a, const at::Tensor &sa,
                         const at::Tensor &b, const at::Tensor &sb,
                         const c10::optional<at::Tensor> &bias, int splits,
                         bool biasless, int64_t tile, cudaStream_t stream,
                         const c10::optional<at::Tensor> &residual) {
  switch (tile) {
  case 0:
    return run<Bits>(a, sa, b, sb, bias, splits, biasless, stream, residual);
  case 1:
    return run<Bits, 16, 64, 64, 16, 32, 3>(a, sa, b, sb, bias, splits,
                                            biasless, stream, residual);
  case 2:
    return run<Bits, 64, 64, 128, 32, 32, 2>(a, sa, b, sb, bias, splits,
                                             biasless, stream, residual);
  default:
    TORCH_CHECK(false, "Unknown split-K tile");
  }
}

at::Tensor split_k_gemm(const at::Tensor &a, const at::Tensor &sa,
                        const at::Tensor &b, const at::Tensor &sb,
                        const c10::optional<at::Tensor> &bias, int64_t bits,
                        int64_t splits, bool biasless, int64_t tile,
                        const c10::optional<at::Tensor> &residual) {
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");
  TORCH_CHECK(splits == 2 || splits == 4 || splits == 8,
              "splits must be 2, 4 or 8");
  TORCH_CHECK(tile >= 0 && tile <= 2, "tile must be 0, 1 or 2");
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
  TORCH_CHECK(a.scalar_type() == (bits == 4 ? at::kByte : at::kChar) &&
                  b.scalar_type() == a.scalar_type(),
              "Invalid integer storage dtype");
  TORCH_CHECK(sa.scalar_type() == at::kFloat &&
                  sb.scalar_type() == at::kFloat && sa.dim() == 1 &&
                  sb.dim() == 1 && sa.size(0) == a.size(0) &&
                  sb.size(0) == b.size(0),
              "Invalid scale buffers");
  int64_t k = a.size(1) * (bits == 4 ? 2 : 1);
  TORCH_CHECK(a.size(0) <= 64 && b.size(0) > 0 && b.size(0) <= 8192 &&
                  b.size(0) % 8 == 0 && k >= 4096 && k <= 8192 && k % 128 == 0,
              "Split-K requires M<=64, N<=8192 with N%8=0, and "
              "4096<=K<=8192 with K%128=0");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0,
              "Packed matrices require 16-byte pointer alignment");
  if (bias)
    TORCH_CHECK(bias->device() == a.device() &&
                    bias->scalar_type() == at::kBFloat16 &&
                    bias->is_contiguous() && bias->dim() == 1 &&
                    bias->size(0) == b.size(0),
                "Invalid BF16 bias");
  if (residual)
    TORCH_CHECK(residual->is_cuda() && residual->device() == a.device() &&
                    residual->scalar_type() == at::kBFloat16 &&
                    residual->is_contiguous() && residual->dim() == 2 &&
                    residual->size(0) == a.size(0) &&
                    residual->size(1) == b.size(0) &&
                    reinterpret_cast<uintptr_t>(residual->data_ptr()) % 16 == 0,
                "Invalid aligned CUDA BF16 [M,Npad] residual");

  c10::cuda::CUDAGuard guard(a.device());
  if (a.size(0) == 0)
    return at::empty({0, b.size(0)}, a.options().dtype(at::kBFloat16));
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  // Protect inputs allocated on another stream if their Python owners are
  // released before this stream finishes. Temporary workspaces need no events.
  for (const auto *tensor : {&a, &sa, &b, &sb})
    c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(),
                                                  stream);
  if (bias)
    c10::cuda::CUDACachingAllocator::recordStream(bias->storage().data_ptr(),
                                                  stream);
  if (residual)
    c10::cuda::CUDACachingAllocator::recordStream(
        residual->storage().data_ptr(), stream);
  return bits == 4 ? dispatch_tile<4>(a, sa, b, sb, bias, splits, biasless,
                                      tile, stream, residual)
                   : dispatch_tile<8>(a, sa, b, sb, bias, splits, biasless,
                                      tile, stream, residual);
}

} // namespace

TORCH_LIBRARY(pir2_integer_split_k, m) {
  m.def("source_identity() -> str", &split_k_source_identity);
  m.def("gemm(Tensor a, Tensor sa, Tensor b, Tensor sb, Tensor? bias, "
        "int bits, int splits, bool biasless, int tile=0, Tensor? "
        "residual=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(pir2_integer_split_k, CUDA, m) {
  m.impl("gemm", &split_k_gemm);
}
