/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Exact residual and segmented group writers over the pinned GEMM run template.
// Only Torch's ignored build cache contains the adapted dependency source.
#include <cutlass/arch/memory.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/numeric_conversion.h>

// The pinned visitors require numeric, GEMM and thread-map declarations first.
#include <cutlass/epilogue/threadblock/fusion/visitor_store.hpp>

#include <cstdint>

namespace pir2_epilogue {
template <class T> struct Identity {
  CUTLASS_HOST_DEVICE T operator()(T const &value) const { return value; }
};

struct GroupSizes {
  int widths[3] = {0, 0, 0};
};

// Coordinate-based mapping is required: an affine dAux stride cannot express
// [Q rows][K rows][V rows]. The pinned base owns partitioning, register clears,
// and RN BF16 conversion; only its vector-store destination is changed.
template <class ThreadMap, class Element, cutlass::FloatRoundStyle RoundStyle,
          class StrideMNL>
struct GroupStore {
  using Base =
      cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, Element,
                                                      RoundStyle, StrideMNL>;
  using VecType = typename Base::VecType;
  static_assert(ThreadMap::kElementsPerAccess == 8 && Base::VecLength == 8 &&
                    sizeof(VecType) == 16,
                "Segment writer requires aligned eight-element BF16 vectors");

  struct Arguments {
    // Base callbacks retain a pointer to this persistent parameter object.
    typename Base::Arguments base;
    GroupSizes group_sizes;
  };
  using Params = Arguments;
  using SharedStorage = typename Base::SharedStorage;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &, Arguments const &args, void *) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const &, Arguments const &) {
    return 0;
  }

  CUTLASS_HOST_DEVICE GroupStore() {}
  CUTLASS_HOST_DEVICE GroupStore(Params const &params,
                                 SharedStorage const &shared_storage)
      : base(params.base, shared_storage), params_ptr(&params) {}

  Base base;
  Params const *params_ptr;

  template <class BaseCallbacks> struct Callbacks : BaseCallbacks {
    CUTLASS_DEVICE Callbacks(BaseCallbacks &&callbacks, Params const &params)
        : BaseCallbacks(cute::move(callbacks)), ptr_aux(params.base.ptr_aux),
          group_sizes(params.group_sizes) {}

    Element *ptr_aux;
    GroupSizes group_sizes;

    CUTLASS_DEVICE void end_step(int step_idx) {
      auto src_v = cute::filter(this->tC_rAux);
      auto coord_v =
          cute::filter(this->tC_cAux(cute::_, cute::_, cute::_, step_idx));
      int64_t rows = cute::get<0>(this->problem_shape);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(src_v); ++i) {
        bool guard = cute::elem_less(coord_v(i), this->problem_shape);
        int64_t row = cute::get<0>(coord_v(i));
        int64_t column = cute::get<1>(coord_v(i));
        int offset = 0, width = group_sizes.widths[0];
        if (column >= width) {
          offset = width;
          width = group_sizes.widths[1];
          if (column >= offset + width) {
            offset += width;
            width = group_sizes.widths[2];
          }
        }
        // Widths and virtual vector coordinates are divisible by eight, so a
        // vector cannot cross a segment and every segment row stays aligned.
        void *destination = ptr_aux;
        if (guard)
          destination = ptr_aux + rows * offset + row * width + column - offset;
        cutlass::arch::global_store<VecType, sizeof(VecType)>(
            src_v(i), destination, guard);
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(cutlass::gemm::GemmCoord threadblock_tile_offset,
                int thread_idx, ProblemShape problem_shape) {
    auto callbacks =
        base.get_callbacks(threadblock_tile_offset, thread_idx, problem_shape);
    return Callbacks<decltype(callbacks)>(cute::move(callbacks), *params_ptr);
  }
};
} // namespace pir2_epilogue

#include "pir2_integer_epilogue_backend.cuh"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>

#include <string>

namespace {

std::string epilogue_source_identity() { return PIR2_EPILOGUE_SOURCE_ID; }

template <int Bits, bool WithBias, bool GroupWrite = false>
at::Tensor dispatch_output(const at::Tensor &a, const at::Tensor &sa,
                           const at::Tensor &b, const at::Tensor &sb,
                           const c10::optional<at::Tensor> &bias,
                           int64_t tactic, const at::Tensor &residual,
                           pir2_epilogue::GroupSizes group_sizes = {}) {
  switch (tactic) {
  case 0:
    return run<WithBias, Bits, 64, 128, 64, 32, 64, 3, GroupWrite>(
        a, sa, b, sb, bias, residual, group_sizes);
  case 2:
    return run<WithBias, Bits, 64, 64, 64, 32, 32, 3, GroupWrite>(
        a, sa, b, sb, bias, residual, group_sizes);
  case 7:
    return run<WithBias, Bits, 32, 128, 64, 32, 64, 3, GroupWrite>(
        a, sa, b, sb, bias, residual, group_sizes);
  default:
    TORCH_CHECK(false, "Integer epilogue tactic must be 0, 2 or 7");
  }
}

template <int Bits, bool GroupWrite = false>
at::Tensor dispatch_bias(const at::Tensor &a, const at::Tensor &sa,
                         const at::Tensor &b, const at::Tensor &sb,
                         const c10::optional<at::Tensor> &bias, int64_t tactic,
                         bool biasless, const at::Tensor &residual,
                         pir2_epilogue::GroupSizes group_sizes = {}) {
  return bias || !biasless
             ? dispatch_output<Bits, true, GroupWrite>(
                   a, sa, b, sb, bias, tactic, residual, group_sizes)
             : dispatch_output<Bits, false, GroupWrite>(
                   a, sa, b, sb, bias, tactic, residual, group_sizes);
}

at::Tensor residual_gemm(const at::Tensor &a, const at::Tensor &sa,
                         const at::Tensor &b, const at::Tensor &sb,
                         const c10::optional<at::Tensor> &bias, int64_t bits,
                         int64_t tactic, bool biasless,
                         const at::Tensor &residual) {
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");
  TORCH_CHECK(tactic == 0 || tactic == 2 || tactic == 7,
              "Residual epilogue tactic must be 0, 2 or 7");
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
  int64_t m = a.size(0), n = b.size(0);
  int64_t k = a.size(1) * (bits == 4 ? 2 : 1);
  TORCH_CHECK(m <= 64 && n > 0 && n <= 8192 && n % 8 == 0 && k >= 128 &&
                  k <= 8192 && k % 128 == 0,
              "Residual epilogue requires M<=64, padded N<=8192 with N%8=0, "
              "and 128<=K<=8192 with K%128=0");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(sb.data_ptr()) % 16 == 0,
              "Packed matrices and weight scales require 16-byte alignment");
  if (bias)
    TORCH_CHECK(
        bias->device() == a.device() && bias->scalar_type() == at::kBFloat16 &&
            bias->is_contiguous() && bias->dim() == 1 && bias->size(0) == n &&
            reinterpret_cast<uintptr_t>(bias->data_ptr()) % 16 == 0,
        "Invalid aligned BF16 bias");
  TORCH_CHECK(residual.is_cuda() && residual.device() == a.device() &&
                  residual.scalar_type() == at::kBFloat16 &&
                  residual.is_contiguous() && residual.dim() == 2 &&
                  residual.size(0) == m && residual.size(1) == n &&
                  reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16 == 0,
              "Invalid aligned CUDA BF16 [M,Npad] residual");

  c10::cuda::CUDAGuard guard(a.device());
  if (m == 0)
    return at::empty({0, n}, a.options().dtype(at::kBFloat16));
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  for (const auto *tensor : {&a, &sa, &b, &sb, &residual})
    c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(),
                                                  stream);
  if (bias)
    c10::cuda::CUDACachingAllocator::recordStream(bias->storage().data_ptr(),
                                                  stream);
  auto output =
      bits == 4
          ? dispatch_bias<4>(a, sa, b, sb, bias, tactic, biasless, residual)
          : dispatch_bias<8>(a, sa, b, sb, bias, tactic, biasless, residual);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor group_gemm(const at::Tensor &a, const at::Tensor &sa,
                      const at::Tensor &b, const at::Tensor &sb,
                      const c10::optional<at::Tensor> &bias, int64_t bits,
                      int64_t tactic, bool biasless, at::IntArrayRef sizes) {
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");
  TORCH_CHECK(tactic == 0 || tactic == 2 || tactic == 7,
              "Group epilogue tactic must be 0, 2 or 7");
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
  int64_t m = a.size(0), n = b.size(0);
  int64_t k = a.size(1) * (bits == 4 ? 2 : 1);
  TORCH_CHECK(m <= 4096 && n > 0 && n <= 8192 && n % 8 == 0 && k >= 128 &&
                  k <= 8192 && k % 128 == 0,
              "Group epilogue requires M<=4096, N<=8192 with N%8=0, "
              "and 128<=K<=8192 with K%128=0");
  TORCH_CHECK(sizes.size() == 2 || sizes.size() == 3,
              "Group epilogue requires two or three segments");
  pir2_epilogue::GroupSizes group_sizes;
  int64_t total = 0;
  for (size_t index = 0; index < sizes.size(); ++index) {
    TORCH_CHECK(sizes[index] > 0 && sizes[index] <= n && sizes[index] % 8 == 0,
                "Group epilogue requires positive aligned segment widths");
    group_sizes.widths[index] = int(sizes[index]);
    total += sizes[index];
  }
  TORCH_CHECK(total == n, "Group epilogue widths must cover physical N");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(sb.data_ptr()) % 16 == 0,
              "Packed matrices and weight scales require 16-byte alignment");
  if (bias)
    TORCH_CHECK(
        bias->device() == a.device() && bias->scalar_type() == at::kBFloat16 &&
            bias->is_contiguous() && bias->dim() == 1 && bias->size(0) == n &&
            reinterpret_cast<uintptr_t>(bias->data_ptr()) % 16 == 0,
        "Invalid aligned BF16 bias");

  c10::cuda::CUDAGuard guard(a.device());
  if (m == 0)
    return at::empty({0}, a.options().dtype(at::kBFloat16));
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  for (const auto *tensor : {&a, &sa, &b, &sb})
    c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(),
                                                  stream);
  if (bias)
    c10::cuda::CUDACachingAllocator::recordStream(bias->storage().data_ptr(),
                                                  stream);
  auto output =
      bits == 4 ? dispatch_bias<4, true>(a, sa, b, sb, bias, tactic, biasless,
                                         at::Tensor{}, group_sizes)
                : dispatch_bias<8, true>(a, sa, b, sb, bias, tactic, biasless,
                                         at::Tensor{}, group_sizes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace

TORCH_LIBRARY(pir2_integer_epilogue, m) {
  m.def("source_identity() -> str", &epilogue_source_identity);
  m.def("gemm(Tensor a, Tensor sa, Tensor b, Tensor sb, Tensor? bias, "
        "int bits, int tactic, bool biasless, Tensor residual) -> Tensor");
  m.def("group_gemm(Tensor a, Tensor sa, Tensor b, Tensor sb, Tensor? bias, "
        "int bits, int tactic, bool biasless, int[] sizes) -> Tensor");
}
TORCH_LIBRARY_IMPL(pir2_integer_epilogue, CUDA, m) {
  m.impl("gemm", &residual_gemm);
  m.impl("group_gemm", &group_gemm);
}
