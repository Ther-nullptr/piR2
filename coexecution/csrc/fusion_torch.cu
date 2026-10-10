// Same strict-rounding kernels, registered as CUDA-only PyTorch operators.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "vision_rope.cu"
#include "adaln_modulation.cu"

static void check_bf(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16, "gr00t fusion: expected CUDA BF16");
    TORCH_CHECK(!x.requires_grad(), "gr00t fusion: inference only");
}

static std::tuple<at::Tensor,at::Tensor> rope_op(
    const at::Tensor& q,const at::Tensor& k,const at::Tensor& c,const at::Tensor& s) {
    check_bf(q);check_bf(k);
    TORCH_CHECK(q.dim()==3 && q.size(1)==16 && q.size(2)==64 && q.size(0)>=1 && q.size(0)<=4096 && k.sizes()==q.sizes(), "gr00t fusion: Q/K shape");
    TORCH_CHECK(c.is_cuda() && s.is_cuda() && k.device()==q.device() && c.device()==q.device() && s.device()==q.device(), "gr00t fusion: devices");
    TORCH_CHECK(c.dim()==2 && c.size(0)==q.size(0) && c.size(1)==64 && s.sizes()==c.sizes() && c.scalar_type()==s.scalar_type() && (c.scalar_type()==at::kFloat || c.scalar_type()==at::kBFloat16), "gr00t fusion: coefficients");
    TORCH_CHECK(q.stride(1)==64 && q.stride(2)==1 && (q.stride(0)==1024 || q.stride(0)==3072) && k.stride(1)==64 && k.stride(2)==1 && (k.stride(0)==1024 || k.stride(0)==3072) && c.is_contiguous() && s.is_contiguous(), "gr00t fusion: strides");
    c10::cuda::CUDAGuard guard(q.device());
    auto oq=at::empty(q.sizes(),q.options());auto ok=at::empty(q.sizes(),q.options());
    auto stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    vision_rope_bf16<<<(q.size(0)*512+255)/256,256,0,stream>>>(
        (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),c.data_ptr(),s.data_ptr(),
        (bf16*)oq.data_ptr(),(bf16*)ok.data_ptr(),q.size(0),q.stride(0),k.stride(0),c.scalar_type()==at::kBFloat16);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {oq,ok};
}

static at::Tensor modulation_op(const at::Tensor& x,const at::Tensor& scale,const at::Tensor& shift) {
    check_bf(x);check_bf(scale);check_bf(shift);
    TORCH_CHECK(x.sizes()==at::IntArrayRef({1,41,1536}) && scale.sizes()==at::IntArrayRef({1,1536}) && shift.sizes()==scale.sizes(), "gr00t fusion: AdaLN shape");
    TORCH_CHECK(x.is_contiguous() && scale.is_contiguous() && shift.is_contiguous() && scale.device()==x.device() && shift.device()==x.device(), "gr00t fusion: AdaLN layout/device");
    c10::cuda::CUDAGuard guard(x.device());auto out=at::empty_like(x);
    adaln_modulation_bf16<<<(x.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream(x.get_device()).stream()>>>(
        (const bf16*)x.data_ptr(),(const bf16*)scale.data_ptr(),(const bf16*)shift.data_ptr(),(bf16*)out.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

TORCH_LIBRARY(gr00t_exact_fusion,m) {
    m.def("vision_rope(Tensor q, Tensor k, Tensor cos, Tensor sin) -> (Tensor, Tensor)");
    m.def("adaln_modulation(Tensor x, Tensor scale, Tensor shift) -> Tensor");
}
TORCH_LIBRARY_IMPL(gr00t_exact_fusion,CUDA,m) {
    m.impl("vision_rope",rope_op);m.impl("adaln_modulation",modulation_op);
}
