// Register-resident INT4 channel scaling/packing; owned integer GEMM unchanged.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <climits>

namespace pir2_balanced {

template<int Pad,int Threads>
__global__ void prepare_register(const __nv_bfloat16* x,const __nv_bfloat16* up,
 const __nv_bfloat16* lut,const float* inverse,unsigned char* q,float* scales,int k,int64_t sx,int64_t su){
 constexpr int Pairs=(Pad+2*Threads-1)/(2*Threads);
 float lo[Pairs],hi[Pairs];
 __shared__ float maxima[Threads/32];
 int t=threadIdx.x,row=blockIdx.x;float maximum=0.f;
 #pragma unroll
 for(int j=0;j<Pairs;++j){
  int idx=(j*Threads+t)*2;float a=0.f,b=0.f;
  if(idx<k){
   __nv_bfloat16 v=x[row*sx+idx];a=lut?__bfloat162float(lut[__bfloat16_as_ushort(v)]):__bfloat162float(v);
   if(up)a=__bfloat162float(__float2bfloat16_rn(a*__bfloat162float(up[row*su+idx])));
   a=__fmul_rn(a,inverse[idx]);
  }
  if(idx+1<k){
   __nv_bfloat16 v=x[row*sx+idx+1];b=lut?__bfloat162float(lut[__bfloat16_as_ushort(v)]):__bfloat162float(v);
   if(up)b=__bfloat162float(__float2bfloat16_rn(b*__bfloat162float(up[row*su+idx+1])));
   b=__fmul_rn(b,inverse[idx+1]);
  }
  lo[j]=a;hi[j]=b;maximum=fmaxf(maximum,fmaxf(fabsf(a),fabsf(b)));
 }
 for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,d));
 if((t&31)==0)maxima[t>>5]=maximum;
 __syncthreads();
 if(t<32){
  maximum=t<Threads/32?maxima[t]:0.f;
  for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,d));
  if(t==0){float s=maximum>0?__fmul_rn(maximum,1.f/7.f):1.f;scales[row]=s;maxima[0]=s;}
 }
 __syncthreads();float scale=maxima[0];
 #pragma unroll
 for(int j=0;j<Pairs;++j){
  int idx=j*Threads+t;
  if(idx<Pad/2){
   int a=max(-7,min(7,__float2int_rn(__fdiv_rn(lo[j],scale))));
   int b=max(-7,min(7,__float2int_rn(__fdiv_rn(hi[j],scale))));
   q[row*(Pad/2)+idx]=(a&15)|((b&15)<<4);
  }
 }
}

template<int Threads>
void launch(const at::Tensor& x,const c10::optional<at::Tensor>& lut,const c10::optional<at::Tensor>& up,
 const at::Tensor& inverse,at::Tensor& q,at::Tensor& scales,int k,int pad,int64_t sx,int64_t su,int rows,cudaStream_t stream){
 auto xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
 auto lp=lut?reinterpret_cast<const __nv_bfloat16*>(lut->data_ptr()):nullptr;
 auto u=up?reinterpret_cast<const __nv_bfloat16*>(up->data_ptr()):nullptr;
 auto inv=inverse.data_ptr<float>();auto qp=q.data_ptr<unsigned char>();auto sp=scales.data_ptr<float>();
 #define CASE(N) case N: prepare_register<N,Threads><<<rows,Threads,0,stream>>>(xp,u,lp,inv,qp,sp,k,sx,su);break
 switch(pad){CASE(128);CASE(256);CASE(512);CASE(768);CASE(1024);CASE(1536);CASE(2048);CASE(3072);CASE(4096);CASE(6144);CASE(8192);default:TORCH_CHECK(false,"Unsupported register-prepare width");}
 #undef CASE
}

std::tuple<at::Tensor,at::Tensor> prepare(const at::Tensor& x,const c10::optional<at::Tensor>& lut,
 const c10::optional<at::Tensor>& up,const at::Tensor& inverse,int64_t threads){
 TORCH_CHECK(x.is_cuda() && x.dim() >= 2 && x.size(-1) > 0 && x.size(-1) <= 8192 && x.numel() > 0,"Expected nonempty CUDA rows, width 1..8192");
 c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.scalar_type()==at::kBFloat16 && x.stride(-1)==1 && inverse.scalar_type()==at::kFloat,"BF16 row input / FP32 inverse required");
 int k=x.size(-1),pad=(k+127)/128*128;
 TORCH_CHECK(x.numel()/k <= INT_MAX,"Too many rows");
 int rows=x.numel()/k;
 TORCH_CHECK(inverse.dim()==1 && inverse.numel()==k && inverse.is_contiguous() && inverse.device()==x.device(),"Invalid channel scale");
 TORCH_CHECK(!lut || (lut->device()==x.device() && lut->scalar_type()==at::kBFloat16 && lut->is_contiguous() && lut->numel()==65536),"Invalid BF16 LUT");
 TORCH_CHECK(!up || (lut && up->device()==x.device() && up->scalar_type()==x.scalar_type() && up->sizes()==x.sizes() && up->stride(-1)==1),"Invalid gated BF16 input");
 auto a=x.view({rows,k});auto q=at::empty({rows,pad/2},x.options().dtype(at::kByte));auto s=at::empty({rows},x.options().dtype(at::kFloat));
 int64_t sx=a.stride(0),su=up?up->view({rows,k}).stride(0):sx;
 auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
 if(threads==128)launch<128>(x,lut,up,inverse,q,s,k,pad,sx,su,rows,stream);
 else if(threads==256)launch<256>(x,lut,up,inverse,q,s,k,pad,sx,su,rows,stream);
 else TORCH_CHECK(false,"Threads must be 128 or 256");
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,s};
}

at::Tensor linear(const at::Tensor& x,const at::Tensor& w,const at::Tensor& ws,
 const c10::optional<at::Tensor>& bias,const c10::optional<at::Tensor>& lut,
 const c10::optional<at::Tensor>& up,const at::Tensor& inverse,int64_t threads,int64_t tactic){
 c10::cuda::CUDAGuard guard(x.device());
 auto [q,s]=prepare(x,lut,up,inverse,threads);
 using Sig=at::Tensor(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const c10::optional<at::Tensor>&,int64_t,int64_t);
 static auto gemm=c10::Dispatcher::singleton().findSchemaOrThrow("pir2_integer::gemm_biasless","").typed<Sig>();
 return gemm.call(q,s,w,ws,bias,4,tactic);
}
} // namespace pir2_balanced

TORCH_LIBRARY(pir2_int4_prepare,m){
 m.def("prepare(Tensor x, Tensor? lut, Tensor? up, Tensor inverse, int threads) -> (Tensor, Tensor)");
 m.def("linear(Tensor x, Tensor w, Tensor ws, Tensor? bias, Tensor? lut, Tensor? up, Tensor inverse, int threads, int tactic) -> Tensor");
}
TORCH_LIBRARY_IMPL(pir2_int4_prepare,CUDA,m){m.impl("prepare",&pir2_balanced::prepare);m.impl("linear",&pir2_balanced::linear);}
