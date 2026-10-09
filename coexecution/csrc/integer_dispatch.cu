// One host dispatch for activation preparation and the unchanged robotics GEMM.
// Symmetric row quantization/INT4 layout follows robotics-the-speedup-paradox
// 239b4a3ef2268048571c9f508ad5700f98398c2f; its CUTLASS GEMM remains external.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

template<int Bits>
__global__ void prepare_integer(const __nv_bfloat16* x,const __nv_bfloat16* up,
 const __nv_bfloat16* lut,unsigned char* q,float* scales,int k,int pad,int sx,int su) {
 extern __shared__ float values[];
 __shared__ float maxima[8];
 int row=blockIdx.x,t=threadIdx.x;
 float maximum=0.f;
 for(int i=t;i<pad;i+=256){
  float v=0.f;
  if(i<k){
   __nv_bfloat16 input=x[row*sx+i];
   v=lut?__bfloat162float(lut[__bfloat16_as_ushort(input)]):__bfloat162float(input);
   if(up) v=__bfloat162float(__float2bfloat16_rn(v*__bfloat162float(up[row*su+i])));
  }
  values[i]=v; maximum=fmaxf(maximum,fabsf(v));
 }
 for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,d));
 if((t&31)==0)maxima[t>>5]=maximum;
 __syncthreads();
 if(t<32){
  maximum=t<8?maxima[t]:0.f;
  for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,d));
  if(t==0){float s=maximum>0?maximum*(1.f/((1<<(Bits-1))-1)):1.f;scales[row]=s;maxima[0]=s;}
 }
 __syncthreads();
 constexpr int limit=(1<<(Bits-1))-1;
 float scale=maxima[0];
 if constexpr(Bits==8){
  for(int i=t;i<pad;i+=256){int v=__float2int_rn(__fdiv_rn(values[i],scale));q[row*pad+i]=(unsigned char)max(-limit,min(limit,v));}
 }else{
  for(int i=t;i<pad/2;i+=256){
   int lo=max(-limit,min(limit,__float2int_rn(__fdiv_rn(values[2*i],scale))));
   int hi=max(-limit,min(limit,__float2int_rn(__fdiv_rn(values[2*i+1],scale))));
   q[row*(pad/2)+i]=(lo&15)|((hi&15)<<4);
  }
 }
}

at::Tensor fused_linear(const at::Tensor& x,const at::Tensor& w,const at::Tensor& ws,
 const c10::optional<at::Tensor>& bias,const c10::optional<at::Tensor>& lut,
 const c10::optional<at::Tensor>& up,int64_t bits,int64_t tactic){
 c10::cuda::CUDAGuard guard(x.device());
 int k=x.size(-1),m=x.numel()/k,pad=(k+127)/128*128;
 auto rows=x.view({m,k});
 auto q=at::empty({m,bits==8?pad:pad/2},x.options().dtype(bits==8?at::kChar:at::kByte));
 auto scale=at::empty({m},x.options().dtype(at::kFloat));
 auto xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
 auto lp=lut?reinterpret_cast<const __nv_bfloat16*>(lut->data_ptr()):nullptr;
 auto uptr=up?reinterpret_cast<const __nv_bfloat16*>(up->data_ptr()):nullptr;
 int su=up?up->view({m,k}).stride(0):rows.stride(0);
 auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
 if(bits==8)prepare_integer<8><<<m,256,pad*sizeof(float),stream>>>(xp,uptr,lp,reinterpret_cast<unsigned char*>(q.data_ptr()),scale.data_ptr<float>(),k,pad,rows.stride(0),su);
 else prepare_integer<4><<<m,256,pad*sizeof(float),stream>>>(xp,uptr,lp,reinterpret_cast<unsigned char*>(q.data_ptr()),scale.data_ptr<float>(),k,pad,rows.stride(0),su);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 using Sig=at::Tensor(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const c10::optional<at::Tensor>&,int64_t,int64_t);
 static auto gemm=c10::Dispatcher::singleton().findSchemaOrThrow("robotics_integer::gemm_biasless","").typed<Sig>();
 return gemm.call(q,scale,w,ws,bias,bits,tactic);
}

TORCH_LIBRARY(pir2_integer_dispatch,m){
 m.def("linear(Tensor x, Tensor w, Tensor ws, Tensor? bias, Tensor? lut, Tensor? up, int bits, int tactic) -> Tensor");
}
TORCH_LIBRARY_IMPL(pir2_integer_dispatch,CUDA,m){m.impl("linear",&fused_linear);}
