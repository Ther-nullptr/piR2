// Header-free NVRTC CUDA; no external toolkit or host compiler required.
// All low-precision materialization boundaries are explicit. Compile --fmad=false.
typedef unsigned short bf16;
__device__ __forceinline__ float read_bf(bf16 x) {
    return __uint_as_float(((unsigned int)x) << 16);
}
__device__ __forceinline__ bf16 round_bf(float x) {
    unsigned short out;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(x));
    return out;
}
__device__ __forceinline__ bf16 modulate(float normalized, bf16 scale, bf16 shift) {
    float factor = read_bf(round_bf(1.0f + read_bf(scale)));
    float product = read_bf(round_bf(normalized * factor));
    return round_bf(product + read_bf(shift));
}
extern "C" __global__ void adaln_modulation_bf16(
    const bf16* x, const bf16* scale, const bf16* shift, bf16* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 41 * 1536) out[i] = modulate(read_bf(x[i]), scale[i % 1536], shift[i % 1536]);
}
