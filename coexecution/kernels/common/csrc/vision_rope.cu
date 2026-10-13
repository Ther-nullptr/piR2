// Qwen3-VL visual RoPE. Header-free NVRTC, SM89, no fast math / FMA.
typedef unsigned short bf16;
__device__ __forceinline__ float from_bf(bf16 x) {
    return __uint_as_float(((unsigned int)x) << 16);
}
__device__ __forceinline__ bf16 to_bf(float x) {
    unsigned short result;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(x));
    return result;
}
__device__ __forceinline__ float coefficient(const void* p, int i, int is_bf) {
    return is_bf ? from_bf(((const bf16*)p)[i]) : ((const float*)p)[i];
}
extern "C" __global__ void vision_rope_bf16(
    const bf16* q, const bf16* k, const void* cos, const void* sin,
    bf16* oq, bf16* ok, int tokens, int qs, int ks, int coeff_bf) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= tokens * 16 * 32) return;
    int d = pair % 32;
    int head = (pair / 32) % 16;
    int token = pair / (32 * 16);
    int qi = token * qs + head * 64 + d;
    int ki = token * ks + head * 64 + d;
    int oi = token * 1024 + head * 64 + d;
    int ci = token * 64 + d;
    float c0 = coefficient(cos, ci, coeff_bf);
    float c1 = coefficient(cos, ci + 32, coeff_bf);
    float s0 = coefficient(sin, ci, coeff_bf);
    float s1 = coefficient(sin, ci + 32, coeff_bf);
    float q0 = from_bf(q[qi]), q1 = from_bf(q[qi + 32]);
    float k0 = from_bf(k[ki]), k1 = from_bf(k[ki + 32]);
    // Preserve separate FP32 product rounding and FP32 addition rounding.
    oq[oi] = to_bf(__fadd_rn(__fmul_rn(q0, c0), __fmul_rn(-q1, s0)));
    oq[oi + 32] = to_bf(__fadd_rn(__fmul_rn(q1, c1), __fmul_rn(q0, s1)));
    ok[oi] = to_bf(__fadd_rn(__fmul_rn(k0, c0), __fmul_rn(-k1, s0)));
    ok[oi + 32] = to_bf(__fadd_rn(__fmul_rn(k1, c1), __fmul_rn(k0, s1)));
}
