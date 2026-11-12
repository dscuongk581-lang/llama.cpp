// VLANIX P3.0 experimental CPU Q4_0 x Q8_0 dot operator.
// OFF by default. Included only when GGML_VCPU_LAB is explicitly built.
// This is deliberately narrower than a complete GGML backend.
#pragma once
#if defined(GGML_VCPU_LAB) && defined(__SSE2__) && (defined(__x86_64__) || defined(__i386__))
#include <emmintrin.h>
#include <stdint.h>
#include <string.h>

// Q4_0 low/high 4-bit halves are paired with the first/last 16 Q8_0 values.
// nrc must be one: do not intercept GGML's multi-row dot kernels.
static void ggml_vcpu_lab_vec_dot_q4_0_q8_0(
        int n, float * restrict s, size_t bs,
        const void * restrict vx, size_t bx,
        const void * restrict vy, size_t by, int nrc) {
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by);
    GGML_ASSERT(nrc == 1 && n % 32 == 0);
    const block_q4_0 * w = (const block_q4_0 *) vx;
    const block_q8_0 * a = (const block_q8_0 *) vy;
    const __m128i zero = _mm_setzero_si128();
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i bias = _mm_set1_epi8(8);
    float total = 0.0f;
    for (int block=0; block<n/32; ++block) {
        const __m128i bytes = _mm_loadu_si128((const __m128i *)w[block].qs);
        const __m128i low = _mm_sub_epi8(_mm_and_si128(bytes,mask),bias);
        const __m128i hi  = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(bytes,4),mask),bias);
        const __m128i a0 = _mm_loadu_si128((const __m128i *) a[block].qs);
        const __m128i a1 = _mm_loadu_si128((const __m128i *)(a[block].qs+16));
        const __m128i ls = _mm_cmpgt_epi8(zero,low);
        const __m128i hs = _mm_cmpgt_epi8(zero,hi);
        const __m128i s0 = _mm_cmpgt_epi8(zero,a0);
        const __m128i s1 = _mm_cmpgt_epi8(zero,a1);
        __m128i sum = _mm_setzero_si128();
        sum = _mm_add_epi32(sum,_mm_madd_epi16(_mm_unpacklo_epi8(low,ls),_mm_unpacklo_epi8(a0,s0)));
        sum = _mm_add_epi32(sum,_mm_madd_epi16(_mm_unpackhi_epi8(low,ls),_mm_unpackhi_epi8(a0,s0)));
        sum = _mm_add_epi32(sum,_mm_madd_epi16(_mm_unpacklo_epi8(hi,hs),_mm_unpacklo_epi8(a1,s1)));
        sum = _mm_add_epi32(sum,_mm_madd_epi16(_mm_unpackhi_epi8(hi,hs),_mm_unpackhi_epi8(a1,s1)));
        int32_t lanes[4];
        _mm_storeu_si128((__m128i*)lanes,sum);
        const int dot=lanes[0]+lanes[1]+lanes[2]+lanes[3];
        total+=(float)dot*(ggml_fp16_to_fp32(w[block].d)*ggml_fp16_to_fp32(a[block].d));
    }
    *s=total;
}
#endif
