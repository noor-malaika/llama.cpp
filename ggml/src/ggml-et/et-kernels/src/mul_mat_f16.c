//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

// Dot one F16 weight row against 4 activation columns at once. The weight
// row is the same for all 4 columns, so its per-block F16->F32 gather+
// convert (the "decode" cost) is done ONCE per block and reused across all
// 4 fmadd chains, instead of being redone from scratch for every column
// like the single-column path (compute_block_dot_product_f16_naive) does.
static inline void
dot_f16_row_n4(const uint16_t* f16_row, int64_t K_blocks,
                const float* b_col0, const float* b_col1,
                const float* b_col2, const float* b_col3,
                float out[4]) {
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    __asm__ volatile(
        "fbci.pi f20, 0\n" "fbci.pi f21, 0\n"
        "fbci.pi f22, 0\n" "fbci.pi f23, 0\n"
        ::: "f20", "f21", "f22", "f23");

    static const int32_t gather_pattern[8] = {0, 2, 4, 6, 8, 10, 12, 14};
    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t(*)[8])gather_pattern) : "f31");

    for (int64_t kb = 0; kb < K_blocks; kb++) {
        const int64_t off = kb * QK_F16;

        for (int chunk = 0; chunk < 4; chunk++) {
            const int co = chunk << 3;
            __asm__ volatile(
                "fgh.ps   f11, f31(%[a_ptr])\n"   // gather 8 f16 weights (shared)
                "fcvt.ps.f16 f11, f11\n"          // -> float
                "flw.ps   f12, %[b0]\n"
                "fmadd.ps f20, f11, f12, f20\n"
                "flw.ps   f12, %[b1]\n"
                "fmadd.ps f21, f11, f12, f21\n"
                "flw.ps   f12, %[b2]\n"
                "fmadd.ps f22, f11, f12, f22\n"
                "flw.ps   f12, %[b3]\n"
                "fmadd.ps f23, f11, f12, f23\n"
                :
                : [a_ptr] "r"(&f16_row[off + co]),
                  [b0] "m"(*(const float(*)[8])&b_col0[off + co]),
                  [b1] "m"(*(const float(*)[8])&b_col1[off + co]),
                  [b2] "m"(*(const float(*)[8])&b_col2[off + co]),
                  [b3] "m"(*(const float(*)[8])&b_col3[off + co])
                : "f11", "f12", "f20", "f21", "f22", "f23"
            );
        }
    }

#define HREDUCE(reg, dest)                                                   \
    __asm__ __volatile__(                                                   \
        "fswizz.ps f1, " #reg ", 0xB1 \n\t"                                 \
        "fadd.ps   f2, " #reg ", f1, rne \n\t"                              \
        "fswizz.ps f3, f2, 0x4E \n\t"                                       \
        "fadd.ps   f4, f2, f3, rne \n\t"                                    \
        "fmvz.x.ps t0, f4, 4 \n\t"                                          \
        "fbcx.ps   f5, t0 \n\t"                                             \
        "fadd.ps   %[vout], f4, f5, rne \n\t"                               \
        : [vout] "=f"(dest)                                                 \
        :: "t0", "f1", "f2", "f3", "f4", "f5")

    HREDUCE(f20, out[0]);
    HREDUCE(f21, out[1]);
    HREDUCE(f22, out[2]);
    HREDUCE(f23, out[3]);
#undef HREDUCE

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

int entry_point(struct ggml_et_binary_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env || params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1;
    }

    // Thread coordination
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0 || (thread_id & 1)) {
        return 0; // Skip odd threads to avoid resource contention
    }

    int effective_thread_id = thread_id / 2;
    int effective_num_threads = (num_threads + 1) / 2;

    // Extract tensor references
    struct ggml_tensor* src0 = &params->src0; // Weight matrix A (F16)
    struct ggml_tensor* src1 = &params->src1; // Activation matrix B (F32)
    struct ggml_tensor* dst  = &params->dst;  // Output matrix C (F32)

    // Strictly validate: src0 is F16, others are F32
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const uint16_t* src0_data = (const uint16_t*)src0->data;
    const float* src1_data = (const float*)src1->data;
    float* dst_data  = (float*)dst->data;

    // Dimensions and Strides
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    const int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    const int64_t ne2  = dst->ne[2],  ne3  = dst->ne[3];

    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    // F16 specific block size (Usually QK_F16)
    const int block_size = QK_F16;
    const int64_t K_blocks = K / block_size;
    const int64_t K_remainder = K % block_size;

    // Broadcasting support
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    // Work units are (m, n-block, i2, i3) rather than individual (m, n, i2,
    // i3) elements - each n-block covers 4 columns of a shared weight row
    // (see dot_f16_row_n4), with one smaller tail block if N % 4 != 0. This
    // keeps the SAME flat-index-over-everything distribution the original
    // per-element scheme used (so load balance across M/N/batch stays even
    // no matter how skewed the individual dims are - critical here since M
    // is often small, e.g. 64-512), just at a coarser 4-column granularity
    // that lets the F16 decode be shared across those 4 columns.
    const int64_t n_full_groups = N / 4;
    const int64_t n_tail = N % 4;
    const int64_t n_blocks = n_full_groups + (n_tail > 0 ? 1 : 0);
    const uint64_t total_units = (uint64_t) M * n_blocks * ne2 * ne3;

    for (uint64_t unit = effective_thread_id; unit < total_units; unit += effective_num_threads) {
        const int64_t i3 = unit / (M * n_blocks * ne2);
        const int64_t rem3 = unit % (M * n_blocks * ne2);
        const int64_t i2 = rem3 / (M * n_blocks);
        const int64_t rem2 = rem3 % (M * n_blocks);
        const int64_t m = rem2 / n_blocks;
        const int64_t nb = rem2 % n_blocks;

        const int64_t i03 = i3 / r3, i02 = i2 / r2;
        const int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

        const uint16_t* f16_row = (const uint16_t*)((const char*)src0_data + m * nb01 + i02 * nb02 + i03 * nb03);
        const char* src1_batch = (const char*)src1_data + i12 * nb12 + i13 * nb13;
        char* dst_batch = (char*)dst_data + i2 * nb2 + i3 * nb3;

        if (nb < n_full_groups) {
            const int64_t n = nb * 4;
            const float* b0 = (const float*)(src1_batch + (n + 0) * nb11);
            const float* b1 = (const float*)(src1_batch + (n + 1) * nb11);
            const float* b2 = (const float*)(src1_batch + (n + 2) * nb11);
            const float* b3 = (const float*)(src1_batch + (n + 3) * nb11);

            float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dot_f16_row_n4(f16_row, K_blocks, b0, b1, b2, b3, sums);

            if (K_remainder > 0) {
                const int64_t offset = K_blocks * block_size;
                sums[0] += compute_block_dot_product_f16_partial(&f16_row[offset], b0 + offset, K_remainder);
                sums[1] += compute_block_dot_product_f16_partial(&f16_row[offset], b1 + offset, K_remainder);
                sums[2] += compute_block_dot_product_f16_partial(&f16_row[offset], b2 + offset, K_remainder);
                sums[3] += compute_block_dot_product_f16_partial(&f16_row[offset], b3 + offset, K_remainder);
            }

            for (int c = 0; c < 4; c++) {
                float* dst_entry = (float*)(dst_batch + (n + c) * nb1 + m * dst->nb[0]);
                atomic_store_f32((volatile float*)dst_entry, sums[c]);
            }
        } else {
            // Tail block: the N % 4 leftover columns, original single-column path.
            for (int64_t n = n_full_groups * 4; n < N; n++) {
                const float* b_col_ptr = (const float*)(src1_batch + n * nb11);
                float sum = 0.0f;

                for (int64_t kb = 0; kb < K_blocks; kb++) {
                    sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr + kb * block_size);
                }
                if (K_remainder > 0) {
                    const int64_t offset = K_blocks * block_size;
                    sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr + offset, K_remainder);
                }

                float* dst_entry = (float*)(dst_batch + n * nb1 + m * dst->nb[0]);
                atomic_store_f32((volatile float*)dst_entry, sum);
            }
        }
    }

    return 0;
}
