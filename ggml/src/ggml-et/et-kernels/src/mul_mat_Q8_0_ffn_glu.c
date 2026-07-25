//******************************************************************************
// Fused SwiGLU feed-forward kernel (Q8_0 weights, F32 activations)
//
//   dst[m] = silu(gate[m] . act) * (up[m] . act)
//
// Replaces three kernel launches (ffn_gate MUL_MAT, ffn_up MUL_MAT, GLU) with
// one. Both dot products for a given output element are produced by the same
// hart, so the two n_ff-sized intermediates never leave registers -- the
// unfused chain writes both to memory and reads them back in the GLU.
//
// Row distribution matches mul_mat_Q8_0.c: one output row per hart, strided by
// the number of harts actually launched.
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

// Pull a byte range into L2 for the calling hart. Mirrors mul_mat_Q8_0.c; see
// the rationale there. This kernel streams two weight rows per output element,
// so it has twice the miss traffic to cover.
static inline void et_prefetch_range(const void* start_ptr, int64_t num_bytes) {
    if (num_bytes <= 0) {
        return;
    }
    uintptr_t ptr     = (uintptr_t)start_ptr & ~(uintptr_t)63;
    uintptr_t end_ptr = (uintptr_t)start_ptr + (uintptr_t)num_bytes;
    int64_t   lines   = (int64_t)(((end_ptr - ptr) + 63) >> 6);

    while (lines > 0) {
        const uint64_t batch = (uint64_t)((lines > 16 ? 16 : lines) - 1);
        __asm__ __volatile__ (
            "li    x1, 0x400000000000000 \n"  // Dest = L2 (bits 59:58 = 01)
            "addi  x31, zero, 64\n"           // Stride = 64 bytes
            "or    x3, x1, %[ptr]\n"          // Combine Dest + VA
            "or    x3, x3, %[sz]\n"           // Combine with NumLines
            "csrw  0x81f, x3\n"               // prefetch_va
            :
            : [ptr] "r" (ptr), [sz] "r" (batch)
            : "x1", "x3", "x31", "memory"
        );
        ptr   += 16 * 64;
        lines -= 16;
    }
}

// silu(x) = x / (1 + exp(-x)), guarded at the tails so exp() cannot overflow.
// Same formulation as glu_f32.c's silu_f32 so the fused and unfused paths agree.
static inline float ffn_silu_f32(float x) {
    if (x > 20.0f) {
        return x;
    } else if (x < -20.0f) {
        return 0.0f;
    }
    const float exp_neg_x = et_expf(-x);
    return et_fdiv(x, 1.0f + exp_neg_x);
}

int entry_point(struct ggml_et_mm_q8_ffn_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) {
        return -1;
    }

    const int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }

    const int64_t hart_id  = (int64_t)thread_id;
    const int64_t stride_m = get_num_threads(kernel_env->shire_mask);

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1;
    }

    if (params->gate.type != GGML_TYPE_Q8_0 ||
        params->up.type   != GGML_TYPE_Q8_0 ||
        params->act.type  != GGML_TYPE_F32  ||
        params->dst.type  != GGML_TYPE_F32) {
        return -1;
    }

    const int64_t K = params->gate.ne[0];
    const int64_t M = params->gate.ne[1];
    const int64_t N = params->act.ne[1];

    // The host only dispatches here when gate and up have identical geometry.
    if (params->up.ne[0] != K || params->up.ne[1] != M) {
        return -1;
    }
    if ((K % 32) != 0) {
        return -1;  // Q8_0 block size
    }
    const int64_t K_blocks = K / 32;

    const size_t nbg1 = params->gate.nb[1];
    const size_t nbu1 = params->up.nb[1];
    const size_t nba1 = params->act.nb[1];
    const size_t nbd1 = params->dst.nb[1];

    const char* gate_data = (const char*)params->gate.data;
    const char* up_data   = (const char*)params->up.data;
    const char* act_data  = (const char*)params->act.data;
    char*       dst_data  = (char*)params->dst.data;

    if (!gate_data || !up_data || !act_data || !dst_data) {
        return -1;
    }

    const int64_t row_bytes     = K_blocks * (int64_t)sizeof(block_q8_0);
    const int32_t prefetch_rows = params->prefetch_rows;

    for (int64_t n = 0; n < N; n++) {
        const float* act_col = (const float*)(act_data + n * nba1);

        // Warm L2 with this hart's first pair of rows before the chain starts.
        if (prefetch_rows > 0 && hart_id < M) {
            et_prefetch_range(gate_data + hart_id * nbg1, row_bytes);
            et_prefetch_range(up_data   + hart_id * nbu1, row_bytes);
        }

        for (int64_t m = hart_id; m < M; m += stride_m) {
            // Stay prefetch_rows iterations ahead on both weight streams.
            if (prefetch_rows > 0) {
                const int64_t ahead = m + (int64_t)prefetch_rows * stride_m;
                if (ahead < M) {
                    et_prefetch_range(gate_data + ahead * nbg1, row_bytes);
                    et_prefetch_range(up_data   + ahead * nbu1, row_bytes);
                }
            }

            const block_q8_0* gate_row = (const block_q8_0*)(gate_data + m * nbg1);
            const block_q8_0* up_row   = (const block_q8_0*)(up_data   + m * nbu1);

            float gate_sum = 0.0f;
            float up_sum   = 0.0f;

            for (int64_t kb = 0; kb < K_blocks; kb++) {
                const float* act_blk = act_col + (kb << 5);
                gate_sum += compute_block_dot_product_q8_0(gate_row + kb, act_blk);
                up_sum   += compute_block_dot_product_q8_0(up_row   + kb, act_blk);
            }

            float* dst_entry = (float*)(dst_data + n * nbd1 + m * sizeof(float));
            atomic_store_f32((volatile float*)dst_entry, ffn_silu_f32(gate_sum) * up_sum);
        }
    }

    return 0;
}
