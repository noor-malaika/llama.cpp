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

// Pull a byte range into L2 for the calling hart.
//
// Decode is latency-bound, not bandwidth-bound: measured 19.7 GB/s against a
// ~68 GB/s DDR peak. Each hart walks its weight row as a dependent
// load -> dot -> accumulate chain, so an in-order minion stalls on every miss
// with nothing else in flight. Issuing the row this hart will need *next*
// while it computes the current one gives the memory system a full row of
// lead time. Unlike prefetch_weight_row() below, this covers a single hart's
// own row -- there is no cross-hart split, because each hart owns whole rows.
static inline void et_prefetch_range(const void* start_ptr, int64_t num_bytes) {
    if (num_bytes <= 0) {
        return;
    }
    uintptr_t ptr     = (uintptr_t)start_ptr & ~(uintptr_t)63;
    uintptr_t end_ptr = (uintptr_t)start_ptr + (uintptr_t)num_bytes;
    int64_t   lines   = (int64_t)(((end_ptr - ptr) + 63) >> 6);

    // The CSR takes at most 16 lines per issue (count encoded in bits 3:0).
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

// Using the block prefetch logic
static inline void prefetch_weight_row(const void* start_ptr, int64_t num_blocks, uint32_t worker_id) {
    const uint64_t cache_line_size = 64;
    uintptr_t self_ptr = (uintptr_t)start_ptr;
    uintptr_t self_ptr_end = self_ptr + (num_blocks * sizeof(block_q8_0));

    // 1. Align to cache lines and calculate range
    uint64_t startCL = (self_ptr + 63) >> 6;
    uint64_t endCL = (self_ptr_end + 63) >> 6;
    if (endCL >= startCL) {
        uint64_t total_lines = endCL - startCL + 1;
        // 2. Load balance across the minions in the Shire (assuming 8 per group for this logic)
        // Adjust worker_id if using global_id
        uint32_t local_worker_id = worker_id % 8;
        uint64_t lines_per_minion = total_lines >> 3;
        uint64_t extra = total_lines & 7;

        uint64_t offset = local_worker_id * lines_per_minion;
        if (local_worker_id < extra) {
            offset += local_worker_id;
            lines_per_minion++;
        } else {
            offset += extra;
        }
        self_ptr = (startCL + offset) << 6;
        int pending_lines = lines_per_minion;

        // 3. Hardware Prefetch Loop (16 lines at a time)
        for (; pending_lines > 0; pending_lines -= 16) {
            uint64_t current_batch = (pending_lines > 16 ? 16 : pending_lines) - 1;
            uint64_t self_size = current_batch; // bits 3:0

            __asm__ __volatile__ (
                "li    x1, 0x400000000000000 \n"  // Dest = L2 (bits 59:58 = 01)
                "addi  x31, zero, 64\n"  // Stride = 64 bytes
                "or    x3, x1, %[ptr]\n"  // Combine Dest + VA
                "or    x3, x3, %[sz]\n"  // Combine with NumLines
                "csrw  0x81f, x3\n"  // prefetch_va
                :
                : [ptr] "r" (self_ptr),
                  [sz] "r" (self_size)
                : "x1", "x3", "x31", "memory"
            );
            self_ptr += (16 * 64);
        }
    }
}

int entry_point(struct ggml_et_mm_q8_params* params, void* env) {
    // Rows are dealt round-robin across the harts that were actually launched.
    // The stride used to be the constant 2048 (32 shires x 32 minions x 2 harts),
    // which silently required every launch to use all 32 shires. Deriving it from
    // the mask instead lets a caller launch a small matmul on fewer shires without
    // dropping rows; with the full mask this is still exactly 2048, so the default
    // path is unchanged.
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) {
        return -1;
    }

    const int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;  // this hart's shire is not part of the launch
    }

    const uint64_t hart_id  = (uint64_t)thread_id;
    const int64_t  stride_m = get_num_threads(kernel_env->shire_mask);

    // Matrix dimensions
    const int64_t K    = params->src0.ne[0];
    const int64_t M    = params->src0.ne[1];
    const int64_t N    = params->src1.ne[1];
    const int64_t ne02 = params->src0.ne[2];
    const int64_t ne03 = params->src0.ne[3];
    const int64_t ne12 = params->src1.ne[2];
    const int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    const size_t nb01 = params->src0.nb[1];
    const size_t nb02 = params->src0.nb[2];
    const size_t nb03 = params->src0.nb[3];

    const size_t nb11 = params->src1.nb[1];
    const size_t nb12 = params->src1.nb[2];
    const size_t nb13 = params->src1.nb[3];

    const size_t nbd1 = params->dst.nb[1];
    const size_t nbd2 = params->dst.nb[2];
    const size_t nbd3 = params->dst.nb[3];

    // Fused residual add. The host sets bias only when it has verified that the
    // addend has exactly the same shape and strides as dst, so the same index
    // arithmetic addresses both. bias.data == NULL means a plain MUL_MAT.
    const char* bias_data = (const char*)params->bias.data;

    // Q8_0 block size is 32
    const int64_t K_blocks = K / 32;

    // Bytes of Q8_0 weight actually consumed per row. Taken from the block
    // count rather than nb01 so a padded row stride never over-prefetches.
    const int64_t row_bytes      = K_blocks * (int64_t)sizeof(block_q8_0);
    const int32_t prefetch_rows  = params->prefetch_rows;

    // Broadcasting ratios
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i3 = 0; i3 < ne13; i3++) {
        const int64_t i03 = i3 / r3;
        const char* src0_ptr3 = (const char*)params->src0.data + i03 * nb03;
        const char* src1_ptr3 = (const char*)params->src1.data + i3 * nb13;
        char* dst_ptr3       = (char*)params->dst.data + i3 * nbd3;
        const char* bias_ptr3 = bias_data ? bias_data + i3 * nbd3 : 0;

        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const int64_t i02 = i2 / r2;
            const char* src0_ptr2 = src0_ptr3 + i02 * nb02;
            const char* src1_ptr2 = src1_ptr3 + i2 * nb12;
            char* dst_ptr2       = dst_ptr3 + i2 * nbd2;
            const char* bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbd2 : 0;

            for (int64_t n = 0; n < N; n++) {
                // src1 is F32, so column pointer moves by nb11
                const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                // Warm L2 with this hart's first row before the chain starts.
                if (prefetch_rows > 0 && (int64_t)hart_id < M) {
                    et_prefetch_range(src0_ptr2 + (int64_t)hart_id * nb01, row_bytes);
                }

                for (int64_t m = hart_id; m < M; m += stride_m) {
                    // Stay prefetch_rows iterations ahead of the row being consumed.
                    if (prefetch_rows > 0) {
                        const int64_t ahead = m + (int64_t)prefetch_rows * stride_m;
                        if (ahead < M) {
                            et_prefetch_range(src0_ptr2 + ahead * nb01, row_bytes);
                        }
                    }

                    // src0 is Q8_0 blocks, row pointer moves by nb01
                    const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                    float sum = 0.0f;

                    for (int64_t kb = 0; kb < K_blocks; kb++) {
                        // q_row is a pointer to blocks, so + kb moves by sizeof(block_q8_0)
                        // b_col is float*, so we move 32 elements (kb << 5)
                        sum += compute_block_dot_product_q8_0(q_row + kb, b_col_base + (kb << 5));
                    }

                    if (bias_ptr2) {
                        sum += *(const float*)(bias_ptr2 + n * nbd1 + m * sizeof(float));
                    }

                    // Store result in dst[m, n, i2, i3]
                    float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                    atomic_store_f32((volatile float*)dst_entry, sum);
                }
            }
        }
    }
    return 0;
}
