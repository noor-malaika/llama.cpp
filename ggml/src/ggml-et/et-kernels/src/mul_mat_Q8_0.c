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
#include "tensor.h"

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

int entry_point(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    const int64_t stride_m = 2048;

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

    // Q8_0 block size is 32
    const int64_t K_blocks = K / 32;

    // Broadcasting ratios
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    // Stage the activation vector (src1) into per-shire L2 SCP once per call,
    // so every one of up to M rows' dot product reads it from on-chip memory
    // instead of re-fetching it from DRAM. Decode's B is small (N=1, K up to
    // 8192 => <=32KB) and every row's dot product re-reads all of it, so
    // without staging, streaming the (large) weight matrix through cache
    // repeatedly evicts this small, reused vector between rows. Only applies
    // to the common no-broadcast case (matches every per-layer projection and
    // lm_head; matmul_id / batched-broadcast shapes fall back to the
    // untouched direct-DRAM-read path below).
    //
    // NOTE (unverified as of this port): et_tensor_load_l2scp is declared in
    // tensor.h and used by no other kernel in this tree currently - there is
    // no proven mainline call site to confirm its prerequisites against, only
    // this port's own source (an unmerged branch, itself never board-verified
    // as far as this repo's history shows). First thing to sanity-check once
    // hardware is available again.
    const int      b_contig = (nb11 == (size_t) K * sizeof(float));
    const uint64_t b_lines  = ((uint64_t) N * K * sizeof(float) + 63) / 64;
    const int      stage_b  = b_contig && ne12 == 1 && ne13 == 1 && ne02 == 1 && ne03 == 1 &&
                              b_lines <= 8192;  // <= 512 KB, within the per-shire SCP budget
    const float * b_scp = (const float *) et_shire_l2scp_local(0);

    if (stage_b) {
        if ((hart_id & 63) == 0) {
            et_tensor_load_l2scp_conf_t conf;
            conf.use_tmask = false;
            conf.stride    = 64;
            uint64_t remaining = b_lines;
            uint64_t dst_ln    = 0;
            uint64_t addr      = (uint64_t) params->src1.data;
            while (remaining > 0) {
                uint64_t cl = (remaining >= 16) ? 16 : remaining;
                conf.dst_start = dst_ln;
                conf.addr      = addr;
                conf.num_lines = cl - 1;   // 4-bit field encodes (lines - 1)
                conf.id        = 0;
                et_tensor_load_l2scp(&conf);
                WAIT_TENSOR_LOAD_L2_0;
                dst_ln    += cl;
                addr      += cl * 64;
                remaining -= cl;
            }
        }
        et_barrier(ET_BARRIER_SHIRE);   // B is now resident in L2 SCP for every hart in this shire
    }

    // Vector mask (all 8 lanes) is the same for every block of every row of
    // the whole call - set it once here instead of once per block inside the
    // dot product (q8_0_dot_tile no longer touches it).
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t i3 = 0; i3 < ne13; i3++) {
        const int64_t i03 = i3 / r3;
        const char* src0_ptr3 = (const char*)params->src0.data + i03 * nb03;
        const char* src1_ptr3 = (const char*)params->src1.data + i3 * nb13;
        char* dst_ptr3       = (char*)params->dst.data + i3 * nbd3;

        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const int64_t i02 = i2 / r2;
            const char* src0_ptr2 = src0_ptr3 + i02 * nb02;
            const char* src1_ptr2 = src1_ptr3 + i2 * nb12;
            char* dst_ptr2       = dst_ptr3 + i2 * nbd2;

            for (int64_t n = 0; n < N; n++) {
                // src1 is F32, so column pointer moves by nb11. If staged,
                // read from on-chip L2 SCP instead (laid out as N contiguous
                // rows of K floats each, written by the staging loop above).
                const float* b_col_base = stage_b
                    ? (b_scp + n * K)
                    : (const float*)(src1_ptr2 + n * nb11);

                for (int64_t m = hart_id; m < M; m += stride_m) {
                    // src0 is Q8_0 blocks, row pointer moves by nb01
                    const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);

                    q8_0_dot_reset();
                    for (int64_t kb = 0; kb < K_blocks; kb++) {
                        // b_col is float*, so a block (32 elements) moves by (kb << 5)
                        q8_0_dot_tile(q_row, b_col_base + (kb << 5), kb, K_blocks);
                    }
                    float sum = q8_0_dot_reduce();

                    // Store result in dst[m, n, i2, i3]
                    float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                    atomic_store_f32((volatile float*)dst_entry, sum);
                }
            }
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return 0;
}
