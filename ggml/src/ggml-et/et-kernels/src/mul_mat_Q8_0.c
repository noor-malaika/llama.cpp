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
    // Reserve the first 4KB of this shire's L2 SCP for lever C2's per-group
    // coordination scratch (see below) so the two levers' scratch usage can
    // never collide when stacked together. Real per-shire SCP capacity is
    // NOT documented anywhere found in this repo (project notes record a
    // prior hard crash - garbage stream-sync errors, not a clean failure -
    // from a different kernel exceeding it); this reservation is a
    // known-safe-relative-to-lever-A-alone shift, not proof the combined
    // footprint fits real hardware. Board-verify before trusting.
    #define LEVER_C2_SCRATCH_BYTES 4096

    const int      b_contig = (nb11 == (size_t) K * sizeof(float));
    const uint64_t b_lines  = ((uint64_t) N * K * sizeof(float) + 63) / 64;
    const int      stage_b  = b_contig && ne12 == 1 && ne13 == 1 && ne02 == 1 && ne03 == 1 &&
                              b_lines <= 8128;  // <= 508 KB, leaves room for the 4KB reservation above
    const float * b_scp = (const float *) et_shire_l2scp_local(LEVER_C2_SCRATCH_BYTES);

    // Lever C2 group-coordination scratch: 4 groups of 16 harts per shire
    // (64 harts/shire / 16), each group gets 16 slots of 64B (one per
    // group member - result float at offset 0, "ready" flag at offset 4,
    // rest padding). One slot per cache line so 16 different harts writing
    // their own slot concurrently never share a line - the same
    // false-sharing hazard this whole file exists to remove would otherwise
    // just move into the coordination mechanism itself.
    void * c2_scratch_base = et_shire_l2scp_local(0);

    if ((hart_id & 63) == 0) {
        // Zero the whole reserved region once per call, before anyone reads
        // or writes a flag - piggybacks on the shire barrier below instead
        // of needing a separate one.
        volatile uint32_t * z = (volatile uint32_t *) c2_scratch_base;
        for (uint64_t i = 0; i < LEVER_C2_SCRATCH_BYTES / sizeof(uint32_t); i++) {
            z[i] = 0;
        }
    }

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
    }
    // Shire barrier covers both: staged B (if any) and the zeroed C2
    // scratch are both visible to every hart in the shire past this point.
    et_barrier(ET_BARRIER_SHIRE);

    // C2 only activates when every hart in every 16-hart group has the same
    // trip count through the m-loop below (M % 16 == 0) - guarantees no
    // group member is ever missing when the leader waits for all 16 flags.
    // This model's actual shapes (2048/8192/128256, all multiples of 16)
    // always satisfy this; falls back to the plain atomic store otherwise.
    const int c2_active = (M % 16) == 0;
    const int c2_local_idx = (int) (hart_id % 16);   // 0 = leader of its group
    char * c2_group_scratch = (char *) c2_scratch_base +
        (uint64_t) ((hart_id % 64) / 16) * (16 * 64);

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

                    if (!c2_active) {
                        // Fallback: plain per-row atomic store (identical to
                        // lever B's behavior), used whenever group trip
                        // counts aren't guaranteed equal.
                        float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                        atomic_store_f32((volatile float*)dst_entry, sum);
                        continue;
                    }

                    // Lever C2: full 2048-way compute parallelism kept (every
                    // hart still computes exactly one row, same as lever B) -
                    // only the final write is rerouted through this group's
                    // scratch so only the leader (local_idx==0) ever touches
                    // the real dst array, once per group, with a plain store.
                    char * my_slot          = c2_group_scratch + c2_local_idx * 64;
                    float * my_result_slot  = (float *) my_slot;
                    volatile uint32_t * my_flag = (volatile uint32_t *) (my_slot + 4);

                    // Wait for the PREVIOUS round's leader to have consumed
                    // and cleared this slot before overwriting it (double
                    // handshake - without this, a fast hart could reach its
                    // next iteration's write before the leader finishes
                    // reading the current one). No-op on the first
                    // iteration since the zeroing above already left every
                    // flag at 0.
                    while (*my_flag != 0) {
                        evict_to_l2((const void *) my_flag, 1, 64);
                        WAIT_CACHEOPS;
                        FENCE;
                    }

                    *my_result_slot = sum;
                    FENCE;
                    *my_flag = 1;
                    FENCE;
                    evict_to_l2((const void *) my_slot, 1, 64);
                    WAIT_CACHEOPS;
                    FENCE;

                    if (c2_local_idx == 0) {
                        for (int j = 1; j < 16; j++) {
                            volatile uint32_t * flag_j =
                                (volatile uint32_t *) (c2_group_scratch + j * 64 + 4);
                            while (*flag_j == 0) {
                                evict_to_l2((const void *) flag_j, 1, 64);
                                WAIT_CACHEOPS;
                                FENCE;
                            }
                        }

                        const int64_t group_m0   = m;   // leader's own m == group start
                        const int64_t rows_here  = (group_m0 + 16 <= M) ? 16 : (M - group_m0);
                        float results[16];
                        for (int j = 0; j < rows_here; j++) {
                            results[j] = *(float *) (c2_group_scratch + j * 64);
                        }

                        // Sole writer to this dst range for this group - plain store.
                        float* dst_block = (float*)(dst_ptr2 + n * nbd1 + group_m0 * sizeof(float));
                        for (int j = 0; j < rows_here; j++) {
                            dst_block[j] = results[j];
                        }

                        // Clear all 16 flags, releasing every group member
                        // (including this leader) to write the next round.
                        for (int j = 0; j < 16; j++) {
                            volatile uint32_t * flag_j =
                                (volatile uint32_t *) (c2_group_scratch + j * 64 + 4);
                            *flag_j = 0;
                        }
                        FENCE;
                        evict_to_l2((const void *) c2_group_scratch, 16, 64);
                        WAIT_CACHEOPS;
                        FENCE;
                    }
                }
            }
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return 0;
}
