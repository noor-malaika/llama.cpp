//******************************************************************************
// Bare Metal CONT F16 Kernel -- packed-params variant
//
// Identical logic to cont_f16.c, but reads a minimal params struct (88 bytes)
// instead of two full ggml_tensor copies (672 bytes). See cont_f32_packed.c
// for the full rationale. Selected instead of cont_f16 via
// GGML_ET_PACKED_CONT=1 (see ggml-et-ops.h/.cpp).
//
// Must stay layout-identical to ggml_et_cont_params_packed in ggml-et-ops.h.
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include "ggml_tensor.h"
#include "platform.h"

struct ggml_et_cont_params_packed {
    void*   src0_data;
    void*   dst_data;
    int64_t ne[4];
    int64_t nb[4];
    int32_t type;
};

int entry_point(struct ggml_et_cont_params_packed* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    if (params->type != GGML_TYPE_F16) {
        return -1; // Unsupported type
    }

    uint16_t* src0_data = (uint16_t*)params->src0_data;
    uint16_t* dst_data = (uint16_t*)params->dst_data;

    if (!src0_data || !dst_data) {
        return -1; // Null data pointer
    }

    // Source tensor dimensions and strides
    const int64_t ne00 = params->ne[0];
    const int64_t ne01 = params->ne[1];
    const int64_t ne02 = params->ne[2];
    const int64_t ne03 = params->ne[3];

    const int64_t nb00 = params->nb[0];
    const int64_t nb01 = params->nb[1];
    const int64_t nb02 = params->nb[2];
    const int64_t nb03 = params->nb[3];

    // Parallelize by rows (dimension 1)
    const int64_t total_rows = ne01;
    const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
    const int64_t start_row = thread_id * rows_per_thread;
    const int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    if (start_row >= total_rows) {
        return 0;
    }

    // Iterate over source tensor dimensions
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            const int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            for (int64_t i01 = start_row; i01 < end_row; i01++) {
                // Linear index for start of this row in destination
                const int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                // Inner loop over dimension 0
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    const int64_t src_offset_bytes = i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                    const uint16_t* src_ptr = (const uint16_t*)((const char*)src0_data + src_offset_bytes);

                    // Destination linear index (contiguous layout)
                    const int64_t dst_linear_idx = dst_linear_row_base + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((volatile uint16_t*)&dst_data[dst_linear_idx], *src_ptr);
                }
            }
        }
    }

    return 0;
}
