//******************************************************************************
// Bare Metal CONT F32 Kernel -- packed-params variant
//
// Identical logic to cont_f32.c, but reads a minimal params struct (88 bytes)
// instead of two full ggml_tensor copies (672 bytes), to stay under the
// 128-byte DEVICE_OPS_KERNEL_LAUNCH_ARGS_PAYLOAD_MAX and avoid the runtime's
// oversized-args slow path (extra CMA alloc + host->device DMA + forced
// barrier -- see host-side KernelLaunch.cpp doKernelLaunch). Selected instead
// of cont_f32 via GGML_ET_PACKED_CONT=1 (see ggml-et-ops.h/.cpp).
//
// Must stay layout-identical to ggml_et_cont_params_packed in ggml-et-ops.h.
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdbool.h>
#include <stdint.h>

struct ggml_et_cont_params_packed {
    void*   src0_data;
    void*   dst_data;
    int64_t ne[4];
    int64_t nb[4];
    int32_t type;
};

static inline void atomic_copy_f32(float * dst, const float * src, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        atomic_store_f32((volatile float *) &dst[i], src[i]);
    }
}

int entry_point(struct ggml_et_cont_params_packed * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    if (params->type != GGML_TYPE_F32) {
        return -1;
    }

    float * src0_data = (float *) params->src0_data;
    float * dst_data  = (float *) params->dst_data;

    if (!src0_data || !dst_data) {
        return -1;
    }

    const int64_t ne00 = params->ne[0];
    const int64_t ne01 = params->ne[1];
    const int64_t ne02 = params->ne[2];
    const int64_t ne03 = params->ne[3];

    const int64_t nb00 = params->nb[0];
    const int64_t nb01 = params->nb[1];
    const int64_t nb02 = params->nb[2];
    const int64_t nb03 = params->nb[3];

    const int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    if (total_elements == 0) {
        return 0;
    }

    bool src_contiguous = true;
    {
        int64_t expected = 4; // sizeof(float)
        const int64_t ne_arr[4] = {ne00, ne01, ne02, ne03};
        const int64_t nb_arr[4] = {nb00, nb01, nb02, nb03};
        for (int i = 0; i < 4; i++) {
            if (ne_arr[i] > 1 && nb_arr[i] != expected) {
                src_contiguous = false;
                break;
            }
            expected *= ne_arr[i];
        }
    }

    // Fast path: src is contiguous, flat copy distributed by cache line.
    if (src_contiguous) {
        const int64_t elems_per_cl = 16;
        const int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
        const int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        if (cl_end > total_cl) {
            cl_end = total_cl;
        }
        if (cl_start >= total_cl) {
            return 0;
        }

        const int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        if (ee > total_elements) {
            ee = total_elements;
        }

        atomic_copy_f32(dst_data + es, src0_data + es, (int32_t) (ee - es));
        return 0;
    }

    if (nb00 != 4) {
        // Fully non-contiguous fallback: distribute by cache line, reverse-compute src coords.
        const int64_t elems_per_cl = 16;
        const int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
        const int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        if (cl_end > total_cl) {
            cl_end = total_cl;
        }
        if (cl_start >= total_cl) {
            return 0;
        }

        const int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        if (ee > total_elements) {
            ee = total_elements;
        }

        for (int64_t idx = es; idx < ee; idx++) {
            const int64_t i00  = idx % ne00;
            const int64_t rem1 = idx / ne00;
            const int64_t i01  = rem1 % ne01;
            const int64_t rem2 = rem1 / ne01;
            const int64_t i02  = rem2 % ne02;
            const int64_t i03  = rem2 / ne02;

            const float * sp =
                (const float *) ((const char *) src0_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx] = *sp;
        }
        return 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src.
    if (ne00 % 16 == 0) {
        const int64_t total_rows      = ne01 * ne02 * ne03;
        const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
        const int64_t start_row       = thread_id * rows_per_thread;
        const int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

        if (start_row >= total_rows) {
            return 0;
        }

        for (int64_t ir = start_row; ir < end_row; ir++) {
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            const float * src_row = (const float *) ((const char *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            float *       dst_row = dst_data + ir * ne00;

            atomic_copy_f32(dst_row, src_row, (int32_t) ne00);
        }
        return 0;
    }

    // Unaligned path: cache-line chunks, partial rows handled at each chunk boundary.
    {
        const int64_t elems_per_cl = 16;
        const int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
        const int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        if (cl_end > total_cl) {
            cl_end = total_cl;
        }
        if (cl_start >= total_cl) {
            return 0;
        }

        const int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        if (ee > total_elements) {
            ee = total_elements;
        }

        int64_t pos = es;

        int64_t row_idx = pos / ne00;
        int64_t col     = pos % ne00;

        while (pos < ee) {
            const int64_t i03 = row_idx / (ne02 * ne01);
            const int64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;

            const float * src_row = (const float *) ((const char *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);

            int64_t row_remaining   = ne00 - col;
            int64_t chunk_remaining = ee - pos;
            int32_t n               = (int32_t) (row_remaining < chunk_remaining ? row_remaining : chunk_remaining);

            atomic_copy_f32(dst_data + pos, src_row + col, n);

            pos += n;
            col = 0;  // subsequent rows start at column 0
            row_idx++;
        }
    }

    return 0;
}
