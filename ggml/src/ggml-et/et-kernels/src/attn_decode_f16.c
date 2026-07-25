//******************************************************************************
// Fused decode attention (flash-style, single pass)
//
//   dst[h*head_dim + d] = sum_j softmax_j(q.k_j * scale + mask_j) * v_j[d]
//
// Replaces four kernel launches -- MUL_MAT(k,q), SOFT_MAX, MUL_MAT(v,kq) and
// the CONT that lays the result back out -- with one.
//
// Profiling on the board showed every ET launch costs a flat ~110 us
// regardless of the work it does (a 2048x2048 Q8_0 matmul costs 113 us; an
// RMS-norm over one vector costs 110 us). Decode attention is four launches
// per layer doing almost no arithmetic, so it is nearly pure launch overhead:
// ~440 us per layer to perform a few tens of microseconds of work.
//
// Parallelisation: one hart owns one (head, token) pair end to end. That is
// deliberate -- the only barrier primitive available here is et_barrier(),
// which is minion scope (two harts), so there is no way for a group of harts
// to cooperate on a single head's softmax reduction. Owning the whole head
// removes the need for any synchronisation at all.
//
// The softmax is computed online (running max + running sum, rescaling the
// accumulator when a new max appears) so the n_kv-sized score vector never
// has to be materialised anywhere -- no scratch buffer, and one pass over K
// and V instead of three.
//
// Decode only: the host fuses this exclusively when n_tokens == 1. With one
// hart per (head, token) a long prefill would put a full n_kv x head_dim
// serial scan on each hart, which is far slower than the unfused path.
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

// Largest head_dim this kernel will accept. The accumulator is held in a
// stack array of this size, so it bounds per-hart stack use (512 B).
#define ATTN_MAX_HEAD_DIM 128

// ggml masks disallowed positions with -INFINITY. Anything at or below this
// is treated as fully masked and skipped, which also keeps -inf out of the
// running maximum where it would poison the rescale factor.
#define ATTN_MASKED_THRESHOLD (-1.0e30f)

int entry_point(struct ggml_et_attn_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) {
        return -1;
    }

    const int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;  // this hart's shire is not part of the launch
    }
    const int64_t stride = get_num_threads(kernel_env->shire_mask);

    const int64_t head_dim = params->q.ne[0];
    const int64_t n_tokens = params->q.ne[1];
    const int64_t n_head   = params->q.ne[2];
    const int64_t n_kv     = params->k.ne[1];
    const int64_t n_kv_head = params->k.ne[2];

    if (head_dim > ATTN_MAX_HEAD_DIM || head_dim <= 0 || n_kv_head <= 0) {
        return -1;
    }

    // Grouped-query attention: several query heads share one KV head.
    const int64_t gqa = n_head / n_kv_head;
    if (gqa <= 0) {
        return -1;
    }

    const char* q_data    = (const char*)params->q.data;
    const char* k_data    = (const char*)params->k.data;
    const char* v_data    = (const char*)params->v.data;
    const char* mask_data = (const char*)params->mask.data;  // may be NULL
    char*       dst_data  = (char*)params->dst.data;

    if (!q_data || !k_data || !v_data || !dst_data) {
        return -1;
    }

    const size_t q_nb1 = params->q.nb[1];
    const size_t q_nb2 = params->q.nb[2];
    const size_t k_nb1 = params->k.nb[1];
    const size_t k_nb2 = params->k.nb[2];
    const size_t v_nb1 = params->v.nb[1];
    const size_t v_nb2 = params->v.nb[2];
    const size_t m_nb1 = params->mask.nb[1];
    const size_t d_nb1 = params->dst.nb[1];

    const float scale = params->scale;

    // One work unit per (head, token).
    const int64_t total = n_head * n_tokens;

    for (int64_t unit = thread_id; unit < total; unit += stride) {
        const int64_t t   = unit / n_head;
        const int64_t h   = unit % n_head;
        const int64_t kvh = h / gqa;

        const float* q_vec = (const float*)(q_data + h * q_nb2 + t * q_nb1);
        const char*  k_head = k_data + kvh * k_nb2;
        const char*  v_head = v_data + kvh * v_nb2;
        const float* m_row =
            mask_data ? (const float*)(mask_data + t * m_nb1) : 0;

        float acc[ATTN_MAX_HEAD_DIM];
        for (int64_t d = 0; d < head_dim; d++) {
            acc[d] = 0.0f;
        }

        float run_max = 0.0f;
        float run_sum = 0.0f;
        int   seen    = 0;  // whether run_max holds a real score yet

        for (int64_t j = 0; j < n_kv; j++) {
            // score = (q . k_j) * scale + mask_j
            const uint16_t* k_vec = (const uint16_t*)(k_head + j * k_nb1);
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; d++) {
                dot += q_vec[d] * fp16_to_fp32(k_vec[d]);
            }
            float x = dot * scale;
            if (m_row) {
                x += m_row[j];
            }

            // A fully masked position contributes nothing.
            if (x <= ATTN_MASKED_THRESHOLD) {
                continue;
            }

            // v is stored transposed: element (j, d) sits at d*v_nb1 + j*2.
            const char* v_col = v_head + (size_t)j * sizeof(uint16_t);

            if (!seen || x > run_max) {
                // First real score, or a new maximum: rescale what we have.
                const float c = seen ? et_expf(run_max - x) : 0.0f;
                run_sum = run_sum * c + 1.0f;
                seen    = 1;
                for (int64_t d = 0; d < head_dim; d++) {
                    const uint16_t vh = *(const uint16_t*)(v_col + d * v_nb1);
                    acc[d] = acc[d] * c + fp16_to_fp32(vh);
                }
                run_max = x;
            } else {
                const float p = et_expf(x - run_max);
                run_sum += p;
                for (int64_t d = 0; d < head_dim; d++) {
                    const uint16_t vh = *(const uint16_t*)(v_col + d * v_nb1);
                    acc[d] += p * fp16_to_fp32(vh);
                }
            }
        }

        // Write straight into the CONT layout: [head_dim*n_head, n_tokens].
        // Each hart owns head_dim contiguous floats, so no two harts share a
        // cache line as long as head_dim*4 is a multiple of 64 (64*4 = 256).
        const float inv = (run_sum > 0.0f) ? et_fdiv(1.0f, run_sum) : 0.0f;
        float* out = (float*)(dst_data + t * d_nb1) + h * head_dim;
        for (int64_t d = 0; d < head_dim; d++) {
            atomic_store_f32((volatile float*)(out + d), acc[d] * inv);
        }
    }

    return 0;
}
