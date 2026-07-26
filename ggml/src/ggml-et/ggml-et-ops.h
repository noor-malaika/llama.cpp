#pragma once

#include "ggml-et-common.h"
#include "ggml.h"
#include <inttypes.h>

// Performance logging macros for ET ops
// Logs in machine-parseable pipe-delimited format: ET_PERF|field=value|...
#ifdef ET_PERF_RECORD
#define ET_PERF_START() int64_t _et_perf_start = ggml_time_us()

#define ET_PERF_END(op_name, kernel_name, node) do { \
    int64_t _et_perf_end = ggml_time_us(); \
    int64_t _et_perf_duration = _et_perf_end - _et_perf_start; \
    GGML_LOG_DEBUG("ET_PERF|op=%s|kernel=%s|duration_us=%" PRId64 "|tensor=%s|shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]|start_us=%" PRId64 "|end_us=%" PRId64 "\n", \
        op_name, kernel_name, _et_perf_duration, (node)->name, \
        (node)->ne[0], (node)->ne[1], (node)->ne[2], (node)->ne[3], \
        _et_perf_start, _et_perf_end); \
} while(0)

#define ET_PERF_END_EXT(op_name, kernel_name, node, fmt, ...) do { \
    int64_t _et_perf_end = ggml_time_us(); \
    int64_t _et_perf_duration = _et_perf_end - _et_perf_start; \
    GGML_LOG_DEBUG("ET_PERF|op=%s|kernel=%s|duration_us=%" PRId64 "|tensor=%s|shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]|start_us=%" PRId64 "|end_us=%" PRId64 "|" fmt "\n", \
        op_name, kernel_name, _et_perf_duration, (node)->name, \
        (node)->ne[0], (node)->ne[1], (node)->ne[2], (node)->ne[3], \
        _et_perf_start, _et_perf_end, ##__VA_ARGS__); \
} while(0)
#else

#define ET_PERF_START() do {} while(0)
#define ET_PERF_END_EXT(op_name, kernel_name, node, fmt, ...) do {(void)(node); } while(0)
#define ET_PERF_END(op_name, kernel_name, node) do {(void)(node);} while(0)

#endif

struct ggml_et_binary_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

// Q8_0 mul_mat with optional residual bias; bias.data == NULL means no bias (kernel skips the add).
struct ggml_et_mm_q8_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
    ggml_tensor bias;
    int32_t prefetch_rows;  // weight rows to prefetch ahead; 0 disables
    int32_t prefetch_dest;  // cache-op destination: 0 = L1, 1 = L2
    int32_t use_regdot;     // 1 = register-blocked dot (lever B), 0 = per-block compute_block_dot_product_q8_0
};

// Fused SwiGLU feed-forward: dst = silu(gate x act) * (up x act).
// Must stay layout-identical to ggml_et_mm_q8_ffn_params in
// et-kernels/src/ggml_tensor.h.
struct ggml_et_mm_q8_ffn_params {
    ggml_tensor gate;  // Q8_0 [K, n_ff] -- operand that receives silu()
    ggml_tensor up;    // Q8_0 [K, n_ff]
    ggml_tensor act;   // F32  [K, N] shared activation
    ggml_tensor dst;   // F32  [n_ff, N]
    int32_t prefetch_rows;  // weight rows to prefetch ahead; 0 disables
    int32_t prefetch_dest;  // cache-op destination: 0 = L1, 1 = L2
};

// Rows of Q8_0 weights each hart prefetches ahead of the row it is computing,
// from GGML_ET_PREFETCH_ROWS. 0 (default) disables prefetching entirely.
int32_t ggml_et_prefetch_rows();

// Cache level prefetched weight rows land in: 0 = L1, 1 = L2 (default).
// Both our -9% result and DarthCeltic's -3.3% (PR #170) prefetched to L2;
// each hart owns whole rows, so there is no sharing that justifies stopping
// short of L1. GGML_ET_PREFETCH_DEST selects it.
int32_t ggml_et_prefetch_dest();

// Default off: register-blocked Q8_0 row dot (lever B), ported from
// lever-b-register-dot (0a8556ac8). Defers the horizontal 8-lane reduction
// from once-per-block to once-per-row instead of compute_block_dot_product_q8_0's
// per-block reduction, and hoists the vector-mask setup out of the row loop.
// Scalar mul_mat_Q8_0 kernel only.
bool ggml_et_regdot_enabled();

// Element map parameters for embarrassingly parallel binary operations (MUL, ADD, etc.)
// Operation type is determined by dst->op (GGML_OP_MUL, GGML_OP_ADD, etc.)
struct ggml_et_elmap_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

typedef struct {
    int32_t n_past;
    int32_t n_dims;        // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;          // ROPE mode (0=normal, 1=neox, 2=glm)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    float   freq_base;     // Base frequency (usually 10000.0f)
    float   freq_scale;    // Frequency scaling factor
    float   ext_factor;    // Extension factor for YaRN
    float   attn_factor;   // Attention factor for YaRN
    float   beta_fast;     // Fast beta for YaRN
    float   beta_slow;     // Slow beta for YaRN
    int32_t sections[4];   // Sections for multi-modal ROPE
} rope_params_t;

struct ggml_et_rope_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor src2;
    ggml_tensor dst;
    rope_params_t rope_params;
};

struct ggml_et_rms_norm_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
    float eps;         // Epsilon parameter for numerical stability
};

struct ggml_et_norm_params {
    ggml_tensor src0;
    ggml_tensor dst;
    float eps;
};

struct ggml_et_unary_params {
    ggml_tensor src0;
    ggml_tensor dst;
    int32_t unary_op;
};

struct ggml_et_im2col_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
    int32_t s0;
    int32_t s1;
    int32_t p0;
    int32_t p1;
    int32_t d0;
    int32_t d1;
    int32_t is_2d;
};

struct ggml_et_glu_params {
    ggml_tensor src0;     // F32 input tensor A (or combined tensor if src1 is null)
    ggml_tensor src1;     // F32 input tensor B (null for single tensor mode)
    ggml_tensor dst;      // F32 output tensor (n/2 columns)
    int32_t glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t swapped;      // Whether gate and value are swapped
};

struct ggml_et_softmax_params {
    ggml_tensor src0;     // F32 input tensor
    ggml_tensor src1;     // F32 mask tensor (optional, may be zeroed if not used)
    ggml_tensor src2;     // F32 sinks tensor (optional, may be zeroed if not used)
    ggml_tensor dst;      // F32 output tensor
    float scale;          // Scale factor
    float max_bias;       // Max bias for ALiBi (0.0f if not used)
};

struct ggml_et_get_rows_params {
    ggml_tensor src0;     // Data tensor (F32 or Q8_0)
    ggml_tensor src1;     // Row indices tensor (I32)
    ggml_tensor dst;      // Output tensor (F32)
};

struct ggml_et_cont_params {
    ggml_tensor src0;     // F32 input tensor (non-contiguous)
    ggml_tensor dst;      // F32 output tensor (contiguous)
};

// Minimal CONT params: only the fields cont_f32.c/cont_f16.c actually read
// from src0/dst (data pointer, ne, nb; dst needs only its data pointer).
// 88 bytes vs ggml_et_cont_params's 672 -- fits under the 128-byte
// DEVICE_OPS_KERNEL_LAUNCH_ARGS_PAYLOAD_MAX, so the runtime embeds it
// directly instead of taking the oversized-args slow path (extra CMA alloc
// + host->device DMA + forced barrier -- see KernelLaunch.cpp doKernelLaunch).
// Must stay layout-identical to ggml_et_cont_params_packed in
// et-kernels/src/cont_f32.c and cont_f16.c.
struct ggml_et_cont_params_packed {
    void*   src0_data;
    void*   dst_data;
    int64_t ne[4];   // src0 ne
    int64_t nb[4];   // src0 nb
    int32_t type;    // GGML_TYPE_F32 or GGML_TYPE_F16, for the kernel's existing safety check
};

struct ggml_et_set_rows_params {
    ggml_tensor src0;     // F32 source data tensor
    ggml_tensor src1;     // I64 row indices tensor
    ggml_tensor dst;      // F32/F16 destination tensor
};

// Two independent SET_ROWS fused into one launch. Must stay layout-identical
// to ggml_et_set_rows_pair_params in et-kernels/src/set_rows_f32_pair.c.
struct ggml_et_set_rows_pair_params {
    ggml_et_set_rows_params a;
    ggml_et_set_rows_params b;
};

struct ggml_et_rms_norm_mul_params {
    ggml_tensor src0;      // F32 input tensor (to be normalized)
    ggml_tensor src1;      // F32 weights tensor (element-wise multiply)
    ggml_tensor dst;       // F32 output tensor
    float eps;             // Epsilon for numerical stability
};

struct ggml_et_mul_mat_id_params {
    ggml_tensor src0;     // Expert weight matrices (Q8_0/F16/F32) [K, M, n_expert]
    ggml_tensor src1;     // Activations (F32) [K, n_expert_used, batch]
    ggml_tensor src2;     // Expert indices (I32) [n_expert_used, batch]
    ggml_tensor dst;      // Output (F32) [M, n_expert_used, batch, 1]
};

struct ggml_et_scale_params {
    ggml_tensor src0;     // F32 input tensor
    ggml_tensor dst;      // F32 output tensor
    float scale;          // Scale factor
    float bias;           // Bias (additive offset)
};

bool ggml_et_op_scale(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_mul(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_add(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_sub(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_mul_mat(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);

// Fusion helpers. Both are runtime-gated (see ggml-et-ops.cpp) so a single build
// can be measured with each fusion on or off.
//
// True when this MUL_MAT would be dispatched to the scalar Q8_0 kernel, which is
// the only mul_mat kernel that honours the fused-bias slot.
bool ggml_et_mul_mat_is_scalar_q8(const ggml_tensor* node);
bool ggml_et_fuse_mm_add_enabled();
bool ggml_et_fuse_ffn_enabled();

// dst = (mul_mat_node) + addend, in one launch.
bool ggml_et_op_mul_mat_add(ggml_backend_et_device_context* dev_ctx,
                            const ggml_tensor* mul_mat_node,
                            const ggml_tensor* add_node);

// dst = silu(gate x act) * (up x act), in one launch.
bool ggml_et_op_mul_mat_ffn_glu(ggml_backend_et_device_context* dev_ctx,
                                const ggml_tensor* gate_node,
                                const ggml_tensor* up_node,
                                const ggml_tensor* glu_node);
bool ggml_et_fuse_set_rows_enabled();

// Default off: exploratory probe for whether shrinking launch params under
// the 128-byte embedding limit (DEVICE_OPS_KERNEL_LAUNCH_ARGS_PAYLOAD_MAX)
// measurably cuts per-launch host enqueue cost. Selects cont_f32_packed /
// cont_f16_packed (88-byte params) over cont_f32 / cont_f16 (672-byte, full
// ggml_tensor copies -- takes the runtime's oversized-args slow path: extra
// CMA alloc, host->device DMA, forced barrier). See KernelLaunch.cpp
// doKernelLaunch for the exact slow-path mechanism this bypasses.
bool ggml_et_packed_cont_enabled();

// Two independent SET_ROWS (K cache and V cache) in one launch.
bool ggml_et_op_set_rows_pair(ggml_backend_et_device_context* dev_ctx,
                              const ggml_tensor* first,
                              const ggml_tensor* second);
bool ggml_et_op_mul_mat_id(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_rope(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_rms_norm(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_norm(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_unary(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_im2col(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_glu(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_softmax(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_get_rows(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_set_rows(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_cont(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_elmap(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_rms_norm_mul(ggml_backend_et_device_context* dev_ctx,
                             const ggml_tensor* rms_norm_node,
                             const ggml_tensor* mul_node);
