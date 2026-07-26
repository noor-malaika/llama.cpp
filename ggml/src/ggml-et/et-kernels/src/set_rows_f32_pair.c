//******************************************************************************
// Two independent SET_ROWS in a single launch
//
// A decode layer writes the K cache and the V cache with two SET_ROWS ops.
// They are independent -- neither reads the other's output -- so they only
// need two launches because the graph emits them as two nodes.
//
// Board profiling showed launch cost is flat at ~110 us regardless of the work
// performed, and SET_ROWS alone accounts for 3136 launches / 11% of total time
// while copying only a few KB per call. Collapsing the pair removes one launch
// per layer: 16 x ~110 us = ~1.76 ms off a ~63 ms token.
//
// The work-splitting logic is not duplicated here. set_rows_f32.c's
// entry_point is pulled in under a different name and simply called twice;
// each call re-derives thread_id/num_threads from the same env, so every hart
// takes its share of the first copy and then its share of the second. No
// barrier is needed between them -- the two destinations are disjoint, and
// each hart's cache-line ownership within a copy is unchanged.
//******************************************************************************

// Reuse set_rows_f32.c wholesale, renaming its entry point. When this file is
// compiled into the combined uberkernel.elf, the build already redefines
// entry_point to a per-kernel unique name via -Dentry_point=<kernel>_entry
// (see et-kernels/CMakeLists.txt); push/pop instead of #undef so that outer
// rename survives the include instead of being destroyed by it -- an
// unconditional #undef here would leave our own entry_point below literally
// named "entry_point" in the uberkernel build, colliding with every other
// kernel's identically-named (before renaming) symbol at link time.
#pragma push_macro("entry_point")
#define entry_point set_rows_run_one
#include "set_rows_f32.c"
#pragma pop_macro("entry_point")

// Must stay layout-identical to ggml_et_set_rows_pair_params in ggml-et-ops.h.
struct ggml_et_set_rows_pair_params {
    struct ggml_et_set_rows_params a;
    struct ggml_et_set_rows_params b;
};

int entry_point(struct ggml_et_set_rows_pair_params* params, void* env) {
    if (!params) {
        return -1;
    }

    const int ra = set_rows_run_one(&params->a, env);
    if (ra != 0) {
        return ra;
    }

    return set_rows_run_one(&params->b, env);
}
