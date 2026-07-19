//******************************************************************************
// ET Platform Hardware Abstraction Layer
// Provides thread coordination, kernel infrastructure, and platform primitives
// for bare metal ET kernels
//******************************************************************************

#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>
#include "etsoc/isa/hart.h"
#include "etsoc/common/utils.h"

#define SOC_MINIONS_PER_SHIRE 32
#define NUM_HARTS_PER_MINION 2

// Environment structure definition
typedef struct {
    uint32_t version;           // Version of the ABI (offset 0)
    uint32_t padding1;          // Padding to align shire_mask to offset 8
    uint64_t shire_mask;        // Bitmask of active compute shires (offset 8)
    uint32_t frequency;         // Frequency of Minion cores in MHz (offset 16)
    uint32_t padding2;          // Padding to maintain alignment
} __attribute__((packed, aligned(64))) kernel_environment_t;

// Manual implementation of count trailing zeros for bare metal environment
// NOTE: This simple loop-based implementation is used for portability.
// Production implementations (like libgcc's __ctzdi2) use optimized bit manipulation
// algorithms with lookup tables and parallel bit operations for O(log n) performance.
static inline int manual_ctzll(uint64_t x) {
    if (x == 0) return 64;
    int count = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        count++;
    }
    return count;
}

// Manual implementation of population count for bare metal environment
// NOTE: This simple loop-based implementation is used for portability.
// Production implementations (like libgcc's __popcountdi2) use optimized bit-parallel
// algorithms with magic constants and bit manipulation tricks for O(1) performance.
static inline int manual_popcountll(uint64_t x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

// Calculate relative thread ID from absolute hart ID using shire mask
// Returns -1 if this hart is not active (not in shire mask)
static inline int get_relative_thread_id(uint64_t shire_mask) {
    int hart_id = (int)get_hart_id();

    // Find starting hart offset from lowest active shire
    int starting_hart = manual_ctzll(shire_mask) * SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION;

    // Return -1 if not an active thread
    if (hart_id < starting_hart) {
        return -1;
    }

    // Calculate relative thread ID
    int thread_id = hart_id - starting_hart;
    return thread_id;
}

// Calculate total number of threads from shire mask
static inline int get_num_threads(uint64_t shire_mask) {
    // Count active shires using popcount, multiply by minions per shire and harts per minion
    return manual_popcountll(shire_mask) * SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION;
}

//******************************************************************************
// Synchronization Primitives
//******************************************************************************

#define NOP   __asm__ __volatile__ ("nop\n");
#define FENCE __asm__ __volatile__ ("fence\n");
#define WFI   __asm__ __volatile__ ("wfi\n");

//******************************************************************************
// Tensor Engine Wait & Error Macros
//
// These write to CSR 0x830 (tensor_wait) to stall the hart until the specified
// tensor unit completes its current operation.  The immediate encodes which
// unit to wait on.
//******************************************************************************

#define WAIT_TENSOR_LOAD_0     __asm__ __volatile__ ( "csrwi 0x830, 0\n"  : : );
#define WAIT_TENSOR_LOAD_1     __asm__ __volatile__ ( "csrwi 0x830, 1\n"  : : );
#define WAIT_TENSOR_LOAD_L2_0  __asm__ __volatile__ ( "csrwi 0x830, 2\n"  : : );
#define WAIT_TENSOR_LOAD_L2_1  __asm__ __volatile__ ( "csrwi 0x830, 3\n"  : : );
#define WAIT_PREFETCH_0        __asm__ __volatile__ ( "csrwi 0x830, 4\n"  : : );
#define WAIT_PREFETCH_1        __asm__ __volatile__ ( "csrwi 0x830, 5\n"  : : );
#define WAIT_CACHEOPS          __asm__ __volatile__ ( "csrwi 0x830, 6\n"  : : );
#define WAIT_TENSOR_FMA        __asm__ __volatile__ ( "csrwi 0x830, 7\n"  : : );
#define WAIT_TENSOR_STORE      __asm__ __volatile__ ( "csrwi 0x830, 8\n"  : : );
#define WAIT_TENSOR_REDUCE     __asm__ __volatile__ ( "csrwi 0x830, 9\n"  : : );
#define WAIT_TENSOR_QUANT      __asm__ __volatile__ ( "csrwi 0x830, 10\n" : : );
#define STALL                  __asm__ __volatile__ ( "csrw stall, x0\n"  : : );

// Write 0 to CSR 0x808 (tensor_error) to clear any latched tensor error bits.
// Must be issued before the first tensor operation in a kernel to avoid stale
// errors from a previous invocation causing spurious faults.
#define CLEAR_TENSOR_ERROR     __asm__ __volatile__ ( "csrwi 0x808, 0" : : );

//******************************************************************************
// L2 Scratchpad (SCP) Addressing
//
// Format 0 (bit[30]=0): direct address into a specific shire's L2 SCP.
//   shire: 0-33 for explicit shire, L2SCP_SHIRE_LOCAL (0x7F) for local
//   offset: byte offset within the shire's scratchpad
//
// Shire ID 0x7F always targets the local shire (instead of figuring out which
// shire you are on).
//******************************************************************************

#define L2SCP_BASE        0x0080000000ULL
#define L2SCP_SHIRE_LOCAL 0x7FULL

static inline void * __attribute__((always_inline)) et_shire_l2scp(uint64_t shire, uint64_t offset) {
    return (void *) (L2SCP_BASE | ((shire & 0x7F) << 23) | (offset & 0x7FFFFF));
}

// Local shire shorthand - no cross-shire traffic.
static inline void * __attribute__((always_inline)) et_shire_l2scp_local(uint64_t offset) {
    return (void *) (L2SCP_BASE | (L2SCP_SHIRE_LOCAL << 23) | (offset & 0x7FFFFF));
}

//******************************************************************************
// Cache Operations
//******************************************************************************

// Flush nlines cache lines at stride apart starting at addr from L1 to L2 via
// FlushVA (CSR 0x8BF). Needed before a tensor_load/tensor_load_setup_b reads
// data a hart just wrote with ordinary stores: tensor loads bypass L1 and
// read from L2/memory directly, so a write sitting only in L1 is invisible to
// them without this. Caller must FENCE before and WAIT_CACHEOPS after.
//
// NOTE: nlines is encoded in a 4-bit field (max 16). DO NOT pass nlines > 16.
static inline void __attribute__((always_inline)) flush_to_l2(const void * addr, uint64_t nlines, uint64_t stride) {
    uint64_t csr_val = (0x1ULL << 58) | ((uint64_t) addr & 0xFFFFFFFFFFC0ULL) | ((nlines - 1) & 0xF);
    uint64_t x31_val = stride & 0xFFFFFFFFFFC0ULL;

    __asm__ __volatile__(
        "mv x31, %[x31]\n"
        "csrw 0x8BF, %[val]\n"
        :
        : [x31] "r"(x31_val), [val] "r"(csr_val)
        : "x31", "memory");
}

// Flush an arbitrary number of lines to L2, working around the 16-line cap of
// a single FlushVA by issuing multiple flushes.
static inline void __attribute__((always_inline)) flush_to_l2_multi(const void * addr, uint64_t nlines, uint64_t stride) {
    const char * p = (const char *) addr;
    while (nlines > 16) {
        flush_to_l2(p, 16, stride);
        p += 16 * stride;
        nlines -= 16;
    }
    if (nlines) {
        flush_to_l2(p, nlines, stride);
    }
}

// Evict nlines cache lines at stride apart starting at addr from L1 to L2.
// Uses EvictVA (CSR 0x89F). Unlike flush_to_l2, this guarantees the line is
// NOT present in L1 after the operation - subsequent loads will miss and go
// to L2/SCP. Caller must FENCE before and WAIT_CACHEOPS after.
//
// NOTE: nlines is encoded in a 4-bit field (max 16). DO NOT pass nlines > 16.
static inline void __attribute__((always_inline)) evict_to_l2(const void * addr, uint64_t nlines, uint64_t stride) {
    uint64_t csr_val = (0x1ULL << 58) | ((uint64_t) addr & 0xFFFFFFFFFFC0ULL) | ((nlines - 1) & 0xF);
    uint64_t x31_val = stride & 0xFFFFFFFFFFC0ULL;

    __asm__ __volatile__(
        "mv x31, %[x31]\n"
        "csrw 0x89F, %[val]\n"
        :
        : [x31] "r"(x31_val), [val] "r"(csr_val)
        : "x31", "memory");
}

//******************************************************************************
// Counter signaling between harts via L2 scratchpad (SCP)
//
// Used by a dual-hart producer/consumer pipeline: hart 1 dequantizes weight
// blocks into L2 SCP while hart 0 drives the tensor engine off the same
// scratch. Both counters live in L2 SCP so they are visible across harts once
// evicted from L1.
//******************************************************************************

// Signal a counter value to the other hart via L2 SCP.
static inline void __attribute__((always_inline))
scp_signal(volatile uint32_t *flag, uint32_t value) {
    *flag = value;
    FENCE;
    evict_to_l2((const void *)flag, 1, 64);
    WAIT_CACHEOPS;
}

// Wait for a counter in L2 SCP to reach the expected value.
static inline void __attribute__((always_inline))
scp_wait(volatile uint32_t *flag, uint32_t expected) {
    while (1) {
        evict_to_l2((const void *)flag, 1, 64);
        WAIT_CACHEOPS;
        if (*flag >= expected) return;
    }
}

//******************************************************************************
// L1 Data Cache / Scratchpad (SCP) Configuration
//
// The ET-SoC-1 L1 data cache can be split so that half its ways operate as a
// software-managed scratchpad (SCP).  Tensor load/store/FMA instructions
// require SCP mode to be active.
//
// CSR 0x810 — ucache_control:
//
//   Bit(s)  Field         Description
//   ──────  ────────────  ──────────────────────────────────────────────────
//   [0]     D1Split       1 = L1 is split (half cache, half SCP).
//                          Read-only from U-mode; set by M-mode firmware
//                          before kernel launch.  Writing ScpEnable while
//                          D1Split=0 is silently ignored.
//   [1]     ScpEnable     1 = scratchpad is active and zeroed.
//   [4:2]   RepRate       Cache-op replay rate (0 = no delay between ops).
//   [10:6]  CacheOpMax    Max outstanding cache ops (0 = unlimited).
//
// Typical kernel prologue for tensor operations:
//     setup_cache_scp();   // enables SCP, waits for zeroing
//     CLEAR_TENSOR_ERROR;  // clear stale error bits
//******************************************************************************

// Write the ucache_control CSR (0x810).
//
//   scp_en       — 1 to enable SCP mode (requires D1Split already set)
//   cacheop_rate — cache-op replay rate (0–7; 0 = no delay)
//   cacheop_max  — max outstanding cache ops (0–31; 0 = unlimited)
static inline void __attribute__((always_inline))
ucache_control(uint64_t scp_en, uint64_t cacheop_rate, uint64_t cacheop_max)
{
    uint64_t csr_enc = ((cacheop_max & 0x1F) << 6) |
                       ((cacheop_rate & 0x7)  << 2) |
                       ((scp_en & 0x1)        << 1);

    __asm__ __volatile__("csrw 0x810, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) : "x31");
}

// Enable L1 scratchpad mode and wait for the transition to complete.
// After this call the SCP lines are zeroed and ready for tensor operations.
//
// Prerequisites:
//   - D1Split must already be 1 (set by M-mode firmware at boot).
//   - Only even harts (hart 0 per minion) should call this, as only they
//     can issue tensor instructions.
static inline void setup_cache_scp(void)
{
    FENCE;                    // drain pending stores before reconfiguring cache
    ucache_control(1, 0, 0);  // ScpEnable=1
    WAIT_CACHEOPS;            // wait for SCP mode transition + zeroing
}

//******************************************************************************
// Atomic Operations
//******************************************************************************

// Atomic store for F32 values to global memory
// Uses ET hardware's custom amoswapg.w instruction for global atomic swap
// This ensures cache coherency when multiple threads write to nearby addresses
static inline void atomic_store_f32(volatile float* addr, float value) {
    uint32_t value_bits = *(uint32_t*)&value;
    __asm__ volatile(
        "amoswapg.w zero, %1, (%0)"
        :
        : "r"(addr), "r"(value_bits)
        : "memory"
    );
}

// Atomic add for F32 values to global memory
// Uses ET hardware's custom amoaddg.w instruction for global atomic add
// This ensures correct accumulation when multiple threads contribute to the same output
static inline void atomic_add_f32(volatile float* addr, float value) {
    uint32_t value_bits = *(uint32_t*)&value;
    __asm__ volatile(
        "amoaddg.w zero, %1, (%0)"
        :
        : "r"(addr), "r"(value_bits)
        : "memory"
    );
}

// Atomic store for F16 values to global memory
// Uses ET hardware's custom shg instruction (store halfword global)
// This ensures cache coherency when multiple threads write to nearby addresses
// Address must be 16-bit aligned
static inline void atomic_store_f16(volatile uint16_t* addr, uint16_t value) {
    __asm__ volatile(
        "shg %1, (%0)"
        :
        : "r"(addr), "r"(value)
        : "memory"
    );
}

#endif // PLATFORM_H
