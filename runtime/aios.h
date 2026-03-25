/**
 * aios.h — AIOS Runtime Public ABI
 *
 * The AIOS runtime is a memory residency controller between inference engines
 * and hardware. It manages how weight data moves from DRAM to compute units.
 *
 * Integration:
 *   - Link against libaios.so (ABI integration — recommended)
 *   - LD_PRELOAD=libaios_preload.so (zero-modification — any pre-compiled binary)
 *
 * The runtime does exactly four things at inference time:
 *   1. Resolve a pointer (aios_resolve_block)
 *   2. Apply a delta (aios_apply_delta — called by resolve if aliased)
 *   3. Skip a sparse row (aios_is_active)
 *   4. Issue a static prefetch (managed internally on prefetch thread)
 *
 * Everything else happens in the profiler — offline, before inference begins.
 *
 * Status: STUB — interfaces defined, implementation needed.
 * See runtime/README.md for implementation guide.
 */

#ifndef AIOS_H
#define AIOS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ────────────────────────────────────────────────────────────────── */
#define AIOS_VERSION_MAJOR 0
#define AIOS_VERSION_MINOR 1
#define AIOS_VERSION_PATCH 0
#define AIOS_VERSION_STRING "0.1.0-dev"

/* ── Residency tiers ─────────────────────────────────────────────────────────── */
typedef enum {
    AIOS_TIER_HOT    = 0,   /* Best-effort memory pinning; last to evict         */
    AIOS_TIER_WARM   = 1,   /* Prefer-resident; shrinks first under pressure     */
    AIOS_TIER_STREAM = 2,   /* Fully evictable; sequential streaming layout      */
} aios_residency_tier_t;

/* ── ISA capability tiers ────────────────────────────────────────────────────── */
typedef enum {
    AIOS_ISA_TIER_1 = 1,   /* SSE4.2 (x86) / NEON (ARM) — minimum floor        */
    AIOS_ISA_TIER_2 = 2,   /* AVX2                                               */
    AIOS_ISA_TIER_3 = 3,   /* AVX-512 / AMX                                      */
    AIOS_ISA_APPLE  = 4,   /* Apple AMX / ANE (unified memory)                   */
} aios_isa_tier_t;

/* ── Block flags ─────────────────────────────────────────────────────────────── */
#define AIOS_FLAG_CANONICAL  (1 << 0)   /* Block is a canonical (non-aliased) block */
#define AIOS_FLAG_ALIASED    (1 << 1)   /* Block aliases a canonical; delta != NULL */
#define AIOS_FLAG_SPARSE     (1 << 2)   /* Block is marked inactive by sparsity map */
#define AIOS_FLAG_HOT        (1 << 3)   /* Block assigned to RWS-HOT tier           */
#define AIOS_FLAG_WARM       (1 << 4)   /* Block assigned to RWS-WARM tier          */

/* ── Pointer table entry ─────────────────────────────────────────────────────── */
typedef struct {
    void    *base_pointer;      /* Physical address in canonical weight store     */
    uint32_t block_length;      /* Length in bytes                                */
    uint32_t flags;             /* Combination of AIOS_FLAG_* above               */
    void    *delta_pointer;     /* Delta vector; NULL if AIOS_FLAG_CANONICAL      */
    uint32_t delta_length;      /* Delta length in bytes; 0 if no delta           */
    uint32_t _reserved;         /* Pad to 32 bytes                                */
} aios_block_entry_t;

/* ── Hardware profile ────────────────────────────────────────────────────────── */
typedef struct {
    aios_isa_tier_t isa_tier;
    uint32_t        cpu_cores;
    uint64_t        l3_cache_bytes;
    uint64_t        effective_ram_bytes;
    uint32_t        numa_nodes;
    bool            hugepages_available;
    bool            memory_pinning_available;
    uint8_t         _pad[6];
} aios_hardware_profile_t;

/* ── Configuration ───────────────────────────────────────────────────────────── */
typedef struct {
    const char *manifest_path;          /* Path to profiler-generated manifest.json */
    uint64_t    kv_reserve_bytes;       /* KV cache RAM reservation (allocated first) */
    uint32_t    max_rws_percent;        /* Max % of effective RAM for weight residency */
    uint32_t    hot_floor_percent;      /* Minimum HOT tier before streaming fallback  */
    uint32_t    prefetch_distance;      /* Layers to prefetch ahead (default: 2)       */
    bool        disable_aliasing;       /* Disable weight aliasing entirely            */
    bool        disable_sparsity;       /* Disable sparsity map                        */
    bool        disable_prefetch;       /* Disable prefetch sequencing                 */
    uint8_t     sparsity_mode;          /* 0=conservative, 1=balanced, 2=aggressive    */
    uint8_t     _pad[3];
} aios_config_t;

/* ── Context (opaque handle) ─────────────────────────────────────────────────── */
typedef struct aios_context aios_context_t;

/* ── Return codes ────────────────────────────────────────────────────────────── */
typedef enum {
    AIOS_OK                  =  0,
    AIOS_ERR_INVALID_ARG     = -1,
    AIOS_ERR_MANIFEST_LOAD   = -2,
    AIOS_ERR_OUT_OF_MEMORY   = -3,
    AIOS_ERR_PINNING_FAILED  = -4,   /* Memory pinning unavailable; degraded     */
    AIOS_ERR_NO_HUGEPAGES    = -5,   /* Large pages unavailable; fallback applied */
    AIOS_ERR_ISA_UNAVAILABLE = -6,   /* Requested ISA tier unavailable; fallback */
    AIOS_ERR_NOT_IMPLEMENTED = -99,  /* Stub — not yet implemented               */
} aios_status_t;

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE API
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * aios_init — Initialize AIOS context.
 *
 * Loads profiler artifacts from manifest, detects hardware capabilities,
 * applies degradation ladder if necessary, starts prefetch thread.
 *
 * Must be called once before any inference. Thread-safe: each model instance
 * should have its own context (AIOS is single-tenant by design).
 *
 * @param ctx     Output: pointer to newly allocated context
 * @param config  Configuration. Pass NULL for defaults.
 * @return        AIOS_OK on success. On AIOS_ERR_PINNING_FAILED or
 *                AIOS_ERR_NO_HUGEPAGES, AIOS still runs in degraded mode.
 */
aios_status_t aios_init(aios_context_t **ctx, const aios_config_t *config);

/**
 * aios_destroy — Shutdown AIOS context and release all resources.
 *
 * Stops prefetch thread, releases memory pins, frees context.
 * Safe to call even if aios_init returned a degraded-mode error.
 */
void aios_destroy(aios_context_t *ctx);

/**
 * aios_get_hardware_profile — Return detected hardware profile.
 */
aios_status_t aios_get_hardware_profile(aios_context_t *ctx,
                                         aios_hardware_profile_t *profile);

/* ═══════════════════════════════════════════════════════════════════════════════
 * HOT PATH API  — called on every block access during inference
 * No dynamic allocation. No OS calls. No decisions.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * aios_resolve_block — Resolve a logical block ID to a physical pointer.
 *
 * Core hot-path function. Does exactly two things:
 *   1. Looks up the pointer table entry for logical_block_id
 *   2. If the block is aliased (AIOS_FLAG_ALIASED), applies the delta
 *      and returns a pointer to the corrected data
 *
 * The returned pointer is valid until the next call to aios_resolve_block
 * from the same thread (delta application uses a per-thread scratch buffer).
 *
 * @param ctx              AIOS context
 * @param logical_block_id Block ID from the inference engine's layer/tensor/tile address
 * @param out_data         Output: pointer to weight data (canonical or delta-corrected)
 * @param out_length       Output: length of data in bytes
 * @return                 AIOS_OK, or AIOS_ERR_INVALID_ARG if block_id out of range
 */
aios_status_t aios_resolve_block(aios_context_t *ctx,
                                  uint32_t        logical_block_id,
                                  const void    **out_data,
                                  uint32_t       *out_length);

/**
 * aios_is_active — Check whether a weight row should be computed.
 *
 * Consults the sparsity bitmask. Returns false if the row is marked inactive
 * for the current layer — the inference engine should skip the matrix
 * multiplication for this row.
 *
 * @param ctx      AIOS context
 * @param layer    Layer index
 * @param row      Row index within the FFN intermediate dimension
 * @return         true if row is active (compute), false if inactive (skip)
 */
bool aios_is_active(aios_context_t *ctx, uint32_t layer, uint32_t row);

/* ═══════════════════════════════════════════════════════════════════════════════
 * TELEMETRY API
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint64_t tokens_generated;
    uint64_t blocks_resolved;
    uint64_t blocks_skipped_sparse;
    uint64_t delta_applications;
    uint64_t prefetch_issued;
    double   hot_hit_ratio;
    uint32_t active_degradation_steps;  /* Bitmask of active degradation steps 1-5 */
    uint32_t isa_tier_active;
    bool     aliasing_enabled;
    bool     sparsity_enabled;
    bool     prefetch_enabled;
    bool     memory_pinning_active;
} aios_telemetry_t;

/**
 * aios_get_telemetry — Snapshot current telemetry counters.
 */
aios_status_t aios_get_telemetry(aios_context_t *ctx, aios_telemetry_t *out);

/**
 * aios_reset_telemetry — Reset all counters to zero.
 */
void aios_reset_telemetry(aios_context_t *ctx);

/* ═══════════════════════════════════════════════════════════════════════════════
 * UTILITY
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * aios_status_string — Human-readable status description.
 */
const char *aios_status_string(aios_status_t status);

/**
 * aios_version — Return version string "MAJOR.MINOR.PATCH".
 */
const char *aios_version(void);

#ifdef __cplusplus
}
#endif

#endif /* AIOS_H */
