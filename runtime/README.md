# AIOS Runtime

The runtime is the reference implementation of the AIOS Model Contract. It is a
**memory residency controller** — not an inference engine. It sits between llama.cpp,
Ollama, or vLLM and the CPU hardware, managing how weight data moves from DRAM.

---

## Design Principles

**Complexity belongs in the profiler, not the runtime.**

At inference time, the runtime does exactly four things:

| Responsibility | Function | Hot path? |
|---|---|---|
| Resolve a pointer | `aios_resolve_block()` | Yes — every block access |
| Apply a delta | Called internally by resolve if block is aliased | Yes |
| Skip a sparse row | `aios_is_active()` | Yes — every FFN row |
| Issue a static prefetch | Managed on prefetch thread | Background |

**Hard constraints — never violate:**
- No dynamic allocation in the hot path
- No runtime model decisions
- No adaptive scheduling
- No per-token OS queries

---

## Implementation Guide

### What needs to be built

```
runtime/
├── aios.h              ← Public ABI (defined — do not change without RFC)
├── aios.c              ← Core context, init/destroy, lifecycle  [NOT YET BUILT]
├── pointer_table.c     ← Pointer table load and block resolution [NOT YET BUILT]
├── delta.c             ← Delta application with SIMD             [NOT YET BUILT]
├── sparsity_map.c      ← Bitmask load and is_active lookup       [NOT YET BUILT]
├── residency.c         ← Memory pinning, KV budgeting            [NOT YET BUILT]
├── prefetch.c          ← Prefetch thread, static sequence walk   [NOT YET BUILT]
├── telemetry.c         ← Counter management                      [NOT YET BUILT]
└── platform/
    ├── isa_detect.c    ← CPUID/HWCAP capability detection        [NOT YET BUILT]
    ├── tier1/          ← SSE4.2 / NEON kernels                   [NOT YET BUILT]
    ├── tier2/          ← AVX2 kernels                            [NOT YET BUILT]
    └── tier3/          ← AVX-512 / AMX kernels                   [NOT YET BUILT]
```

### Recommended build order

1. **`pointer_table.c`** — implement `aios_resolve_block()` reading from profiler artifacts.
   This is the single most important component. Even without aliasing, it enables
   the residency and prefetch systems to function.

2. **`sparsity_map.c`** — implement `aios_is_active()` as a bitmask lookup.
   Straightforward: load the bitmask from `sparsity_map.bin`, return bit at (layer, row).

3. **`residency.c`** — implement memory pinning and KV-first budgeting.
   Platform-appropriate: `mlock()` on Linux, `VirtualLock()` on Windows,
   `vm_wire()` on macOS/Darwin.

4. **`prefetch.c`** — implement the static prefetch thread.
   Walk the prefetch sequence from `prefetch_seq.bin` on a dedicated thread.
   Pin to performance cores on hybrid architectures.

5. **`delta.c`** — implement delta application with SIMD.
   Start with scalar implementation, then add AVX2 and AVX-512 variants.

6. **`platform/isa_detect.c`** — CPUID on x86, HWCAP on ARM, capability verification.

### Build system

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The CMakeLists.txt will be added when the first component is implemented.

### Testing

Every component should pass a correctness test against the reference Python
profiler output before SIMD optimization. Test vectors are in `runtime/tests/`.

---

## Integration

### ABI (recommended)
```c
#include "aios.h"

aios_context_t *ctx;
aios_init(&ctx, NULL);  // loads manifest, starts prefetch thread

// In inference loop:
const void *data; uint32_t len;
aios_resolve_block(ctx, block_id, &data, &len);
if (aios_is_active(ctx, layer, row)) {
    // compute this row
}
```

### LD_PRELOAD (zero modification)
```bash
LD_PRELOAD=/path/to/libaios_preload.so ollama run mistral
```

The LD_PRELOAD library intercepts `malloc()` and `mmap()` calls from llama.cpp
and redirects weight block allocation through the AIOS residency system.
This approach requires no modification to the inference engine binary.

---

## What to contribute

Start with `pointer_table.c` — it is the most impactful and the most straightforward.
The interface is fully defined in `aios.h`. The input format is in `profiler/manifest.py`.
Open a GitHub Discussion before starting to coordinate with other contributors.
