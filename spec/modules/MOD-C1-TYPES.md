<!-- ARIA module: MOD-C1-TYPES.md — do not edit directly; edit this file in spec/modules/ -->
## 4. Type System

### 4.1 Primitive Types

```
type TokenId  = uint32          // single vocabulary index
type TokenIds = List<TokenId>   // ordered token sequence; MUST be non-empty in prefill()

// Result<T, E> — discriminated union returned by all interface methods.
// On success: Ok(value: T)
// On failure: Err(error: E)
// Implementations MUST use the host language's idiomatic equivalent
// (e.g., Result in Rust, Either in Haskell, discriminated union in C).
// The caller MUST inspect the variant before accessing any value.
type Result<T, E> = Ok(T) | Err(E)
```

### 4.2 DType

```
enum DType {
    Float32,       // IEEE 754 single precision
    BFloat16,      // brain float16 (1-8-7)
    Float16,       // IEEE 754 half precision
    Float8E4M3FN,   // 8-bit float, E4M3, OCP MX standard (finite, no NaN; used in AMD, Intel)
    Float8E4M3FNUZ, // 8-bit float, E4M3, NVIDIA variant (finite, no NaN, unsigned zero; H100/H200)
                    // NOTE: Float8E4M3FN and Float8E4M3FNUZ have different NaN/zero encodings
                    // and are NOT interchangeable. Implementations MUST distinguish them.
    Float8E5M2,    // 8-bit float, 5-bit exponent, 2-bit mantissa (OCP MX / NV E5M2)
    Float4E2M1,    // 4-bit float, 2-bit exponent, 1-bit mantissa (OCP MX / NVFP4)
    Int8,          // signed 8-bit integer
    // Int4 removed in 0.5-draft due to lack of standard packing

    // Parameterized float: 1 sign bit, e_bits exponent, m_bits mantissa.
    // Use for hardware-specific float formats not enumerated above.
    // Follows OCP Microscaling Formats (MX) naming conventions:
    //   FloatCustom{4, 3} = FP8 E4M3 equivalent
    //   FloatCustom{2, 1} = FP4 E2M1 equivalent
    //   FloatCustom{3, 2} = hypothetical FP6 E3M2
    // Reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec
    FloatCustom { e_bits: uint8, m_bits: uint8 },

    Custom(string), // vendor-defined dtype; string is a stable identifier
}
```

**Note on GGUF quantizations**: GGUF block quantization formats (Q4_K_M, Q5_K_S, IQ2_XXS, etc.)
apply to **model weights**, not state slot allocations. ARIA's `DType` applies only to
**KV cache and SSM state**. Model weight formats are the model implementation's internal
concern and are not negotiated through ARIA. A llama.cpp GGUF model uses `Float16` or `Int8`
for KV cache dtype negotiation while using any GGUF format internally for its weights.

### 4.3 Layout

```
enum Layout {
    // Dense row-major tensor; no paging or special addressing.
    Contiguous,

    // Paged tensor for KV cache; the runtime controls page granularity.
    // page_tokens: runtime-chosen page size in tokens (model declares 0 = "no preference").
    // The runtime MUST align its page size to the model's compute requirements.
    Paged { preferred_page_tokens: uint32 },

    // Vendor-defined layout; string is a stable identifier.
    // Runtimes that do not recognize the identifier MUST reject with SlotUnsupported.
    Custom(string),
}
```

**Note on page size**: The model MAY declare a `preferred_page_tokens` hint (e.g., 16, 32). The
runtime is not required to honor it but MUST use a page size that does not break the model's
compute. If incompatible, the runtime MUST return `SlotUnsupported`.

### 4.4 Dimension

Dimensions are declared symbolically so the runtime resolves concrete sizes at allocation time.

```
enum Dimension {
    Sequence,                           // resolved to max_tokens at allocation
    Fixed(uint64),                      // literal constant
    ModelParam { name: string, value: uint64 }, // named constant with its concrete value
                                        // (name for human readability; runtime uses value)
}
```

**Note on Batch**: `max_batch` is passed as a parameter to `declare_state()` for the model
to size its internal batch workspace (weight buffers, activation scratchpad). It is NOT a
dimension in per-handle `StateSlot` shapes. Each `StateHandle` represents one request;
its state slots are shaped per-request only.

**Example**: `[Sequence, ModelParam("num_kv_heads", 8), ModelParam("head_dim", 128)]`
resolves to shape `[max_tokens, 8, 128]` per handle.

### 4.5 TensorView

A `TensorView` is a language-binding-defined reference to a runtime-allocated tensor.
It provides typed read/write access to a `StateSlot`'s allocated memory. The binding
MUST document:

- The concrete type returned (e.g., `torch.Tensor` in Python, `*mut f16` in Rust)
- Whether the view is mutable
- The lifetime constraint: views are valid only within the scope of a single `prefill()`
  or `decode()` call. The model MUST NOT store a `TensorView` beyond the call that produced it.

#### 4.5.1 Memory Coherence

Writes made by the model to `TensorView` slots within a `prefill()` or `decode()` call
MUST be fully visible to the runtime after the call returns. The mechanism of coherence is
runtime-defined and depends on the target hardware:

| Hardware | Coherence mechanism |
|----------|---------------------|
| CUDA (NVIDIA) | CUDA stream synchronization after call completes |
| ROCm (AMD) | `hipStreamSynchronize` after call completes; note: fence semantics differ from NVIDIA CUDA in unified-memory edge cases |
| Metal (Apple Silicon) | Metal command buffer completion event |
| CPU / host memory | Memory barrier (e.g., `std::atomic_thread_fence`) |
| Qualcomm QNN (Hexagon NPU) | QNN completion callback / device queue flush |
| Intel Gaudi / SynapseAI | SynapseAI stream completion event |
| TPU / XLA (Google) | XLA buffer donation completion; `jax.block_until_ready()` or equivalent completion fence |
| Huawei Ascend / CANN | AscendCL event primitives; `aclrtSynchronizeDevice()` or stream event |

The model MUST write all state through `TensorView` accessors obtained from the handle
(not through independently allocated memory or model-owned buffers) to ensure the runtime's
coherence guarantee applies.

#### 4.5.2 TensorView for Paged Layouts

When a `StateSlot` declares `Layout::Paged`, the `TensorView` returned by
`handle.slot(id)` represents a **logically contiguous** view of the slot's data.
The runtime is responsible for mapping between logical positions and physical
page locations.

Two conformant strategies exist:

1. **Virtual Contiguity**: The runtime presents a contiguous memory region to the model
   (via virtual memory mapping). The model code sees a standard flat tensor.
   This simplifies model implementation but may incur OS overhead.

2. **Block Table Access**: The runtime exposes the block table structure (page pointers,
   block size) directly. The model must use paging-aware kernels (e.g., PagedAttention).

**Conformance**:
*   Models requesting `Layout::Paged` via `aria.core.layout.paged` capability MUST support
    Strategy 2 (Block Table Access) if the runtime provides it.
*   Runtimes supporting `Layout::Paged` MAY implement Strategy 1, Strategy 2, or both.
*   The C ABI defines the specific `aria_tensor_view_t` interpretation for each strategy.

The `preferred_page_tokens` field in `Layout::Paged` is a hint for block sizing.

#### 4.5.3 AttentionKV Slot Shape

For `AttentionKV` slots, `handle.slot()` returns a `TensorView` over the **full allocation**:

```
shape: [max_tokens, num_heads, head_dim]
```

The runtime allocates this slab once at handle creation time and returns the same view
on every call. The model:
- Writes new K and V entries at index `handle.current_length()`
- Reads prior entries at indices `0 .. handle.current_length() - 1`
- MUST NOT write to indices `< handle.current_length()`

This design avoids per-call re-slicing by the runtime. The model tracks its own write
position via `current_length()`. Headroom is always available as
`handle.tokens_remaining() = handle.max_tokens() - handle.current_length()`.



### 4.6 Logits Output Contract

The model writes logits to a caller-provided buffer (`out_logits`).
This buffer MUST be a `TensorView` with `dtype=Float32`.
The shape depends on the request:
*   **Decode**: `[vocab_size]` (one token)
*   **Prefill**: `[vocab_size]` (last token only) OR `[seq_len, vocab_size]` (all tokens)

The model MUST NOT write probability distributions (softmax). It writes raw pre-softmax scores.
If `out_logits` is `None` (or null), the model skips logit computation (optimization).

**Note for model implementors**: When `out_logits` is shaped for a single position (last token only),
the model is permitted — but not required — to skip materializing logits for all prior positions.
This is an intentional optimization opportunity: high-throughput runtimes will pass a single-position
buffer in the common case, allowing models to avoid the cost of computing full-sequence logits.

### 4.7 Batch Request Types

```
struct PrefillRequest {
    tokens:     TokenIds           // MUST be non-empty
    state:      StateHandle
    out_logits: Option<TensorView> // Destination for logits (or None to skip)
}

struct DecodeRequest {
    token:      uint32             // Single token
    state:      StateHandle
    out_logits: Option<TensorView> // Destination for logits (or None to skip)
}
```

Used in the batch overloads of `prefill()` and `decode()` (§5.1). Each entry in the
batch is an independent request with its own handle. Handles in a batch MUST be distinct.

### 4.8 (Reserved)

(Section 4.8 reserved for future use)

### 4.9 ModelInfo

```
struct ModelInfo {
    aria_version:        (uint32, uint32)  // (major, minor) this model targets
    vocab_size:          uint32            // size of the token vocabulary
    model_name:          string            // human-readable identifier
    max_sequence_length: uint32            // hard upper bound on sequence length this
                                           // model can process; inform declare_state callers
    supported_kv_dtypes: List<DType>       // KV dtype preferences, ordered best-to-acceptable
                                           // runtime picks from this list (see §8)
    capabilities:        List<String>      // OPTIONAL feature flags (see spec/EXTENSIONS.md)
                                           // e.g. "aria.core.context_shift"
    default_tokenizer:   ITokenizer        // model's reference tokenizer implementation;
                                           // runtime MAY substitute (see §5.5)
}
```

### 4.10 StateSpec

```
struct StateSpec {
    slots:    List<StateSlot>
    // Total slots required per request.
    // Slot IDs MUST be unique within a StateSpec.
    // The runtime allocates one copy of this layout per active StateHandle.
}
```

### 4.11 StateSlot

```
struct StateSlot {
    id:           string       // unique stable identifier; used as key in handle accessors
    semantic_tag: SemanticTag  // role of this state (see §6)
    shape:        List<Dimension>
    dtype:        DType        // model's declared preferred dtype; see §8 for negotiation
    layout:       Layout
}
```

---
