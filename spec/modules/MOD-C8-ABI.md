<!-- ARIA module: MOD-C8-ABI.md — do not edit directly; edit this file in spec/modules/ -->
## 16. Binary Interface (ABI)

To ensure interoperability between runtimes (C++, Rust, Python) and models (Rust, C++, etc.),
all ARIA artifacts MUST conform to the normative C ABI defined in Section 16.1.

### 16.1 Normative C Header

The code block below is the authoritative definition of the binary interface.
Runtimes MUST implement the loading mechanism defined below.
Models MUST export the symbols defined below.

```c
#ifndef ARIA_H
#define ARIA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 1. Primitive Types & Handles
// ============================================================================

typedef struct aria_model_s* aria_model_t;
typedef struct aria_state_handle_s* aria_state_handle_t;

typedef enum {
    ARIA_SUCCESS = 0,
    ARIA_ERR_UNKNOWN = 1,
    ARIA_ERR_INVALID_ARGUMENT = 2,
    ARIA_ERR_OUT_OF_MEMORY = 3,
    ARIA_ERR_STATE_UNINITIALIZED = 4,
    ARIA_ERR_BATCH_SIZE_EXCEEDED = 5,
    ARIA_ERR_TOKEN_LIMIT_EXCEEDED = 6,
    ARIA_ERR_MODEL_ERROR = 7,
    ARIA_ERR_CANCELLED = 8          // in-flight call aborted by cancel(); handle is Released
} aria_error_t;

typedef enum {
    ARIA_FLOAT32         = 0,
    ARIA_BFLOAT16        = 1,
    ARIA_FLOAT16         = 2,
    ARIA_INT8            = 3,
    ARIA_FLOAT8_E4M3FN   = 4,   // OCP MX E4M3 (finite, no NaN; AMD, Intel)
    ARIA_FLOAT8_E4M3FNUZ = 5,   // NVIDIA E4M3 variant (unsigned zero; H100/H200)
                                 // NOTE: E4M3FN and E4M3FNUZ are NOT interchangeable
    ARIA_FLOAT8_E5M2     = 6,   // OCP MX / NV E5M2
    ARIA_FLOAT4_E2M1     = 7,   // OCP MX / NVFP4
    // Int4 removed in 0.5. Use ARIA_CUSTOM_FLOAT for parameterized formats.
    ARIA_CUSTOM_FLOAT    = 100, // e_bits, m_bits packed (see FloatCustom in §4.2)
    ARIA_CUSTOM_INT      = 101  // scale_bits, data_bits packed
} aria_dtype_t;

// ============================================================================
// 2. Data Structures
// ============================================================================

typedef struct {
    void*     data;
    uint32_t* shape;       // [dim0, dim1, ...]
    uint32_t  rank;
    aria_dtype_t dtype;
} aria_tensor_view_t;

typedef struct {
    uint32_t major;
    uint32_t minor;
} aria_version_t;

typedef struct {
    aria_version_t version;
    uint32_t       vocab_size;
    const char*    model_name;
    uint32_t       max_sequence_length;
    
    // Capabilities: Array of strings
    const char**   capabilities;
    uint32_t       num_capabilities;

    // Supported DTypes: Array of enums
    aria_dtype_t*  supported_dtypes;
    uint32_t       num_supported_dtypes;

    // Tokenizer definition (optional path or JSON string)
    const char*    tokenizer_source;
    uint32_t       tokenizer_source_len;
} aria_model_info_t;

// SemanticTag types — correspond to SemanticTag variants in §6
typedef enum {
    ARIA_TAG_ATTENTION_KV = 0,
    ARIA_TAG_SSM_STATE    = 1,
    ARIA_TAG_CUSTOM       = 2
} aria_semantic_tag_type_t;

typedef struct {
    aria_semantic_tag_type_t type;
    uint32_t    layer_idx;       // AttentionKV and SSMState
    uint8_t     is_key;          // AttentionKV: 1=key slot, 0=value slot
    uint32_t    window_size;     // AttentionKV: sliding window size; 0 = no window
    const char* extension_id;   // Custom: MUST match an entry in EXTENSIONS.md; NULL otherwise
} aria_semantic_tag_t;

// Dimension types — correspond to Dimension variants in §4.4
typedef enum {
    ARIA_DIM_SEQUENCE    = 0,   // resolved to max_tokens at allocation
    ARIA_DIM_FIXED       = 1,   // literal constant
    ARIA_DIM_MODEL_PARAM = 2    // named constant; runtime uses value field
} aria_dimension_type_t;

typedef struct {
    aria_dimension_type_t type;
    uint64_t    value;          // FIXED and MODEL_PARAM: the concrete value
    const char* name;           // MODEL_PARAM: human-readable name; NULL for FIXED/SEQUENCE
} aria_dimension_t;

// Layout types
typedef enum {
    ARIA_LAYOUT_CONTIGUOUS = 0,
    ARIA_LAYOUT_PAGED      = 1,
    ARIA_LAYOUT_CUSTOM     = 2
} aria_layout_type_t;

// StateSlot — corresponds to StateSlot in §4.11
typedef struct {
    const char*          id;                     // unique stable slot identifier
    aria_semantic_tag_t  semantic_tag;
    aria_dimension_t*    shape;                  // array of dimensions
    uint32_t             rank;                   // length of shape array
    aria_dtype_t         dtype;                  // declared preferred dtype
    aria_layout_type_t   layout;
    uint32_t             preferred_page_tokens;  // Paged only; 0 = no preference
    const char*          layout_custom_id;       // Custom layout only; NULL otherwise
} aria_state_slot_t;

// StateSpec — corresponds to StateSpec in §4.10
// Returned by declare_state(); owned by the model; valid for the model's lifetime.
typedef struct {
    aria_state_slot_t* slots;
    uint32_t           num_slots;
} aria_state_spec_t;

// ── Paged Layout: Block Table (Strategy 2) ───────────────────────────────────
// When a slot uses Layout::Paged and the runtime implements Strategy 2 (Block Table
// Access, §4.5.2), it exposes this structure to the model via the TensorView mechanism.
// The model must use paging-aware kernels (e.g., PagedAttention) to access slot data.
//
// block_table[seq_idx][block_idx] = physical_block_id (uint32)
// Physical block addresses: block_base_ptrs[physical_block_id] → void*
//
typedef struct {
    uint32_t  block_size_tokens;     // page size in tokens (runtime-chosen)
    uint32_t  num_blocks;            // total physical blocks in the pool
    uint32_t  max_blocks_per_seq;    // columns in block_table
    uint32_t* block_table;           // [num_sequences × max_blocks_per_seq] flat array
    void**    block_base_ptrs;       // [num_blocks] array of physical block pointers
} aria_block_table_t;

// ── Batch Result Type ────────────────────────────────────────────────────────
// Implements the two-level batch error semantics defined in §9 for the C ABI.
// Batch inference functions (prefill_batch, decode_batch) return an array of
// aria_batch_item_result_t — one per request in the batch, in input order.
//
// The function's return value (aria_error_t) signals CATASTROPHIC failure only:
//   ARIA_SUCCESS            → batch executed; inspect per-item results
//   ARIA_ERR_MODEL_ERROR    → fatal, unrecoverable; all requests failed
//
// Per-item result:
//   status == ARIA_SUCCESS → this request succeeded
//   status != ARIA_SUCCESS → this request failed; other requests may have succeeded
//
typedef struct {
    aria_error_t status;             // per-item outcome
    uint32_t     error_code;         // model error code if status == ARIA_ERR_MODEL_ERROR
    char         detail[128];        // sanitized model error detail (may be empty)
} aria_batch_item_result_t;

// ============================================================================
// 3. Interface VTable
// ============================================================================

typedef struct {
    // Lifecycle
    aria_error_t (*get_info)(aria_model_t model, aria_model_info_t* out_info);
    
    // State Management
    // out_spec: set to a pointer to a model-owned aria_state_spec_t; valid for model lifetime.
    aria_error_t (*declare_state)(aria_model_t model, uint32_t max_batch, uint32_t max_tokens,
                                  aria_state_spec_t** out_spec);
    
    // Inference
    // out_logits: can be NULL to skip output
    aria_error_t (*prefill)(aria_model_t model, 
                            const uint32_t* tokens, uint32_t num_tokens, 
                            aria_state_handle_t state, 
                            aria_tensor_view_t* out_logits);

    aria_error_t (*decode)(aria_model_t model, 
                           uint32_t token, 
                           aria_state_handle_t state, 
                           aria_tensor_view_t* out_logits);

    // Batch Inference (Production Path) — implements §9 two-level batch error semantics.
    //
    // Return value: ARIA_SUCCESS = batch executed (inspect out_results per-item);
    //               ARIA_ERR_MODEL_ERROR (fatal) = catastrophic failure, all items invalid.
    //
    // tokens: flattened array [sum of all num_tokens_per_seq[i]]
    // num_tokens_per_seq: array of length num_sequences; each entry is tokens for that seq
    // states: array of handles, length num_sequences
    // out_logits: array of tensor_views, length num_sequences (NULL entry = skip that seq)
    // out_results: caller-allocated array of num_sequences aria_batch_item_result_t;
    //              MUST be non-NULL; written by model in input order
    aria_error_t (*prefill_batch)(aria_model_t model,
                                  uint32_t num_sequences,
                                  const uint32_t* tokens,
                                  const uint32_t* num_tokens_per_seq,
                                  aria_state_handle_t* states,
                                  aria_tensor_view_t* out_logits,
                                  aria_batch_item_result_t* out_results);

    // tokens: array of token IDs, length num_sequences (one token per sequence)
    // states: array of handles, length num_sequences
    // out_logits: array of tensor_views, length num_sequences (NULL entry = skip that seq)
    // out_results: caller-allocated array of num_sequences aria_batch_item_result_t;
    //              MUST be non-NULL; written by model in input order
    aria_error_t (*decode_batch)(aria_model_t model,
                                 uint32_t num_sequences,
                                 const uint32_t* tokens,
                                 aria_state_handle_t* states,
                                 aria_tensor_view_t* out_logits,
                                 aria_batch_item_result_t* out_results);

    // ── Cancellation ─────────────────────────────────────────────────────────────
    // Request cancellation of an in-flight prefill or decode on this handle.
    // Thread-safe: MAY be called concurrently with an in-flight inference call.
    // Returns ARIA_SUCCESS immediately; the in-flight call returns ARIA_ERR_CANCELLED.
    // After this call returns, the runtime MUST treat the handle as Released.
    // Calling cancel on an already-Released handle MUST return ARIA_SUCCESS (no-op).
    aria_error_t (*cancel)(aria_model_t model, aria_state_handle_t state);

    // ── Tokenizer (corresponds to ITokenizer in §5.5) ─────────────────────────────
    // All tokenizer function pointers MAY be NULL; runtimes MUST check before calling.
    // Non-NULL implementations MUST conform to §5.5 semantics.

    // encode: encode a single text string to token IDs.
    // Returns number of tokens written to out_tokens.
    // If out_tokens is NULL, returns required capacity.
    int32_t (*encode)(aria_model_t model, const char* text, uint32_t* out_tokens, uint32_t capacity);

    // encode_batch: encode multiple null-terminated strings.
    // texts: array of num_texts const char* pointers
    // out_token_counts[i]: number of tokens for texts[i] (written by callee)
    // out_tokens_flat: caller-allocated flat buffer; callee writes all token sequences
    //                  back-to-back; use out_token_counts to slice
    // capacity: total capacity of out_tokens_flat
    // Returns total tokens written, or required capacity if out_tokens_flat is NULL.
    int32_t (*encode_batch)(aria_model_t model,
                            const char** texts, uint32_t num_texts,
                            uint32_t* out_token_counts,
                            uint32_t* out_tokens_flat, uint32_t capacity);

    // decode_token: decode a single token ID to UTF-8 text (may be a partial UTF-8 sequence).
    // Returns bytes written to out_text (including null terminator).
    // If out_text is NULL, returns required capacity.
    int32_t (*decode_token)(aria_model_t model, uint32_t token, char* out_text, uint32_t capacity);

    // decode_tokens: decode a full token ID sequence to a complete UTF-8 string.
    // tokens: array of num_tokens token IDs
    // Returns bytes written to out_text (including null terminator).
    // If out_text is NULL, returns required capacity.
    int32_t (*decode_tokens)(aria_model_t model,
                             const uint32_t* tokens, uint32_t num_tokens,
                             char* out_text, uint32_t capacity);

    // special_tokens: enumerate special tokens (EOS, BOS, PAD, UNK, etc.).
    // out_names: caller-allocated array of const char* pointers; callee fills with
    //            stable string pointers (valid for model lifetime)
    // out_ids: caller-allocated array of uint32_t; callee fills with token IDs
    // capacity: capacity of out_names and out_ids (must be equal)
    // Returns number of special tokens written, or required capacity if out_names is NULL.
    int32_t (*special_tokens)(aria_model_t model,
                              const char** out_names, uint32_t* out_ids, uint32_t capacity);

} aria_interface_t;

// ============================================================================
// 4. Entry Point
// ============================================================================

// Every ARIA shared library MUST export these two functions.

// Load model: returns the interface vtable. Returns NULL on failure.
// The caller owns the returned pointer until aria_unload_model() is called.
aria_interface_t* aria_load_model(const char* path);

// Unload model: releases all resources associated with the model vtable.
// The caller MUST NOT use the interface pointer after this call.
// Passing NULL is a no-op.
void aria_unload_model(aria_interface_t* iface);

#ifdef __cplusplus
}
#endif

#endif // ARIA_H
```

The ABI includes:
1.  **Single and Batch Inference**: `prefill`, `decode`, `prefill_batch`, `decode_batch`.
    Batch functions implement the §9 two-level error semantics via `aria_batch_item_result_t`:
    the function return value signals catastrophic failure only; per-item outcomes are written
    to the caller-provided `out_results` array.
2.  **Data Model**: `aria_state_spec_t`, `aria_state_slot_t`, `aria_semantic_tag_t`,
    `aria_dimension_t` — the typed C representation of the §4 type system. `declare_state()`
    writes a pointer to a model-owned `aria_state_spec_t`; the runtime reads slot definitions
    from it. The model owns this memory for its lifetime.
3.  **Paged Layout / Block Table**: `aria_block_table_t` defines the structure for §4.5.2
    Strategy 2 (Block Table Access). Runtimes implementing Strategy 2 expose this structure
    to the model; the model reads `block_table` and `block_base_ptrs` to locate physical pages.
4.  **Model Lifecycle**: `aria_load_model()` returns the vtable; `aria_unload_model()` releases
    all model resources. The caller MUST call `aria_unload_model()` when done. Failing to do
    so is a resource leak.
5.  **Tokenizer Interface**: Complete `ITokenizer` implementation via function pointers:
    `encode`, `encode_batch`, `decode_token`, `decode_tokens`, `special_tokens`. All are
    optional (MAY be NULL); runtimes MUST check for NULL before calling. This vtable
    completes the §5.5 ITokenizer contract at the C ABI level.
6.  **State Handle Lifecycle**: `aria_state_handle_t` is an opaque pointer owned by the RUNTIME.
    -   The runtime allocates, manages, and destroys all handle instances.
    -   The model receives handles as read/write parameters.
    -   **CRITICAL**: The model MUST NOT attempt to free, delete, or call a destructor on a handle.
    -   **CRITICAL**: The model MUST NOT store handle pointers beyond the scope of a call (unless
        specifically designed as a stateful C++ object, which is discouraged).

### 16.2 Calling Convention
All exported functions use the platform-native C calling convention:
- **x86/x64**: `cdecl` (caller cleans stack, left-to-right argument order)
- **AArch64 / ARM64**: AAPCS64 (ARM Architecture Procedure Call Standard for 64-bit)
  — applies to Apple Silicon (M-series), Qualcomm Snapdragon, Huawei Ascend 910B/C,
  AWS Graviton, and all other ARM64 deployments
- **Other architectures**: the platform's standard C calling convention applies

Symbols are exported with visibility `default`. On macOS/iOS, `-exported_symbols_list`
or equivalent MUST include `aria_load_model` and `aria_unload_model`.

**Note on compiled-graph models**: Models implemented as TensorRT engines, CoreML packages,
QNN graphs, or SynapseAI binaries that do not produce a loadable shared library MUST
provide a thin C wrapper shim that exports these symbols and delegates to the compiled
artifact. The shim is considered part of the ARIA model artifact.

---
