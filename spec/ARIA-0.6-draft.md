# ARIA — Agnostic Runtime Inference Abstraction
## Specification 0.6-draft

**Status**: Draft  
**Authors**: HaiberDyn  
**Repository**: https://github.com/HaiberDyn/ARIA  
**Version**: 0.6-draft (2026-03-06)  
**License**: Spec text: CC BY-SA 4.0 · Reference code: Apache 2.0  
**IP Policy**: See `CONTRIBUTING.md` — patent non-assertion pledge, DCO contribution terms  
**Governance**: See `CONTRIBUTING.md` — working group structure, how to join  
**Conformance**: See `CONFORMANCE.md` — model and runtime requirements, portability levels

**Keywords**: The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are
to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

---

## Abstract

ARIA defines a minimal, architecture-agnostic contract between a language model implementation
and a serving runtime. It specifies two interfaces: `ILanguageModel` — the stateful inference
contract (`model_info`, `declare_state`, `prefill`, `decode`) — and `ITokenizer` — the
text↔token seam the model ships with and the runtime may optimize. The central design
principle: a model *declares* its state requirements — shape, dtype, semantic role — and the
runtime *allocates* those resources opaquely. Neither side exposes internals to the other.

This separation enables model architectures, serving runtimes, and hardware backends to evolve
independently. Adding a new architecture (attention → SSM → hybrid → future) requires no
framework modification. Switching runtimes requires no model modification. Operators compose
freely.

---

## 1. Motivation

### 1.1 The Coupling Problem

Every new LLM architecture generation forces code changes in every serving framework:

```
GPT → LLaMA       requires: grouped-query attention support in KV cache allocator
LLaMA → Mamba     requires: new SSM state manager, constant-length state contract
Mamba → Hybrid    requires: per-layer heterogeneous state (KV + SSM simultaneously)
Dense → MoE       requires: new expert routing in model runner, skip logic in batcher
Qwen3 → Qwen3.5   requires: updated attention patterns and memory profiling again
```

Root cause: frameworks (SGLang, vLLM, llama.cpp, TGI, LMDeploy) are coupled to model internals.
They know about attention head counts, KV cache shapes, expert routing tables — implementation
details they have no architectural reason to know.

This violates four SOLID principles simultaneously:

| Principle | Violation |
|-----------|-----------|
| **Open/Closed** | Every new architecture requires modifying framework internals |
| **Liskov Substitution** | Models are not substitutable — each requires bespoke runtime paths |
| **Interface Segregation** | Runtime consumes model internals it never needs to reason about |
| **Dependency Inversion** | Framework depends on model concretions, not stable abstractions |

It also violates DRY: KV cache management, batching logic, and state scheduling are reimplemented
per architecture family in every framework.

### 1.2 What ARIA Adds to the Mix

Each existing specification solves a real problem well. None address the **stateful
autoregressive decode loop with heterogeneous state types** — not because they got it
wrong, but because it wasn't their problem to solve. ARIA is additive.

| Spec | What it does well | What ARIA adds |
|------|------------------|----------------|
| ONNX | Static graph portability, cross-framework operator coverage | Stateful per-request decode loop; dynamic control flow (MoE routing, recurrent state) |
| WASI-NN | Hardware-agnostic inference backend; WebAssembly portability | Per-request state lifecycle; heterogeneous state type declaration and negotiation |
| GGUF | Single-file model distribution with weights, metadata, tokenizer | The serving-time contract: how a runtime calls into the model at inference time |
| Triton Inference Server backend API | Strong serving contract, production-proven | Standardized stateful decode; model-declared state layout instead of per-backend bespoke KV management |
| HuggingFace `generate()` | Broad model support, sampling library, Python ecosystem | Language-agnostic contract; hides model internals; usable outside Python |

The universal gap none of them fill: no existing spec defines a standard where the model
**declares** its state requirements and the runtime **allocates** them opaquely. For
pure-attention models this is a convenience. For hybrid attention+SSM models it is the
only portable approach.

### 1.3 ARIA's Solution

```
interface ILanguageModel {
    ModelInfo          model_info()
    StateSpec          declare_state(max_batch: uint32, max_tokens: uint32)
    List[Logits]       prefill(tokens: TokenIds, state: StateHandle, mode: LogitsMode = LastPosition)
    List[List[Logits]] prefill(batch: List<PrefillRequest>)
    Logits             decode(token: TokenId, state: StateHandle)
    List[Logits]       decode(batch: List<DecodeRequest>)
}
```

The model tells the runtime what state it needs. The runtime allocates it, owns the handle
lifecycle, and passes handles to the model at call time. The model reads and writes through
the handle. Neither party looks inside the other.

---

## 2. Scope

### 2.1 In Scope — ARIA 0.5-draft

- Decoder-only autoregressive language models
- Discrete token vocabulary (TokenId = uint32, as defined in §4.1)
- Synchronous, per-request generation
- Heterogeneous state types: attention KV cache, recurrent SSM state, and model-defined Custom state
- Basic capability negotiation: ARIA version, vocab size, supported KV dtypes
- Single-node inference (no cross-node communication protocol)

### 2.2 Explicitly Out of Scope — ARIA 0.5-draft

The following are NOT ARIA's concern. Each layer owns its domain:

| Concern | Owner |
|---------|-------|
| Model file format | SafeTensors, GGUF, ONNX |
| Weight quantization format | Model card, nvidia-modelopt, GGUF quant |
| Tokenization | HuggingFace tokenizers, tiktoken, sentencepiece |
| Sampling strategy (temperature, top-p, top-k, beam) | Runtime — above ARIA |
| Consumer API (chat completions, streaming) | OpenAI-compatible layer — above ARIA |
| Hardware dispatch (CUDA kernels, Metal, CPU SIMD) | Model implementation — below ARIA |
| GPU batching and scheduling | Runtime — below ARIA |
| Tensor parallelism / pipeline parallelism | Model implementation — internal |
| LoRA / adapter weight management | Out of scope 0.6 (see §15) |
| Encoder-decoder models | Out of scope 0.6 (see §15) |
| Multimodal embedding inputs | Out of scope 0.6 (see §15) |
| Async / cancellation protocol | Runtime implementation detail |
| CUDA graph capture | Model implementation detail |

**Design principle**: ARIA does not prevent innovation in any of these areas. It creates a stable
boundary so that innovation in one layer does not require changes in another. See §11.

**Scope extension schedule**: Future versions will expand this list systematically.

| Extension | Target | Notes |
|-----------|--------|-------|
| Encoder-decoder models, `CrossAttentionKV` | ARIA 1.1 | H2 2026 |
| LoRA / adapter interface | ARIA 1.1 | H2 2026 |
| Multimodal embedding inputs (vision, audio) | ARIA 2.0 | 2027 |
| Distributed multi-node inference contract | ARIA 2.0 | 2027 |

These timelines require multi-stakeholder review and at least two independent
implementations before promotion. See `ROADMAP.md` for details.

---

## 3. Definitions

| Term | Definition |
|------|-----------|
| **Model** | An implementation of `ILanguageModel` — weights plus compute logic |
| **Runtime** | A serving system (vLLM, SGLang, llama.cpp, etc.) that calls `ILanguageModel` to generate tokens |
| **Harness** | Synonym for **Runtime**. The component responsible for orchestrating model execution and state management. |
| **State** | All mutable per-request data required to produce correct next-token logits |
| **StateHandle** | An opaque reference to runtime-allocated state storage for one request |
| **Slot** | One contiguous allocation described by a `StateSlot`; accessed via StateHandle |
| **Consumer** | A client of the runtime's serving API (e.g., an OpenAI-compatible endpoint caller) |
| **Binding** | A language-specific expression of the ARIA interface (Python, C, Rust, etc.) |

---

## 3.1 Implementation Principles

ARIA is designed around compiled, type-safe runtimes. The following principles are design
expectations; they are not mandates on implementation language choice, but implementations
that do not uphold them forfeit the safety guarantees the design depends on.

**Encapsulation.** `StateHandle` is opaque. The consumer has no access to slot memory, handle
internals, or model state outside of the defined interface methods. `ModelInfo` MUST be treated
as immutable by the consumer; the model MUST return identical values on repeated calls.

**Immutability at boundaries.** `TensorView` is consumer-immutable: the consumer MUST NOT write
into model-owned tensor memory. The model MUST NOT retain a `TensorView` reference beyond the
scope of the call that produced it.

**Type safety.** The interface contract is expressed in terms of typed structures and enums.
Implementations SHOULD enforce this contract at compile time. Runtime type errors are a
conformance failure, not a recoverable condition.

**Separation of concerns.** The model owns inference. The runtime owns memory management,
scheduling, and hardware dispatch. Neither side should reach across this boundary.

Implementations in interpreted or dynamically-typed languages can conform to the letter of
this specification, but compile-time enforcement of the contract is unavailable. The design
rewards implementations that take full advantage of type-safe, compiled runtimes.

---

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
    Float8E4M3,    // 8-bit float, 4-bit exponent, 3-bit mantissa (OCP MX / NV E4M3)
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
| CUDA (NVIDIA, AMD ROCm) | CUDA stream synchronization after call completes |
| Metal (Apple Silicon) | Metal command buffer completion event |
| CPU / host memory | Memory barrier (e.g., `std::atomic_thread_fence`) |
| NPU (Qualcomm, Intel Gaudi) | Device command queue flush / synapse barrier |

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

**Note for model implementors**: With `LastPosition`, the model is permitted (but not
required) to skip materializing logits for positions 0..len(tokens)-2. This is the
optimization opportunity — high-throughput runtimes will call `prefill` with
`LastPosition` in the common case.

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

## 5. ILanguageModel — The Contract

### 5.1 Interface Definition

```
interface ILanguageModel {

    // Called once at model load time.
    // Returns model metadata needed by the runtime before state allocation.
    model_info() -> Result<ModelInfo, ARIAError>

    // Called once after model_info(), before any prefill or decode.
    // Returns the state layout the model requires per active request.
    // max_batch: maximum number of concurrent StateHandles.
    // max_tokens: maximum total tokens per handle (prefill + decode combined).
    // MUST be deterministic: identical inputs MUST produce identical StateSpec.
    declare_state(max_batch: uint32, max_tokens: uint32) -> Result<StateSpec, ARIAError>

    // Single-handle: process tokens for one request and populate state.
    // out_logits: destination for output logits. If None, skip computation.
    // May be called on Uninitialized handles (first call) or Ready handles (extend).
    // See §7.4 for multi-call prefill semantics. tokens MUST be non-empty.
    prefill(tokens: TokenIds, state: StateHandle, out_logits: Option<TensorView> = None) -> Result<void, ARIAError>

    // Batch overload: process prefill for multiple requests in a single GPU operation.
    // The runtime assembles the batch; the model executes it as a unit.
    // Each PrefillRequest carries its own tokens, handle, and out_logits buffer.
    // Returns one Result per request, in the same order as the input batch.
    prefill(batch: List<PrefillRequest>) -> Result<List<Result<void, ARIAError>>, ARIAError>

    // Single-handle: process one decode token, writes logits for next position to out_logits.
    // State MUST be in Ready state (prefill must have been called).
    decode(token: TokenId, state: StateHandle, out_logits: Option<TensorView> = None) -> Result<void, ARIAError>

    // Batch overload: process one decode step for multiple requests simultaneously.
    // The runtime assembles the batch; the model executes all tokens in one GPU operation.
    // Returns one Result per request, in the same order as the input batch.
    // All handles MUST be in Ready state. Handles in the batch MUST be distinct.
    decode(batch: List<DecodeRequest>) -> Result<List<Result<void, ARIAError>>, ARIAError>
}
```

### 5.2 Call Ordering Invariants

1. `model_info()` MUST be called before `declare_state()`.
2. `declare_state()` MUST be called before `prefill()` or `decode()`.
3. The harness MUST create a handle and pass it before any `prefill()` or `decode()` call on that handle.
4. `prefill()` MUST be called on a handle before the first `decode()` on that handle.
5. `decode()` calls on a single handle MUST be sequential (not concurrent).
6. The harness MUST NOT use a handle after destroying it.
7. After `declare_state()`, the model MUST NOT change the returned `StateSpec` for the lifetime of the model instance.
8. All handles within a single `prefill(batch)` or `decode(batch)` call MUST be distinct.
9. For batch calls, the runtime MUST update `current_length()` for all successful requests in the batch.
   Failed requests MUST NOT have their length updated.

### 5.3 Concurrency Contract

- Concurrent calls on **different** `StateHandle`s are safe and expected. The model MUST
  be implemented to support this without external locking.
- Concurrent calls on the **same** `StateHandle` are undefined behavior. The runtime is
  responsible for serializing access to a single handle.
- `model_info()` and `declare_state()` MUST be thread-safe (they may be called from any thread).

**Batch overloads are the production path.** The runtime assembles the batch each iteration
and calls `decode(batch)` or `prefill(batch)` once. The model executes all requests in a
single GPU operation. `max_batch` informs the model of the maximum batch size, enabling
pre-sizing of batch workspace. Single-handle overloads are for single-request use cases
(edge devices, testing, sequential workloads).

**Language bindings** that do not support method overloading SHOULD name the batch
overloads `prefill_batch` and `decode_batch` and document this convention in the binding.

### 5.4 Logit Properties

- All logits are pre-softmax scores (not probabilities).
- For valid (non-erroneous) inputs, logits MUST be finite (no NaN, no ±Inf).
- `prefill()` with `LogitsMode.LastPosition` (default) returns a `List` of length 1.
  Only the last position's logits are valid; the runtime uses these to select the next token.
- `prefill()` with `LogitsMode.AllPositions` returns a `List` of length `len(tokens)`.
  All positions are valid; use for speculative decoding verification and prompt logprobs.
- `decode()` (single-handle) returns exactly one `Logits`.
- `decode(batch)` returns one `Logits` per request, in input order.

---

## 5.5 ITokenizer — The Text↔Token Seam

```
interface ITokenizer {

    // Encode a single text string to a token ID sequence.
    encode(text: String) -> Result<List[TokenId], ARIAError>

    // Batch overload: encode multiple strings simultaneously.
    // Returns one List[TokenId] per input string, in the same order.
    encode(batch: List[String]) -> Result<List[List[TokenId]], ARIAError>

    // Decode a token ID sequence to a text string.
    decode(tokens: List[TokenId]) -> Result<String, ARIAError>

    // Batch overload: decode multiple token sequences simultaneously.
    // Returns one String per input sequence, in the same order.
    decode(batch: List[List[TokenId]]) -> Result<List[String], ARIAError>

    // Total number of tokens in the vocabulary.
    vocab_size() -> Result<uint32, ARIAError>

    // Special token identifiers (e.g., "eos_token", "bos_token", "pad_token", "unk_token").
    // Keys are canonical names; values are token IDs.
    special_tokens() -> Result<Map[String, TokenId], ARIAError>
}
```

The tokenizer vocabulary — merge rules, special tokens, token-to-ID mappings — is part of the
model's published contract. It MUST be identical between the default implementation and any
substituted implementation. The model MUST provide a tokenizer definition; this MAY be an
executable implementation or a standard data format (e.g., `tokenizer.json`) wrapped in this interface.

The runtime MAY substitute its own `ITokenizer` implementation (e.g., an accelerated tokenizer)
provided it is **semantically equivalent** (produces identical token IDs for all valid inputs).
Vocabulary incompatibility produces corrupt output; the runtime bears full responsibility for any substitution.

---

## 6. Semantic Tags

Semantic tags communicate the *role* of a state slot to the runtime without exposing
model internals. The runtime uses this information to apply appropriate memory management
policies. Critically: the runtime never needs to understand *how* the model computes with
the state — only *how to manage the memory*.

```
enum SemanticTag {

    // ─────────────────────────────────────────────────────────────
    // AttentionKV: Standard KV cache for transformer attention layers.
    // ─────────────────────────────────────────────────────────────
    //
    // The shape MUST reflect actual KV dimensions (num_kv_heads, head_dim),
    // which for GQA/MQA will differ from the query head count.
    //
    // KV slots are APPEND-ONLY within an active handle. The model writes
    // new entries at position current_length() and reads all prior positions.
    // The model MUST NOT overwrite positions < current_length().
    //
    // The runtime MAY preempt (swap out) an entire handle's state and restore
    // it later. Within an active, non-preempted handle, KV is never evicted.
    //
    // window_size: if set, positions older than window_size tokens are stale.
    //              The runtime MAY evict those pages without notifying the model.
    //              If absent, the runtime MUST NOT evict any page within a handle.
    AttentionKV {
        layer_idx:   uint32,
        is_key:      bool,
        window_size: Optional<uint32>,  // sliding window attention window, if any
    },

    // ─────────────────────────────────────────────────────────────
    // SSMState: Recurrent state for SSM architectures (Mamba, GDN, RWKV, etc.)
    // ─────────────────────────────────────────────────────────────
    //
    // Fixed size — does NOT grow with sequence length. The slot holds the
    // complete recurrent state for one sequence at one moment in time.
    //
    // The model FULLY OVERWRITES this slot on every prefill() and decode() call.
    // It is not append-only; the runtime MUST treat it as a dense, mutable tensor.
    //
    // SSMState MUST NOT be paged, prefix-shared, or evicted within an active handle.
    // The runtime MAY checkpoint (serialize) SSMState for handle preemption,
    // provided it restores exact values before the next call.
    SSMState {
        layer_idx: uint32,
    },

    // ─────────────────────────────────────────────────────────────
    // Custom: Model-defined state the runtime must not interpret.
    // ─────────────────────────────────────────────────────────────
    //
    // Use for state that does not fit AttentionKV or SSMState semantics.
    // Common uses: per-request adapter state, auxiliary loss accumulators,
    // routing decisions in novel architectures.
    //
    // WARNING: Custom slots break ARIA-Portable status.
    // Models MUST use tags registered in `spec/EXTENSIONS.md`.
    // Unregistered custom tags are considered non-compliant.
    //
    // The runtime MUST allocate exactly as declared and preserve exact values.
    // No eviction, no paging, no prefix sharing.
    Custom {
        extension_id: string, // MUST match an entry in EXTENSIONS.md
    },
}
```

### 6.1 Runtime Obligations by Tag

| Tag | Handle preemption (swap out/in) | Intra-handle eviction | Paging | Prefix sharing |
|-----|---------------------------------|-----------------------|--------|----------------|
| `AttentionKV` (no window) | Yes | No | Yes | Yes |
| `AttentionKV` (with window) | Yes | Yes (stale pages only) | Yes | Yes (non-stale) |
| `SSMState` | Yes (must serialize full state) | No | No | No |
| `Custom` | Yes (must serialize full state) | No | No | No |

---

## 7. StateHandle

### 7.1 Harness Ownership

The harness owns the `StateHandle` lifecycle. After `declare_state()`, the harness manages a
pool of state slots sized for `max_batch` concurrent requests:

1. Calls `declare_state()` to obtain the `StateSpec` and allocate state buffers.
2. Creates a handle (in `Uninitialized` state) for each new request, backed by one copy of the `StateSpec` allocation.
3. Passes handles to `prefill()` and `decode()` calls.
4. Destroys the handle when the request is complete, returning capacity to the pool.

The harness decides concurrency limits, pool sizing, and eviction policy. The model has no
visibility into how handles are managed or how many exist at any time.

### 7.2 Required Accessors

Every `StateHandle` MUST expose the following accessors to the model. The concrete API
is binding-specific (see §4.5), but the semantics are defined here:

```
handle.id()                -> uint64     // runtime-assigned unique identifier
                                          // stable for the handle's lifetime;
                                          // used in ARIAError payloads

handle.current_length()    -> uint32     // tokens processed so far
                                          // = sum of all prefill token counts
                                          //   + number of decode calls completed
                                          // Updated by the runtime after each
                                          // successful prefill() or decode() call.

handle.max_tokens()        -> uint32     // max_tokens from declare_state()

handle.tokens_remaining()  -> uint32     // max_tokens() - current_length()

handle.slot(id: string)    -> Result<TensorView, ARIAError>
                                          // mutable view into slot's allocated tensor
                                          // AttentionKV shape: [max_tokens, num_heads, head_dim]
                                          //   full allocation returned every call; model writes
                                          //   at index current_length() and reads 0..current_length()-1
                                          // SSMState shape: declared fixed dimensions (no Sequence)
                                          // Sequence resolved to max_tokens for all other slot types
                                          // valid only within the current call scope
```

### 7.3 Lifecycle State Machine

```
                  ┌─────────────────────────────────┐
                  │                                 ▼
Uninitialized ──[prefill]──► Ready ──[prefill, extend]──► Ready
                               │
                               ├──[decode]──► Ready  (tokens_remaining > 0)
                               │
                               └──[decode]──► Exhausted  (tokens_remaining == 0)

Ready ──────────────────────────────────────────────────────────────►┐
Exhausted ─────────────────────────────────────────────────────────►─┤
                                                               [runtime] ▼
                                                                    Released
```

State definitions:
- **Uninitialized**: handle allocated, no prefill yet. `current_length() == 0`.
- **Ready**: at least one prefill completed. `decode()` may proceed. `current_length() > 0`.
- **Exhausted**: `current_length() == max_tokens()`. No further decode is allowed.
- **Released**: handle returned to runtime pool. MUST NOT be passed to any model method.

**Important**: This lifecycle is the **model's view** of handle transitions. The runtime MAY
implement any internal state management strategy behind this interface — preemption
(swap-to-host), copy-on-write for prefix sharing, block table virtualization
(PagedAttention), RadixAttention tree structures, or any other optimization — provided
the model always observes the documented transitions on its `StateHandle` interface. The
model never sees these runtime-internal mechanisms directly; it only ever sees
`current_length()` and `TensorView` values consistent with the state machine above.

### 7.4 Multi-Call Prefill (Chunked Prefill and Multi-Turn)

`prefill()` MAY be called on a handle in either `Uninitialized` or `Ready` state.
A call on a `Ready` handle **extends** the sequence, appending new tokens after `current_length()`.

This enables two standard patterns:

**Chunked prefill** (long prompts broken into GPU-friendly chunks):
```
# Default mode=LastPosition; intermediate chunk logits discarded by runtime
prefill(tokens[0:512], handle)    # Uninitialized → Ready, current_length = 512  → List[Logits](len=1), discarded
prefill(tokens[512:1024], handle) # Ready → Ready,  current_length = 1024        → List[Logits](len=1), discarded
prefill(tokens[1024:], handle)    # Ready → Ready,  current_length = full_len    → List[Logits](len=1), used
decode(sampled, handle)           # standard decode
```

**Multi-turn conversation** (KV state preserved across turns):
```
prefill(system_prompt + turn_1_tokens, handle)  # Uninitialized → Ready
decode(...)  # generate assistant response; KV extends with each decode step
# handle.current_length() now = len(system_prompt) + len(turn_1_tokens) + len(response_tokens)
# User sends turn 2 — only NEW tokens need prefilling:
prefill(turn_2_tokens, handle)  # Ready → Ready, KV extended from current_length
decode(...)  # generate turn 2 response
```

> **Note**: Tokens processed by prior `prefill()` or `decode()` calls already have state
> in the handle. Only new, unprocessed tokens should be passed to subsequent `prefill()`
> calls. Re-prefilling already-cached tokens would corrupt the KV state.

**Invariant**: The total tokens across all prefill() calls plus all decode() calls on a
single handle MUST NOT exceed `max_tokens()`. If an extension would exceed capacity,
the model MUST return `SequenceTooLong`. The handle remains valid for shorter operations.

---

## 8. KV Dtype Negotiation

The model declares `supported_kv_dtypes` in `ModelInfo` (ordered by preference) and
declares a `dtype` on each `AttentionKV` slot in `StateSpec`.

The runtime selects an actual allocation dtype from `supported_kv_dtypes` and allocates
`AttentionKV` slots at that dtype. The runtime communicates the selected dtype to the
model via the `TensorView.dtype` property of the slot accessor.

**Rules:**
1. The runtime MUST select a dtype from `ModelInfo.supported_kv_dtypes`. If the list
   is empty, the runtime MUST use the `StateSlot.dtype` as declared.
2. The model MUST handle any dtype in its `supported_kv_dtypes` list. If it cannot,
   that dtype MUST NOT appear in the list.
3. If the runtime cannot allocate at any dtype in `supported_kv_dtypes`, it MUST
   return `NoDtypeCompatible`.
4. `SSMState` and `Custom` slots are allocated at their declared `dtype`. Negotiation
   does not apply to non-KV slots.

**Example (llama.cpp KV quantization)**:
```
# Model declares:
supported_kv_dtypes: [Float16, Float8E4M3, Int8]

# Runtime (llama.cpp, user requested -ctk q8_0) allocates KV at Int8.
# Model checks handle.slot("kv_k_0").dtype == Int8, dequantizes appropriately.
```

---

## 9. Error Model

```
enum ARIAError {

    // ── Lifecycle errors ──────────────────────────────────────────────────
    // decode() called before prefill()
    NotPrefilled { handle_id: uint64 },

    // decode() or prefill() on an Exhausted handle
    StateExhausted { handle_id: uint64, max_tokens: uint32 },

    // Handle used after destruction, or handle from a different model instance
    InvalidHandle { handle_id: uint64 },

    // ── Capacity errors ───────────────────────────────────────────────────
    // Total tokens would exceed max_tokens
    SequenceTooLong { length: uint32, max: uint32 },

    // tokens argument to prefill() is empty
    EmptyInput,

    // ── Allocation errors ─────────────────────────────────────────────────
    // Runtime cannot allocate a slot as declared (insufficient memory, unsupported dtype)
    SlotUnsupported { slot_id: string, reason: string },

    // Runtime cannot allocate any KV dtype from the model's supported list
    NoDtypeCompatible { slot_id: string, supported: List<DType> },

    // Out of memory during state pool allocation
    OutOfMemory { bytes_requested: uint64 },

    // ── Shape and type errors ─────────────────────────────────────────────
    // Token value exceeds vocab_size
    TokenOutOfRange { token: TokenId, vocab_size: uint32 },

    // Slot dtype mismatch between StateSpec and what was allocated
    DtypeMismatch { slot_id: string, expected: DType, actual: DType },

    // ── Model errors ──────────────────────────────────────────────────────
    // Non-recoverable internal model failure (e.g., kernel crash, NaN explosion)
    ModelError { code: uint32, detail: string, fatal: bool },

    // ARIA version mismatch between model and runtime
    VersionIncompatible { model_version: (uint32, uint32), runtime_supports: (uint32, uint32) },
}
```

**Error taxonomy**:

| Category | Raised by | Fatal | Retriable | Handle valid after? |
|----------|-----------|-------|-----------|-------------------|
| `NotPrefilled` | Model | No | Yes (call prefill first) | Yes |
| `StateExhausted` | Model | No | No (handle is done) | No |
| `InvalidHandle` | Model | No | No | No |
| `SequenceTooLong` | Model | No | Yes (retry with fewer tokens) | Yes (handle usable for shorter seqs) |
| `EmptyInput` | Model | No | Yes (fix input) | Yes |
| `SlotUnsupported` | Runtime | Yes | No | N/A |
| `NoDtypeCompatible` | Runtime | Yes | No | N/A |
| `OutOfMemory` | Runtime | Yes | Implementation-defined | N/A |
| `TokenOutOfRange` | Model | No | Yes (fix input) | Yes |
| `DtypeMismatch` | Runtime | Yes | No | No |
| `ModelError` (fatal) | Model | Yes | No | No |
| `ModelError` (non-fatal) | Model | No | Yes | Implementation-defined |
| `VersionIncompatible` | Runtime | Yes | No | N/A |

The runtime MUST propagate all errors to the consumer. Silent fallback is a spec violation.

**Batch error semantics**: If any request in a `prefill(batch)` or `decode(batch)` call
produces an error, the entire batch call returns `Err`. Results for successful requests in
the same batch are not returned. Handle state for requests that did not cause the error is
implementation-defined: the runtime MUST treat affected handles as suspect and SHOULD
destroy and recreate them before reuse. The consumer MUST retry failed requests individually
or in a new batch.

---

## 10. Invariants

The following invariants MUST hold in any conforming implementation.

**Model invariants (implementor obligations):**
1. `declare_state()` is deterministic: identical `(max_batch, max_tokens)` inputs produce identical `StateSpec`.
2. Slot IDs within a `StateSpec` are unique.
3. `AttentionKV` slots are append-only: the model MUST NOT write to positions `< handle.current_length()`.
   The model reads `handle.current_length()` at the *start* of each call; it reflects tokens processed
   in all prior calls, not the current call. Implementations that require context extension (e.g.,
   context shifting) MUST declare the `aria.core.context_shift` capability in `ModelInfo` (see `spec/EXTENSIONS.md`)
   and MUST document the modified write pattern.
4. `SSMState` slots are mutable recurrent state. The model typically reads the previous state, computes the new state, and overwrites the slot.
   - `SSMState` slot shapes MUST NOT use `Dimension.Sequence`; recurrent state is fixed-size by definition.
   - The runtime MUST NOT zero-initialize `SSMState` buffers between calls; the model MAY read existing values before overwriting.
   - The model MUST initialize the state on the first call (when `current_length == 0`).
5. The model MUST NOT allocate persistent per-request state outside the StateHandle.
   All mutable per-request state MUST be accessed exclusively through handle slot
   accessors. Transient computation buffers (activation workspace, intermediate tensors,
   logit output buffers) allocated and freed within a single call are permitted.
   Static buffers (weight tensors, reusable scratchpads) SHOULD be allocated at model
   load time for predictable memory behavior.
6. The model MUST NOT retain a `TensorView` reference beyond the scope of the call that produced it.
7. The model MUST be safe to call from concurrent threads on different handles without external locking.
8. Logits returned MUST be finite (no NaN, no ±Inf) for valid inputs.

**Runtime invariants (implementor obligations):**
1. The runtime MUST allocate exactly the slots declared in `StateSpec` at the negotiated dtypes.
2. The runtime MUST NOT pass the same handle to two concurrent `prefill()` or `decode()` calls.
3. The runtime MUST NOT call `decode()` on a handle in `Exhausted` or `Released` state.
4. The runtime MUST NOT call `prefill()` on a handle in `Released` state.
5. For `AttentionKV` slots with no `window_size`: the runtime MUST NOT evict or modify KV data within an active, non-preempted handle.
6. For handle preemption (swap to host): the runtime MUST restore all slot data exactly before the next call on that handle.
7. The runtime MUST propagate all `ARIAError` values to the consumer. The runtime MUST NOT silently suppress or retry errors without the consumer's knowledge.
8. Handle IDs MUST be unique among all active handles for a given model instance.
9. After a successful `prefill(tokens, handle)` call returns, the runtime MUST
   set `handle.current_length()` to its previous value plus `len(tokens)`.
10. After a successful `decode(token, handle)` call returns, the runtime MUST
    set `handle.current_length()` to its previous value plus 1.

---

## 11. Governance and Innovation Freedom

ARIA is designed to maximize innovation freedom while enforcing a strict interface contract.
Governance details are maintained in `CONTRIBUTING.md`.

### 11.1 Model Implementation Freedom

A model implementation conforming to ARIA is free to:

- Use **any attention algorithm**: FlashAttention 2/3, FlashInfer, xFormers, naive SDPA
- Use **any weight precision**: FP4, FP8, INT4 AWQ/GPTQ, BF16, FP16, mixed
- Use **any architecture**: pure attention, pure SSM, hybrid, MoE, sparse, future designs
- Use **any positional encoding**: RoPE, ALiBi, NoPE, learned absolute, future schemes
- Implement **tensor parallelism** or **pipeline parallelism** internally and transparently
- Capture **CUDA graphs** or other compute graphs internally
- Use **batch overloads**: call `decode(batch)` or `prefill(batch)` to fuse multiple
  requests into a single GPU kernel — `max_batch` exists precisely to pre-size batch
  workspace for this path
- Implement **context extension** (YaRN, NTK scaling, etc.) internally
- Apply **speculative decoding** as a model-level optimization (draft model is a separate
  `ILanguageModel` instance; verifier uses `prefill()` with `LogitsMode.AllPositions`)
- Use **any hardware**: CUDA, ROCm, Metal, CPU, NPU, custom accelerators

The only constraint: the model accesses its per-request state exclusively through the
`StateHandle` accessors defined in §7.2.

### 11.2 Runtime Implementation Freedom

A runtime conforming to ARIA is free to:

- Use **any paging strategy**: PagedAttention, virtual KV addressing, contiguous pools
- Use **any eviction policy** for eligible slots: LRU, LFU, SLO-aware, learned
- Use **any batching strategy**: continuous batching, iteration-level scheduling, micro-batching
- Use **any scheduling algorithm**: FCFS, priority queues, SLO-aware schedulers
- Implement **prefix KV caching** (RadixAttention, hash-based) transparently — the model
  never knows which physical pages are shared; it only writes through the handle
- Implement **KV cache quantization** within the negotiated dtype range (§8)
- Implement **handle preemption** (swap to host, restore on demand)
- Implement **speculative decoding** by running a draft model alongside the target model
  and verifying with `prefill()` (use `LogitsMode.AllPositions` for this)
- Use **any sampling strategy**: temperature, top-p, top-k, beam search, guided decoding
- Implement **streaming** by consuming `decode()` logits one token at a time
- Implement **LoRA / adapter management** without ARIA involvement (adapters are weight
  management, not state management)
- Use **any request queue** strategy: priority, fair, rate-limited, preemptive
- Implement **async execution** by wrapping ARIA's synchronous interface in an async executor

The only constraint: the runtime MUST allocate state as declared in `StateSpec`, pass handles
correctly, and propagate errors faithfully.

---

## 12. Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Consumer Layer                                 │
│              OpenAI-compatible API (chat, completions)               │
│           Streaming, cancellation, rate limiting, auth               │
│    ← Runtime is FREE to innovate: SLO-aware scheduling, autoscaling  │
├─────────────────────────────────────────────────────────────────────┤
│                       Runtime Layer                                   │
│   SGLang · vLLM · llama.cpp · TGI · LMDeploy · future runtimes      │
│   Batching · scheduling · paging · prefix caching · KV eviction      │
│   Sampling · guided decoding · speculative decoding orchestration     │
│    ← Runtime is FREE to innovate: new schedulers, prefill strategies  │
├═════════════════════════════════════════════════════════════════════╡
║               ── ARIA CONTRACT (this specification) ──               ║
║           model_info · declare_state · prefill · decode              ║
║                    ITokenizer (text ↔ token seam)                    ║
╞═════════════════════════════════════════════════════════════════════╡
│                        Model Layer                                    │
│   Qwen3 · LLaMA · Mamba · Jamba · Mixtral · future architectures     │
│   Weights · compute kernels · internal batching · architecture logic  │
│    ← Model is FREE to innovate: new architectures, new state types    │
├─────────────────────────────────────────────────────────────────────┤
│                       Hardware Layer                                  │
│   NVIDIA CUTLASS · Triton · FlashInfer · Apple Metal · CPU SIMD      │
│   CUDA / ROCm / OneAPI / NPU compilers / custom accelerators          │
│    ← Hardware is FREE to innovate: new dtypes, new memory models      │
└─────────────────────────────────────────────────────────────────────┘
```

The ARIA contract (double line) is the single boundary that all parties agree on.
Every layer above and below it is **completely unconstrained** by the spec.
ARIA standardizes exactly the minimum surface required for interoperability — nothing more.

**For operators**: concerns above the ARIA line (consumer API, streaming, multi-tenancy,
auth, scheduling) remain the runtime's domain — competitive innovation space, not
standardized. ARIA operates below that layer.

---

## 13. Design Rationale

### 13.1 Four Methods: One Metadata Query, Three Generation Primitives

`model_info()` is a load-time metadata query, not part of the generation loop.
`declare_state / prefill / decode` IS the complete autoregressive generation loop:
declare what you need → populate state from prompt → generate one token at a time.

There is no fourth generation primitive. Everything else (sampling, beam search, speculative
decoding, batching) is policy applied above or below this loop. Adding methods for these would
couple the spec to specific generation strategies that are appropriately the runtime's domain.

### 13.2 `declare_state()` vs. a Config File

Config files are parsed at load time with no runtime context. `declare_state()` is called
with `(max_batch, max_tokens)` — the runtime's actual allocation parameters. This lets the
model declare shapes as functions of allocation context (e.g., a model running with
max_batch=1 might allocate differently than one running with max_batch=256). It also makes
the contract verifiable at call time with full type checking, rather than at config parse time
with string matching.

### 13.3 Semantic Tags vs. Custom Memory

A fully opaque state contract would reduce the spec to "give me N bytes per request."
This prevents runtimes from making intelligent policy decisions. An `AttentionKV` slot can
be paged, prefix-shared, and evicted when the request window passes. An `SSMState` slot
must be preserved exactly and cannot be paged across token positions. The runtime needs
to know this difference — not to understand the model's math, but to manage memory correctly.
Semantic tags communicate *role* without exposing *implementation*.

#### On Architecture Disclosure

`declare_state()` reveals layer count and per-layer state types. For organizations with
proprietary model architectures, this may be a concern.

Models MAY aggregate state into fewer slots using `Custom` tags to preserve architectural
confidentiality. Example: instead of per-layer KV slots, declare a single
`Custom{extension_id: "aria.core.attention_state"}` slot with total size requirements. The model
manages internal layout; the runtime allocates the blob and preserves it.

This sacrifices runtime optimization (the runtime cannot page, prefix-share, or evict
within an `Custom` slot) in exchange for confidentiality. It is a supported pattern,
not a workaround.

The ARIA recommendation is: use Standard tags wherever possible for maximum runtime
optimization potential, and fall back to Custom aggregation only where architectural
confidentiality is a hard requirement. See `CONFORMANCE.md` for portability implications.

### 13.4 Per-Request Handles with Batch Overloads

ARIA provides both single-handle and batch overloads for `prefill()` and `decode()`.
The two paths serve different use cases:

- **Single-handle overloads** (`prefill(tokens, state)`, `decode(token, state)`): For
  single-request workloads, edge devices, sequential testing, and language bindings where
  per-call simplicity matters.

- **Batch overloads** (`prefill(batch)`, `decode(batch)`): The production path. The
  runtime assembles requests each iteration and calls the model once. The model executes
  all tokens in a single GPU kernel. This is the correct design for continuous batching,
  iteration-level scheduling, and high-throughput serving.

`max_batch` in `declare_state()` declares capacity — the model uses it to pre-size batch
workspace (intermediate tensors, activation buffers) at load time rather than per-call.

### 13.5 Why Not Extend ONNX or WASI-NN

ONNX's static computational graph cannot express dynamic control flow (MoE routing,
conditional computation) or mutable recurrent state without vendor extensions that break
cross-platform portability. Adding stateful decode would require a ground-up redesign.

WASI-NN is architecturally promising but its inference contract is stateless by design.
Its graph execution model does not have a `declare_state` analogue. Retrofitting stateful
autoregressive decode would require changes as substantial as defining a new interface.
We chose to define the correct abstraction directly.

Neither this analysis nor ARIA's existence is a criticism of ONNX or WASI-NN. They solve
different problems in the same space. ARIA fills the gap both leave open.

### 13.6 Synchronous-First Design

ARIA's interface is synchronous. Async execution, token streaming, and request cancellation
are above ARIA — the runtime wraps `decode()` in an async executor. This keeps the spec
language-agnostic and avoids prescribing a specific async model (futures, coroutines,
callbacks, CSP channels) that would conflict with different language ecosystems.

### 13.7 ARIA Works With Compiled Graphs

ARIA does not assume eager (interpreter-based) model execution. A conformant model may be
an ahead-of-time compiled computation graph — a TensorRT engine, a Qualcomm QNN graph,
a CoreML model, an Intel SynapseAI binary, or any custom NPU artifact — that implements
`ILanguageModel` through its native execution mechanism.

In this case, `prefill()` and `decode()` are logical operations, not necessarily separate
GPU kernel dispatches. The compiled graph's execution engine calls the appropriate compiled
kernel and writes results into the `StateHandle` slots. The ARIA contract is satisfied at
the interface level regardless of how underlying compute is orchestrated.

This applies to: Qualcomm Hexagon NPU (QNN), Apple Silicon (CoreML), Intel Gaudi
(SynapseAI), NVIDIA TensorRT, and any future hardware that benefits from ahead-of-time
compilation.

---

## 14. Versioning and Capability Negotiation

### 14.1 Version Format

ARIA uses `major.minor` versioning:
- **Major bump**: breaking change (new required method, changed signature, removed method)
- **Minor bump**: additive change (new optional semantic tag, new error variant, new field
  in an existing struct)

A model targets a specific version by declaring it in `ModelInfo.aria_version`.
A runtime declares the versions it supports.

**Compatibility rules**:
- **Within a major version**: A runtime supporting ARIA `X.N` MUST support any model
  targeting ARIA `X.M` where M ≤ N (full minor-version backward compatibility).
- **Across major versions**: A runtime MAY support multiple major versions but is not
  required to. A runtime MUST return `VersionIncompatible` for any major version it
  does not support.
- **Deprecation**: When a new major version is released, the previous major version
  receives critical fixes for 24 months (per CONTRIBUTING.md).

### 14.2 Conformance

A **conforming model** MUST implement all six methods of `ILanguageModel` (`model_info`,
`declare_state`, `prefill` ×2 overloads, `decode` ×2 overloads), provide a conformant
`ITokenizer` via `ModelInfo.default_tokenizer`, and uphold all model invariants in §10.

A **conforming runtime** MUST call methods in the specified order, allocate all declared slots,
provide all required `StateHandle` accessors, and uphold all runtime invariants in §10.

Partial conformance (implementing a subset of methods) is not valid.

ARIA 0.5-draft defines one conformance level with one **portability profile**:

- **ARIA-Conformant**: Full contract including `Custom` semantic tags, dtypes, and layouts.
  Sufficient for interoperability where both sides agree on Custom definitions.
- **ARIA-Portable** (portability profile): All state uses Standard semantic tags only
  (zero `Custom` slots). An ARIA-Portable model is guaranteed to work with any ARIA-Portable
  runtime without out-of-band agreements. This is a stricter subset of conformant.

See `CONFORMANCE.md` for the full checklists and portability requirements.

### 14.3 Intellectual Property

ARIA specification text is available under
[Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Reference implementations and tooling are available under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

**Patent non-assertion**: The authors make no patent claims on the interface contracts,
semantic tag system, state declaration mechanism, KV dtype negotiation protocol, or any
other concept described in this specification. To the best of the authors' knowledge,
the interfaces described herein are free to implement without patent license.

Contributions to this specification are accepted under the Developer Certificate of Origin
(DCO). By contributing, you certify that you have the right to submit the contribution
under these terms. See `CONTRIBUTING.md` for the complete policy.

---

## 15. Future Extensions (ARIA 1.0+)

This section notes the planned evolution of ARIA to avoid conflicting design decisions
in 0.6. The full roadmap with dates, milestones, and process is maintained in
[`ROADMAP.md`](../ROADMAP.md).

**ARIA 1.0 planned additions** (H2 2026, requires 2+ independent 0.6 implementations):
`CrossAttentionKV` tag, full encoder-decoder scope, LoRA/adapter interface,
`truncate()` operation, multi-token prediction, `BatchProfile` interface,
CUDA graph capture hints, conformance tiers.

> **Known limitation (0.6)**: `declare_state()` accepts a single `max_tokens` value,
> meaning all handles share the same maximum sequence length allocation. In production
> deployments with mixed request lengths, this may cause memory overallocation for short
> requests. The `BatchProfile` interface planned for ARIA 1.0 will address this by
> allowing per-profile max_tokens configuration. Implementors building memory-efficient
> runtimes should consider tracking actual `current_length()` per handle to reclaim
> unused sequence capacity within their allocator.

**ARIA 2.0 planned additions** (2027):
Multimodal inputs, distributed inference contract, training/fine-tuning hooks.

Each item will follow the process in `CONTRIBUTING.md` — proposal, two implementations,
Steering consensus — before merging into the spec.

---

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
    ARIA_ERR_MODEL_ERROR = 7
} aria_error_t;

typedef enum {
    ARIA_FLOAT32 = 0,
    ARIA_BFLOAT16 = 1,
    ARIA_FLOAT16 = 2,
    ARIA_INT8 = 3,
    // Int4 removed in 0.5. Use ARIA_CUSTOM_INT for quantized formats.
    ARIA_CUSTOM_FLOAT = 100, // e_bits, m_bits packed
    ARIA_CUSTOM_INT = 101    // scale_bits, data_bits packed
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

// ============================================================================
// 3. Interface VTable
// ============================================================================

typedef struct {
    // Lifecycle
    aria_error_t (*get_info)(aria_model_t model, aria_model_info_t* out_info);
    
    // State Management
    aria_error_t (*declare_state)(aria_model_t model, uint32_t max_batch, uint32_t max_tokens, void** out_spec);
    
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

    // Batch Inference (Production Path)
    // tokens: flattened array of [num_sequences * (tokens_per_seq)]
    // num_tokens_per_seq: array of length num_sequences
    // states: array of handles, length num_sequences
    // out_logits: array of tensor_views, length num_sequences (can be NULL to skip all)
    aria_error_t (*prefill_batch)(aria_model_t model, 
                                  uint32_t num_sequences,
                                  const uint32_t* tokens, 
                                  const uint32_t* num_tokens_per_seq,
                                  aria_state_handle_t* states, 
                                  aria_tensor_view_t* out_logits);

    // tokens: array of token IDs, length num_sequences
    // states: array of handles, length num_sequences
    // out_logits: array of tensor_views, length num_sequences (can be NULL to skip all)
    aria_error_t (*decode_batch)(aria_model_t model, 
                                 uint32_t num_sequences,
                                 const uint32_t* tokens, 
                                 aria_state_handle_t* states, 
                                 aria_tensor_view_t* out_logits);

    // Tokenizer (Optional - can be NULL if model relies on external tokenizer)
    // Returns number of tokens written to out_tokens.
    // If out_tokens is NULL, returns required capacity.
    int32_t (*encode)(aria_model_t model, const char* text, uint32_t* out_tokens, uint32_t capacity);
    
    // Returns number of bytes written to out_text (including null terminator).
    // If out_text is NULL, returns required capacity.
    int32_t (*decode_token)(aria_model_t model, uint32_t token, char* out_text, uint32_t capacity);

} aria_interface_t;

// ============================================================================
// 4. Entry Point
// ============================================================================

// Every ARIA shared library must export this function.
// Returns the interface vtable for the loaded model.
aria_interface_t* aria_load_model(const char* path);

#ifdef __cplusplus
}
#endif

#endif // ARIA_H
```

The ABI includes:
1.  **Single and Batch Inference**: `prefill`, `decode`, `prefill_batch`, `decode_batch`.
2.  **Tokenizer Interface**: Optional `encode` and `decode_token` function pointers for models
    that embed tokenizer logic. Runtimes MUST check for NULL before calling.
3.  **State Handle Lifecycle**: `aria_state_handle_t` is an opaque pointer owned by the RUNTIME.
    -   The runtime allocates, manages, and destroys all handle instances.
    -   The model receives handles as read/write parameters.
    -   **CRITICAL**: The model MUST NOT attempt to free, delete, or call a destructor on a handle.
    -   **CRITICAL**: The model MUST NOT store handle pointers beyond the scope of a call (unless
        specifically designed as a stateful C++ object, which is discouraged).

### 16.2 Calling Convention
All exported functions use the standard C calling convention (`cdecl` on x86/x64).
Symbols are exported with visibility `default`.

---

## Appendix A: ARIA in Context — Complementary Specifications

ARIA is not a replacement for any existing specification. Each spec solves a different
problem. The table below shows how ARIA is **additive**, not competitive.

| Spec | What it does well | The gap ARIA fills | How they work together |
|------|------------------|-------------------|------------------------|
| **ONNX** | Static model interchange, cross-framework portability, operator coverage | Static graph — cannot express mutable per-request decode state or dynamic control flow (MoE routing) | ONNX exports model weights; ARIA standardizes the serving-time stateful inference contract |
| **WASI-NN** | Hardware-agnostic inference backend interface; WebAssembly portability | Stateless contract — no per-request lifecycle, no state allocation protocol | ARIA's `ILanguageModel` could be adopted as a WASI-NN Stateful Inference Profile (see §13.5) |
| **GGUF** | Single-file distribution with weights, metadata, and tokenizer config in one file | File format — no serving-time contract; how the model runs is undefined | GGUF distributes the model file; ARIA defines how a llama.cpp runtime calls into it at serving time |
| **SafeTensors** | Safe, fast, ML-framework-agnostic weight serialization; zero-copy loading | Weight storage only — no inference contract | SafeTensors stores weights; ARIA defines the inference interface |
| **TIS Backend API** | Triton Inference Server's framework-to-backend serving contract | No standardized stateful decode — each backend implements its own KV state management | Complementary serving contracts at different layers; TIS consumer API above, ARIA model-runtime boundary below |
| **HF `generate()`** | Broad model support, sampling library, Python ecosystem integration | Python-only; exposes model internals (past_key_values); not language-agnostic | HuggingFace models can expose an ARIA interface for use outside Python; generate() remains the sampling-layer API |

**The single gap none of these address**: a language-agnostic, vendor-neutral,
architecture-agnostic contract for the **stateful autoregressive decode loop with
heterogeneous state types**. That gap is what ARIA fills.

These complementary specs are also potential integration partners for demonstrating ARIA's
value: an ONNX-exported model wrapped in an ARIA interface, or an ARIA runtime adapter for
WASI-NN, would demonstrate interoperability rather than competition.

---

## Appendix B: Framework Compatibility Notes

### vLLM

- **PagedAttention**: `AttentionKV` + `Layout::Paged` maps directly to vLLM's physical block
  tables. vLLM chooses its own `page_tokens` granularity; the model declares a preference.
  See §4.5.2 for the two conformant TensorView strategies for paged layouts.
- **Continuous batching**: ARIA handles are per-request. vLLM's iteration-level scheduler
  assembles batches and can call `decode()` across N handles per scheduling step; the model
  implementation batches the GPU kernel internally.
- **Prefix caching**: vLLM's RadixAttention is a runtime optimization on `AttentionKV` pages;
  the model writes through its handle and never observes physical sharing.
- **Speculative decoding**: vLLM drives a draft model as a separate `ILanguageModel` instance;
  the target model verifies via `prefill(mode=LogitsMode.AllPositions)`.

### SGLang

- **Chunked prefill**: `prefill()` may be called multiple times on a `Ready` handle.
  SGLang can chunk any long prompt into GPU-efficient slices without model changes.
- **RadixAttention**: runtime-internal; invisible to the model.
- **Tensor parallelism**: model-internal; the model's `decode()` implementation may use
  NCCL collectives. ARIA's synchronous interface wraps the parallel call.

### llama.cpp

- **KV quantization** (e.g., `-ctk q8_0`): model declares `supported_kv_dtypes`; llama.cpp
  selects a compatible dtype and allocates accordingly. Model reads actual dtype from
  `handle.slot().dtype` and dequantizes in the attention kernel.
- **Context shifting**: llama.cpp's context shifting is a model-level optimization where the
  model overwrites its own KV entries to extend context. In ARIA, this is valid — the model
  controls compute and may rewrite KV positions < `current_length()` if it implements
  context shifting. `AttentionKV`'s append-only invariant is relaxed for implementations
  that explicitly implement context extension algorithms.
- **Metal / CPU backends**: ARIA is hardware-agnostic. llama.cpp selects its backend
  transparently within its model implementation.

### TGI (Text Generation Inference)

- Flash Attention 2 is model-internal. TGI implements ARIA's `prefill/decode` backed by FA2.
- Token streaming: TGI's streaming is above ARIA — it consumes `decode()` results per step.

### LMDeploy (TurboMind engine)

- TurboMind manages its own block tables and persistent batch state. In an ARIA binding,
  TurboMind implements the runtime side — allocating handles backed by its internal block
  manager, exposing ARIA-compliant accessors.

### Intel Gaudi / IPEX-LLM (intel-analytics/ipex-llm)

- Intel Gaudi accelerators use SynapseAI with a graph compiler that pre-fuses ops at
  load time. In an ARIA binding, `declare_state()` triggers graph compilation; subsequent
  `prefill/decode` calls dispatch compiled kernels.
- `FloatCustom` dtypes cover Intel-specific quantization formats where `Float8E4M3` or
  standard enums do not match exactly (e.g., Intel FP8 E4M3FNUZ bias variant).
- IPEX-LLM's CPU optimization paths (AMX, AVX-512) are model-internal; ARIA's interface
  is unchanged.

### Apple Silicon (CoreML / Metal)

- A CoreML-compiled model implements `ILanguageModel` via a Swift/ObjC bridge. `prefill()`
  and `decode()` dispatch Metal compute shaders. State slots map to Metal buffers allocated
  by the runtime in unified memory.
- Memory coherence is via Metal command buffer completion (§4.5.1); the runtime waits on
  the command buffer before reading logits.
- `DType.Float16` maps naturally to Apple's preferred `float16` compute dtype. KV in
  `Int8` is also viable on M-series hardware.

### Qualcomm QNN (Hexagon NPU)

- A QNN-compiled model wraps a Qualcomm graph in an ARIA binding. `declare_state()` returns
  the tensor shapes required by the compiled graph. `prefill/decode` dispatch QNN graph
  execution on the Hexagon NPU.
- State memory is allocated via QNN's memory API and exposed through ARIA's TensorView
  accessors; coherence uses QNN completion callbacks mapped to synchronous return.
- `FloatCustom` covers Qualcomm-specific quantization formats where standard enums don't
  map cleanly.

---

## Appendix C: Reference Implementation Skeleton

```python
# Illustrates a hybrid attention+SSM model's ARIA implementation.
# Not production code — shows the contract shape only.

from aria import (
    ILanguageModel, ModelInfo, StateSpec, StateSlot, StateHandle,
    SemanticTag, DType, Layout, Dimension, ARIAError, Result, ITokenizer, TensorView
)

class HybridModel(ILanguageModel):

    def model_info(self) -> Result[ModelInfo, ARIAError]:
        return ModelInfo(
            aria_version=(0, 5),
            vocab_size=152064,
            model_name="HybridModel-7B",
            max_sequence_length=32768,
            supported_kv_dtypes=[DType.Float16, DType.Float8E4M3, DType.Int8],
            capabilities=["aria.core.context_shift", "aria.core.layout.paged"],
            default_tokenizer=MyTokenizer(),
        )

    def declare_state(self, max_batch: int, max_tokens: int) -> Result[StateSpec, ARIAError]:
        slots = []
        for i in range(self.num_layers):
            if self.layer_type(i) == "attention":
                # GQA: use num_kv_heads, not num_q_heads
                for is_key in [True, False]:
                    slots.append(StateSlot(
                        id=f"kv_{'k' if is_key else 'v'}_{i}",
                        semantic_tag=SemanticTag.AttentionKV(layer_idx=i, is_key=is_key),
                        shape=[
                            Dimension.Sequence,
                            Dimension.ModelParam("num_kv_heads", self.num_kv_heads),
                            Dimension.ModelParam("head_dim", self.head_dim),
                        ],
                        dtype=DType.Float16,          # preferred; negotiated via §8
                        layout=Layout.Paged(preferred_page_tokens=16),
                    ))
            elif self.layer_type(i) == "ssm":
                slots.append(StateSlot(
                    id=f"ssm_{i}",
                    semantic_tag=SemanticTag.SSMState(layer_idx=i),
                    shape=[
                        Dimension.ModelParam("d_state", self.d_state),
                        Dimension.ModelParam("d_inner", self.d_inner),
                    ],
                    dtype=DType.Float16,
                    layout=Layout.Contiguous,
                ))
        return StateSpec(slots=slots)

    def prefill(self, tokens: list[int], state: StateHandle,
                out_logits: TensorView = None) -> Result[None, ARIAError]:
        pos_offset = state.current_length()
        num_tokens = len(tokens)

        # Decide which positions need logits based on output buffer shape
        start_logits_at = num_tokens  # default: none (S10 optimization)
        if out_logits:
            # B1: Write directly to caller-owned buffer
            if out_logits.shape[0] == 1:            # Last position only
                start_logits_at = num_tokens - 1
            elif out_logits.shape[0] == num_tokens: # All positions
                start_logits_at = 0

        for i, token in enumerate(tokens):
            abs_pos = pos_offset + i
            compute_logits = (i >= start_logits_at)

            # Forward pass writes to state immediately
            logits = self._forward(token, abs_pos, state, write_state=True,
                                   return_logits=compute_logits)
            if compute_logits:
                out_idx = i if start_logits_at == 0 else 0
                out_logits[out_idx] = logits

        return None

    def decode(self, token: int, state: StateHandle,
               out_logits: TensorView = None) -> Result[None, ARIAError]:
        pos = state.current_length()
        # Decode always computes logits if buffer provided
        logits = self._forward(token, pos, state, write_state=True,
                               return_logits=(out_logits is not None))
        if out_logits:
            out_logits[0] = logits  # B1: Zero-copy write
        return None

    def _forward(self, token, position, state, write_state, return_logits=True):
        x = self.embed(token)
        for i, layer in enumerate(self.layers):
            if layer.type == "attention":
                # Read KV for all prior positions; write current KV
                kv_k = state.slot(f"kv_k_{i}")   # TensorView into runtime memory
                kv_v = state.slot(f"kv_v_{i}")
                x = layer(x, position, kv_k, kv_v, write=write_state)
            elif layer.type == "ssm":
                # S9: SSM state initialized if new, overwritten if existing
                ssm = state.slot(f"ssm_{i}")
                x = layer(x, ssm, write=write_state)
        if return_logits:
            return self.lm_head(x)
        return None
```
