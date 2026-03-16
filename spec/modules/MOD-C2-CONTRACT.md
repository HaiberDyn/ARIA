<!-- ARIA module: MOD-C2-CONTRACT.md — do not edit directly; edit this file in spec/modules/ -->
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

    // Request cancellation of an in-flight prefill() or decode() on this handle.
    //
    // Thread-safe: MAY be called from any thread, including concurrently with an
    // in-flight prefill() or decode() on the same handle. This is the primary use case:
    // the runtime receives a consumer cancellation event and calls cancel() while the
    // model is executing on a worker thread.
    //
    // Semantics:
    //   - If a call is in flight: the model MUST abort as soon as possible (cooperative
    //     cancellation). The in-flight call MUST return Err(Cancelled { handle_id }).
    //   - If no call is in flight: cancel() marks the handle for immediate release.
    //   - After cancel() returns: handle is in Released state. Runtime MUST NOT pass
    //     it to any further model calls.
    //   - Idempotent: cancel() on an already-Released handle MUST be a no-op, Ok(void).
    //   - Partial state: the model is not required to undo writes made before the
    //     cancellation point. The runtime discards the handle regardless.
    cancel(state: StateHandle) -> Result<void, ARIAError>
}
```

### 5.2 Call Ordering Invariants

1. `model_info()` MUST be called before `declare_state()`.
2. `declare_state()` MUST be called before `prefill()` or `decode()`.
3. The harness MUST create a handle and pass it before any `prefill()` or `decode()` call on that handle.
4. `prefill()` MUST be called on a handle before the first `decode()` on that handle.
5. `decode()` calls on a single handle MUST be sequential (not concurrent).
6. The harness MUST NOT use a handle after destroying it or after `cancel()` returns.
7. After `declare_state()`, the model MUST NOT change the returned `StateSpec` for the lifetime of the model instance.
8. All handles within a single `prefill(batch)` or `decode(batch)` call MUST be distinct.
9. For batch calls, the runtime MUST update `current_length()` for all successful requests in the batch.
   Failed requests MUST NOT have their length updated.
10. `cancel()` is the only method that MAY be called concurrently with an in-flight call on the same handle.

### 5.3 Concurrency Contract

- Concurrent calls on **different** `StateHandle`s are safe and expected. The model MUST
  be implemented to support this without external locking.
- Concurrent calls on the **same** `StateHandle` are undefined behavior. The runtime is
  responsible for serializing access to a single handle.
- `model_info()` and `declare_state()` MUST be thread-safe (they may be called from any thread).

**Both overloads are first-class interfaces.** The choice between them depends on deployment:

- **Batch overloads** (`prefill(batch)`, `decode(batch)`): The production path for
  datacenter serving. The runtime assembles requests each iteration and calls the model
  once. The model executes all requests in a single GPU operation. `max_batch` informs
  the model of the maximum batch size, enabling pre-sizing of batch workspace.

- **Single-handle overloads** (`prefill(tokens, state)`, `decode(token, state)`): The
  natural path for edge devices (mobile, NPU), sequential workloads, and testing.
  Single-request-at-a-time IS the production workload on many edge targets (Qualcomm
  Snapdragon, Apple ANE, constrained MCUs). These overloads are not a lesser interface.

**Language bindings** that do not support method overloading SHOULD name the batch
overloads `prefill_batch` and `decode_batch` and document this convention in the binding.

### 5.4 Logit Properties

- All logits are pre-softmax scores (not probabilities).
- For valid (non-erroneous) inputs, logits MUST be finite (no NaN, no ±Inf).
- `prefill()` with a **single-position** `out_logits` buffer writes only the last token's logits.
  This is the default path — the runtime uses the last position to select the next token.
- `prefill()` with a **full-sequence** `out_logits` buffer writes logits for all positions.
  Use this for speculative decoding verification and prompt logprobs.
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
provided it is **semantically equivalent** — meaning it produces **byte-for-byte identical token IDs**
to the default tokenizer for all valid inputs. "Semantically equivalent" is verifiable, not an
honor system: the model MUST provide a normative test vector set as part of its conformance
documentation. This test set MUST include at minimum:
- Empty string
- ASCII alphanumeric sequences of varying lengths (1, 16, 512 characters)
- Unicode multibyte sequences (CJK, Arabic, emoji, combining characters)
- All special tokens by their canonical string representations (EOS, BOS, pad, unknown)
- A sequence containing every special token in order

A substituted tokenizer that produces different token IDs than the test vectors is non-conformant.
Vocabulary incompatibility produces corrupt output; the runtime bears full responsibility for any substitution.

---
