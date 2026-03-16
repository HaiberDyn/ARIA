<!-- ARIA module: MOD-C3-STATE.md — do not edit directly; edit this file in spec/modules/ -->
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
| `AttentionKV` + `context_shift` | Yes | Yes (model-controlled) | Strategy 1 only | No |
| `SSMState` | Yes (must serialize full state) | No | No | No |
| `Custom` | Yes (must serialize full state) | No | No | No |

**`context_shift` + paged layout interaction**: When a model declares `aria.core.context_shift`,
the model MAY overwrite any KV position at any time. Block-table-based paging (Strategy 2,
§4.5.2) cannot support arbitrary position overwrites across non-contiguous physical pages.
Therefore: a runtime that supports both `aria.core.context_shift` and `Layout::Paged` on the
same slot MUST use Strategy 1 (Virtual Contiguity — §4.5.2) for that slot. A runtime that
only supports Strategy 2 for paged layouts MUST return `SlotUnsupported` when a model
declares both `aria.core.context_shift` and `Layout::Paged` on an `AttentionKV` slot.

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

Uninitialized ──[cancel]──────────────────────────────────────────►┐
Ready ──────────────────────────────────────────────────────────►──┤
Exhausted ─────────────────────────────────────────────────────►───┤  [runtime releases]
Ready (in-flight call) ──[cancel, concurrent]── in-flight returns  │  ▼
                                                Err(Cancelled) ───►─┤  Released
                                                               [runtime] ▼
                                                                    Released
```

State definitions:
- **Uninitialized**: handle allocated, no prefill yet. `current_length() == 0`.
- **Ready**: at least one prefill completed. `decode()` may proceed. `current_length() > 0`.
- **Exhausted**: `current_length() == max_tokens()`. No further decode is allowed.
- **Released**: handle returned to runtime pool. MUST NOT be passed to any model method.

`cancel()` is a terminal transition from any non-Released state directly to Released.
The runtime MUST treat the handle as Released immediately after `cancel()` returns,
regardless of whether an in-flight call has yet returned `Err(Cancelled)`.

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
# Single-position out_logits buffer (default); intermediate chunk logits discarded by runtime
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
