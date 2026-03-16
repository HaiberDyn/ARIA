<!-- ARIA module: MOD-C7-OBLIGATIONS.md — do not edit directly; edit this file in spec/modules/ -->
## 10. Invariants

The following invariants MUST hold in any conforming implementation.

**Model invariants (implementor obligations):**
1. `declare_state()` is deterministic: identical `(max_batch, max_tokens)` inputs produce identical `StateSpec`.
2. Slot IDs within a `StateSpec` are unique.
3. `AttentionKV` slots are append-only: the model MUST NOT write to positions `< handle.current_length()`.
   The model reads `handle.current_length()` at the *start* of each call; it reflects tokens processed
   in all prior calls, not the current call. This invariant does not apply to models that declare the
   `aria.core.context_shift` capability — such models MAY overwrite prior KV positions as part of a
   context extension algorithm (e.g., sliding window, NTK scaling). Models using context shift MUST
   declare `aria.core.context_shift` in `ModelInfo.capabilities` (see `spec/EXTENSIONS.md`) and MUST
   document the modified write pattern.
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
11. The runtime SHOULD enforce operator-configured resource limits on `declare_state()`
    before attempting allocation. If the declared `StateSpec` would exceed configured limits,
    the runtime MUST return `OutOfMemory` rather than attempting the allocation. In the
    absence of operator configuration, runtimes SHOULD apply conservative defaults:
    no more than 1024 slots per `StateSpec`, and no single slot allocation exceeding
    available device memory. This prevents a malformed or adversarial model from
    exhausting allocator resources via `declare_state()`.

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
  `ILanguageModel` instance; verifier uses `prefill()` with a full-sequence `out_logits` buffer).
  The runtime allocates the verification buffer sized `[draft_len, vocab_size]` where `draft_len`
  is the number of speculative tokens produced by the draft model — a value the runtime determines
  from its speculative decoding policy, independent of the target model.
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
  and verifying with `prefill()` using a full-sequence `out_logits` buffer
- Use **any sampling strategy**: temperature, top-p, top-k, beam search, guided decoding
- Implement **streaming** by consuming `decode()` logits one token at a time
- Implement **LoRA / adapter management** without ARIA involvement (adapters are weight
  management, not state management)
- Use **any request queue** strategy: priority, fair, rate-limited, preemptive
- Implement **async execution** by wrapping ARIA's synchronous interface in an async executor

The only constraint: the runtime MUST allocate state as declared in `StateSpec`, pass handles
correctly, and propagate errors faithfully.

---
