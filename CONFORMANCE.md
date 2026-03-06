# ARIA Conformance

**Spec version**: 0.5-draft  
**Status**: Checklist (conformance test suite implementation pending — tracked in project roadmap)

---

## Overview

ARIA defines two conformance roles: **model** and **runtime**. Both must pass their
respective requirements to claim ARIA 0.5-draft conformance. Additionally, ARIA distinguishes
between "conformant" and "portable" — an important distinction for adoption decisions.

---

## Conformance Levels

### ARIA-Conformant

A model or runtime that implements the full ARIA 0.5-draft contract, including support for
`Custom` semantic tags and dtypes. Conformant implementations interoperate wherever
the two sides agree on Custom tag descriptions. Not guaranteed to be portable across
arbitrary runtimes/models.

### ARIA-Portable

A stricter subset: a model or runtime that uses **zero Custom semantic tags** (all state
uses `AttentionKV`, `SSMState`, or future Standard tags). An ARIA-Portable model can be
loaded by any ARIA-Portable runtime without out-of-band tag agreements.

**The Custom portability rule**: Models SHOULD minimize Custom usage. Each minor ARIA
version will promote common Custom patterns to Standard tags, reducing Custom usage
across the ecosystem. A model is encouraged to document which of its Custom tags it
plans to propose for registration (see CONTRIBUTING.md §4).

---

## Model Conformance Requirements

A conforming ARIA 0.5-draft model MUST:

### Interface Completeness
- [ ] Implement all methods: `model_info()`, `declare_state()`, `prefill()` (single + batch), `decode()` (single + batch)
- [ ] `model_info()` returns a valid `ModelInfo` with all required fields populated, including `default_tokenizer`
- [ ] `declare_state()` returns a `StateSpec` with at least one `StateSlot`
- [ ] `declare_state()` is deterministic: identical inputs produce identical `StateSpec`
- [ ] `prefill()` accepts `LogitsMode.LastPosition` and returns a List of length 1
- [ ] `prefill()` accepts `LogitsMode.AllPositions` and returns a List of length `len(tokens)`
- [ ] `decode()` returns exactly one `Logits`

### State Handling
- [ ] Never creates, copies, or destroys `StateHandle` objects
- [ ] Accesses state only through `handle.slot(id)` TensorView accessors
- [ ] Does not store `TensorView` references beyond the call that produced them
- [ ] Writes `AttentionKV` append-only (no overwriting positions < `current_length()`,
      except for explicit context-shifting implementations that document this behavior) (per §6, §10 Inv.3)
- [ ] Fully overwrites `SSMState` on every `prefill()` and `decode()` call (per §6, §10 Inv.4)
- [ ] Handles multi-call prefill: correctly extends KV state on `Ready` handles
- [ ] Reads `handle.current_length()` to determine prefill offset for multi-call scenarios

### ITokenizer
- [ ] `ModelInfo.default_tokenizer` is populated with a conformant `ITokenizer`
- [ ] `encode()` and `encode(batch)` produce identical token IDs for identical input
- [ ] `decode()` and `decode(batch)` are inverse of `encode()` for all in-vocabulary tokens
- [ ] `vocab_size()` matches `ModelInfo.vocab_size`
- [ ] `special_tokens()` includes at minimum an EOS token

### Output Correctness
- [ ] Returns pre-softmax logits (not probabilities)
- [ ] Returns finite logits (no NaN, no ±Inf) for valid inputs
- [ ] Returns `EmptyInput` if `prefill()` is called with an empty token list

### KV Dtype Negotiation
- [ ] `model_info().supported_kv_dtypes` is non-empty and ordered by preference
- [ ] Model reads the actual dtype from `TensorView.dtype` (not hardcoding the declared dtype)
- [ ] Model handles all dtypes it declares in `supported_kv_dtypes`

### Thread Safety
- [ ] `model_info()` and `declare_state()` are thread-safe
- [ ] Concurrent calls on different `StateHandle`s produce correct results

---

## Runtime Conformance Requirements

A conforming ARIA 0.5-draft runtime MUST:

### Call Protocol
- [ ] Call `model_info()` before `declare_state()`
- [ ] Call `declare_state()` before any `prefill()` or `decode()`
- [ ] Call `prefill()` on a handle before the first `decode()` on that handle
- [ ] Serialize concurrent access to a single `StateHandle`

### State Allocation
- [ ] Allocate all slots declared in `StateSpec` before passing a handle to the model
- [ ] Allocate each slot with the declared shape, dtype (or negotiated dtype per §8), and layout
- [ ] For `Layout::Paged` slots: use a page size compatible with the model's `preferred_page_tokens`
      (or return `SlotUnsupported` if incompatible)
- [ ] Reject with `SlotUnsupported` for `Layout::Custom` strings it does not recognize

### StateHandle Accessors
- [ ] `handle.id()` returns a stable uint64 unique to the handle's lifetime
- [ ] `handle.current_length()` returns the correct count of processed tokens
- [ ] `handle.max_tokens()` returns the `max_tokens` from `declare_state()`
- [ ] `handle.tokens_remaining()` returns `max_tokens() - current_length()`
- [ ] `handle.slot(id)` returns a valid `TensorView` for all declared slot IDs
- [ ] `handle.slot(id)` with an unknown ID raises or returns an error

### Memory Management (by SemanticTag) — per §6
- [ ] `AttentionKV`: MAY page (with `Layout::Paged`); MAY prefix-share; MUST NOT evict
      pages within an active non-preempted handle (except stale window pages for sliding
      window attention where `window_size` is set) (per §6, §10 Runtime Inv.5)
- [ ] `SSMState`: MUST NOT page or prefix-share; MUST preserve exact values (per §6)
- [ ] `Custom`: MUST allocate exactly as declared; MUST preserve exact values; no paging (per §6)

### Memory Coherence
- [ ] Ensure all model writes via TensorView are visible after the call returns
      (mechanism is implementation-defined per §4.5.1)

### Error Handling
- [ ] Propagate `ARIAError` variants from model calls to callers
- [ ] Return `VersionIncompatible` if the model's `aria_version.major` is not supported

---

## KV Dtype Negotiation Conformance

A conforming runtime MUST implement the KV dtype negotiation protocol (§8):

- [ ] Reads `model_info().supported_kv_dtypes` (ordered by model preference)
- [ ] Selects a dtype it can allocate from the list, or returns `NoDtypeCompatible` if none match
- [ ] Allocates `AttentionKV` slots with the selected dtype
- [ ] Exposes the selected dtype via `TensorView.dtype`

---

## Custom Usage Tracking

To achieve **ARIA-Portable** status, a model MUST:

- [ ] Declare zero `SemanticTag::Custom` state slots in its `StateSpec`
- [ ] Use only `DType` values from the standard enumerated list (no `DType::Custom`)
- [ ] Use only `Layout::Contiguous` or `Layout::Paged` (no `Layout::Custom`)

Any use of Custom variants moves the model from ARIA-Portable to ARIA-Conformant.

---

## Conformance Test Suite

A machine-verifiable conformance test suite is planned for Phase 2 of ARIA development.
The test suite will provide:

- **Model tests**: A reference runtime that verifies a candidate model against the
  requirements above. Written in Python (primary) and C (for native models).
- **Runtime tests**: A reference model that exercises all runtime requirements. The
  reference model declares all semantic tag types and verifies correct runtime behavior.
- **Interoperability tests**: Reference model + reference runtime running end-to-end,
  verifying logit correctness against a known-good implementation.

Until the test suite is available, implementers should use this document as a manual
checklist and publish their conformance attestation in `PARTICIPANTS.md` when created.

---

## Claiming Conformance

To publicly claim ARIA 0.5-draft conformance for your model or runtime:

1. Complete the applicable checklist above (all items checked)
2. Open a GitHub Issue tagged `[CONFORMANCE CLAIM]` with:
   - Your model/runtime name and version
   - Link to your implementation
   - Which checklist items you verified and how
   - Whether you claim ARIA-Conformant or ARIA-Portable status
3. Editors review within 14 days; listed in `PARTICIPANTS.md` on acceptance

Self-attestation is accepted in the absence of an automated test suite. False claims
are a violation of community norms and will be publicly corrected.
