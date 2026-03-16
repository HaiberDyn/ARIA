<!-- ARIA module: MOD-C6-ERRORS.md — do not edit directly; edit this file in spec/modules/ -->
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

    // ── Cancellation ──────────────────────────────────────────────────────
    // An in-flight prefill() or decode() was aborted by a cancel() call on this handle.
    // Returned by the aborted call — NOT by cancel() itself (cancel() returns Ok(void)).
    // The handle is in Released state. Do not reuse it.
    Cancelled { handle_id: uint64 },

    // ── Model errors ──────────────────────────────────────────────────────
    // Non-recoverable internal model failure (e.g., kernel crash, NaN explosion)
    ModelError { code: uint32, detail: string, fatal: bool },

    // ARIA version mismatch between model and runtime
    VersionIncompatible { model_version: (uint32, uint32), runtime_supports: (uint32, uint32) },
}
```

**Model-controlled string sanitization**: All model-controlled strings in `ARIAError` variants
MUST be treated as untrusted input by the runtime. Before including any such string in a
consumer-facing error message, log entry, or API response, the runtime MUST sanitize it:
restrict to printable ASCII, strip control characters (including newlines), and bound
length to 512 bytes. Unsanitized propagation is a conformance violation.

Model-controlled strings in this specification include:
- `ModelError.detail` — primary error description from the model
- `SlotUnsupported.reason` — model-supplied explanation for unsupported slot
- `SlotUnsupported.slot_id` — model-declared slot identifier
- `NoDtypeCompatible.slot_id` — model-declared slot identifier
- `DtypeMismatch.slot_id` — model-declared slot identifier

**Rationale**: Slot identifiers are declared in `StateSpec` at `declare_state()` time. An
adversarial model could embed injection payloads (newlines, ANSI escape sequences, HTML/JSON
fragments) in slot ID strings that later appear in log output or error responses. The sanitization
requirement applies uniformly to all model-controlled strings regardless of which error variant
carries them.

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
| `Cancelled` | Model | No | No (handle Released) | No |
| `ModelError` (fatal) | Model | Yes | No | No |
| `ModelError` (non-fatal) | Model | No | Yes | Implementation-defined |
| `VersionIncompatible` | Runtime | Yes | No | N/A |

The runtime MUST propagate all errors to the consumer. Silent fallback is a spec violation.

**Batch error semantics**: Batch calls use a two-level result to distinguish per-item errors
from catastrophic batch failure:

- **Outer `Ok(List<Result<>>)`**: The batch executed. Each inner `Result` carries the outcome
  for one request, in input order. Successful requests return `Ok(void)`; failed requests
  return `Err(ARIAError)`. The runtime continues processing all requests in the batch
  regardless of individual failures — partial success is the expected production behavior.
- **Outer `Err(ARIAError)`**: Catastrophic failure that prevented the batch from executing
  at all. This MUST only occur for `ModelError{fatal: true}` or an equivalent unrecoverable
  model-level failure. Outer `Err` is NOT returned for per-request errors such as
  `SequenceTooLong`, `NotPrefilled`, or `TokenOutOfRange`.

**Handle state after per-item error**: A handle whose inner `Result` is `Err` MUST be
treated as suspect by the runtime. The runtime SHOULD destroy and recreate it before reuse.
Handles with `Ok` results in the same batch are unaffected and remain valid.

---
