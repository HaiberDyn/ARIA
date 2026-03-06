# ARIA Roadmap

This document tracks planned extensions to the ARIA specification.
The core spec (`spec/ARIA-0.6-draft.md`) contains only normative 0.6 content.
Future work is tracked here.

---

## ARIA 1.0 (Target: H2 2026)

Requires: 2+ independent 0.6 implementations, working group formation, multi-stakeholder review.

| Feature | Description | Issue/PR |
|---------|-------------|----------|
| `CrossAttentionKV` semantic tag | KV state for encoder-decoder cross-attention; enables Whisper, T5, and similar architectures | — |
| Encoder-decoder scope | Full encoder-decoder contract: separate encoder `prefill()`, cross-attention state lifecycle | — |
| LoRA / adapter interface | Per-handle adapter activation, multi-tenant adapter serving, adapter weight registration | — |
| `truncate()` operation | Per-position KV eviction and context shifting within an active handle (explicit API for what llama.cpp does today) | — |
| Multi-token prediction | Models with N-step prediction heads returning `List[Logits]` from `decode()` | — |
| `BatchProfile` interface | Optional explicit batch coordination for co-designed model-runtime pairs | — |
| CUDA graph capture hints | Warm-up / graph capture lifecycle hooks so models behave correctly during graph recording | — |
| Conformance tiers | `ARIA Core` (decoder-only, AttentionKV only) vs `ARIA Full` (all tag types) | — |

---

## ARIA 2.0 (Target: 2027)

Requires: ARIA 1.0 stable, expanded working group, paper/benchmarks published.

| Feature | Description |
|---------|-------------|
| Multimodal inputs | Embedding-tensor inputs alongside TokenIds; separate vision/audio encoder interface |
| Distributed inference | Tensor and pipeline parallelism contracts for multi-node serving |
| Training contract | Weight export, gradient-compatible state, fine-tuning hooks (in coordination with PyTorch, JAX teams) |
| Async protocol | Optional async/await binding profile for languages with native async support |

---

## Implementation Milestones

These milestones track the ecosystem, not the spec itself.

| Milestone | Status | Target |
|-----------|--------|--------|
| Reference Python binding | Pending | Phase 2 |
| SGLang ARIA adapter | Pending | Phase 2 |
| Conformance test suite (model tests) | Pending | Phase 2 |
| Conformance test suite (runtime tests) | Pending | Phase 2 |
| First ARIA-Portable model published | Pending | Phase 2 |
| Standards body working group submission | Pending | After 2 conforming implementations |
| Peer-reviewed paper (MLSys/OSDI) | Pending | After conformance test suite |

---

## Process for Moving Items to Spec

An item on this roadmap moves to the spec when:
1. A concrete proposal (Issue with full spec text and reference implementation) exists
2. Two independent implementations exist
3. Steering consensus (no unresolved objections in 14-day window)
4. A minor version bump is issued

See `GOVERNANCE.md §4` and `CONTRIBUTING.md §3`.

See also spec §15 (Future Extensions) for the normative list of planned additions.
