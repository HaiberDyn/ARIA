# ARIA 0.6.2-draft — Red Team Green Signals
> All positive signals from all 28 panelists. For per-panelist synthesis (green + red together), see RED-TEAM-SYNTHESIS.md.

---

## HARNESS / RUNTIME MAKERS

---

**Woojin Lee — vLLM — PagedAttention / Continuous Batching Lead**

- §4.5.2: Two conformant TensorView strategies (Virtual Contiguity + Block Table Access) map precisely to vLLM's architecture. Block Table Access was clearly designed with PagedAttention in mind. The physical block table model is understood.
- §6.1: AttentionKV + window_size eviction semantics are exactly what sliding window attention needs. The table is precise and actionable.
- §9 batch semantics: two-level `Result<List<Result<>>, E>` is production-correct. Outer Err = catastrophic only. Per-item failures don't abort the batch. This is the right design for continuous batching.

---

**Wei Zhu — SGLang — RadixAttention / Chunked Prefill Lead**

- §7.4: Multi-call prefill is clean. SGLang's chunked prefill maps directly — prefill on Ready handles extends the sequence. "Only new, unprocessed tokens" note is correct.
- §11.2: RadixAttention is runtime-internal, invisible to model. Correct — prefix sharing happens at the block level; the model never sees physical sharing.
- §6.1 prefix sharing column: AttentionKV (no window) = Yes. SGLang can share prefixes transparently behind the handle abstraction.

---

**Georgi Gerganov — llama.cpp — CPU Inference / GGUF / Quantization Lead**

- §4.2: `FloatCustom{e_bits, m_bits}` covers exotic quantization formats (iq2_xxs, iq4_nl) at the KV level. The note explicitly clarifying that GGUF block quants apply to weights not KV slots is thoughtful and prevents a common misunderstanding.
- §11.1: "use any hardware: CUDA, ROCm, Metal, CPU" — llama.cpp's multi-backend design is explicitly in scope.
- Appendix B llama.cpp: context shifting described correctly. `aria.core.context_shift` capability flag is the right mechanism.

---

**Nicolas Patry — TGI (HuggingFace) — Flash Attention / Production Serving Lead**

- §9 `ModelError.detail` sanitization: exactly what TGI needs for structured logging and multi-tenant error propagation. TGI surfaces errors to end users; unsanitized model strings are a live injection vector today.
- §14: 24-month critical fix window for previous major version. Production systems cannot upgrade immediately. This commitment matters.
- §9 error taxonomy table: "Raised by / Fatal / Retriable / Handle valid after?" maps directly to TGI's retry and circuit-breaker logic.

---

**Kai Zhang — LMDeploy — TurboMind / CN Production Scale Lead**

- §4.3 `preferred_page_tokens` hint: TurboMind uses fixed-size block tables. The hint mechanism is appropriate — TurboMind sets its own granularity.
- §10 Runtime Invariant 11: operator resource limits on `declare_state()`. At CN production scale, memory limits are hard constraints. This invariant is essential.

---

**Jeffrey Morgan — Ollama — Developer UX / Local Model Management Lead**

- §15: LoRA / adapter management explicitly out of scope with correct rationale ("weight management, not state management"). Ollama's modelfile adapter system sits cleanly outside ARIA.
- §13.6: Synchronous-first design. Ollama's HTTP handlers wrap ARIA's sync interface cleanly. No async model prescribed.

---

## MODEL MAKERS

---

**Sam Altman / Ilya Sutskever proxy — OpenAI — GPT-4/o Architecture Team**

- §13.1: "No fourth generation primitive." Minimalism is correct — adding methods for beam search or speculative decoding orchestration would couple the spec to specific strategies that are appropriately above ARIA.
- §8: KV dtype negotiation ordered preference list is well-designed. Expresses preference hierarchy while giving runtimes allocation flexibility.

---

**Dario Amodei proxy — Anthropic — Claude / Constitutional AI Team**

- §6.2.3 `extract()` "MUST NOT raise an error for malformed content — return None": defensive extraction is exactly right. Error on malformed XML mid-stream would break every streaming harness.
- §10 Model Invariant 5: "model MUST NOT allocate persistent per-request state outside StateHandle" — the critical multi-tenant isolation property. Essential for Constitutional AI's safety contract.
- §14.3: Patent non-assertion pledge is essential for adoption. Without it, this spec is dead on arrival for any serious implementor.

---

**Guillaume Lample / Hugo Touvron proxy — Meta AI — LLaMA Team**

- §4.4 `ModelParam("num_kv_heads", 8)`: excellent for GQA. LLaMA 3.1's `num_kv_heads=8` vs `num_q_heads=32` expressed cleanly and self-documentingly.
- §4.5.3: explicitly notes "GQA/MQA will differ from query head count." LLaMA popularized GQA and it is correctly handled.
- §10 Model Invariant 3: context_shift carve-out is appropriate for YaRN, NTK scaling, and sliding window implementations.

---

**Arthur Mensch proxy — Mistral AI — Sliding Window / MoE / Efficiency Lead**

- §6 `AttentionKV` with `window_size`: designed with Mistral's sliding window attention in mind. The semantic is precise — runtime MAY evict pages beyond window_size.
- §6.1 table: the row for `AttentionKV` + window correctly specifies "Yes (stale pages only)" for intra-handle eviction. Exactly right for Mixtral.

---

**Jinze Bao proxy — Alibaba/Qwen — Qwen3 Hybrid / MoE / CN Ecosystem Lead**

- §6 `SSMState`: correct semantics for Qwen3's hybrid Mamba+attention layers. Fixed size, fully overwritten, no paging. The list of examples ("Mamba, GDN, RWKV, etc.") shows awareness of the hybrid landscape.
- §6.2.5 `QwenAdapter`: the spec has a working adapter for Qwen's native format. The JSON-in-XML wrapper is correctly identified.
- §6.1: table correctly differentiates AttentionKV vs SSMState memory management for hybrid model deployment.

---

**Oriol Vinyals proxy — Google DeepMind — Gemma / Gemini / TPU Architecture Team**

- §13.7: "ARIA works with compiled graphs" is the key TPU compatibility statement for TPU/XLA. `declare_state()` triggering XLA graph compilation is a clean integration model.
- §2.2: multimodal inputs correctly deferred to 2.0. Appropriate scope restraint.
- §14.1: "minor bump: new optional semantic tag" allows adding `CrossAttentionKV` without breaking existing models. Clean.

---

**Igor Babuschkin proxy — xAI — Grok / MoE / Long Context / Real-Time Data Lead**

- §7.4 multi-call prefill: clean specification. For Grok's long context (128K+), chunked prefill is essential. The re-prefilling warning is correct.
- §15 known limitation: "max_tokens shared across all handles causes overallocation for short requests" — honest acknowledgment with a practical interim note.

---

**Nick Frosst proxy — Cohere — Command / Enterprise RAG / Tool Use Lead**

- §9 error taxonomy: "Fatal / Retriable / Handle valid after?" columns map directly to enterprise incident classification and SLA-differentiated retry policies.
- §10 Runtime Invariant 7: "MUST NOT silently suppress or retry errors without consumer's knowledge." Critical for enterprise audit trails.

---

**Li Jinyu proxy — 01.AI — Yi Series / Multilingual / Asian Market Lead**

- §4.2 `FloatCustom{e_bits, m_bits}`: covers CN chip ecosystem custom quantization formats not enumerated in standard enums.
- §10 Runtime Invariant 11: conservative defaults for constrained deployment targets are appropriate.

---

**Wenfeng Liang proxy — DeepSeek — MoE / Open Weights / CN AI Lab**

- §6 `SSMState`: DeepSeek-V3's hybrid MLA+MoE architecture requires heterogeneous state. The spec handles this correctly.
- §13.3: Custom tag aggregation for architectural confidentiality — important for commercial model deployment where internal architecture is proprietary IP.

---

## HARDWARE / INFERENCE SILICON

---

**Jensen Huang proxy — NVIDIA — H100/B200/GB200 / CUDA / TensorRT Lead**

- §4.5.1 memory coherence: "CUDA stream synchronization after call completes" — correct mechanism for NVIDIA's actual pipeline.
- §4.2: `Float8E4M3`, `Float8E5M2`, `Float4E2M1` explicitly named — H100 FP8 tensor cores and B200 NVFP4 are both covered.
- §11.1: CUDA graph capture listed as model-internal freedom — NVIDIA's entire inference performance story depends on this being model-internal.

---

**Victor Peng proxy — AMD — MI300X / ROCm / HIP / CDNA Lead**

- §4.5.1: "CUDA (NVIDIA, AMD ROCm)" — ROCm correctly grouped for stream synchronization.
- §11.1: ROCm explicitly mentioned in hardware freedom list.

---

**Jay Huang proxy — Huawei Ascend — Ascend 910B/910C / CANN Stack Lead**

- §4.2 `DType::Custom(string)`: essential for Ascend's INT8 variants with non-standard scaling.
- §13.7: compiled graph approach fits Ascend's CANN graph compiler. `declare_state()` triggering graph compilation is the correct integration pattern.

---

**Norm Jouppi proxy — Google — TPUv5 / Trillium / XLA / JAX Lead**

- §13.7: ARIA's compatibility with compiled graphs is the key TPU compatibility statement.
- §2.2: multimodal and distributed inference correctly deferred. Appropriate scope.

---

**Raja Koduri proxy — Intel — Gaudi 3 / SynapseAI / AMX / Datacenter Lead**

- Appendix B Intel Gaudi section: accurate and specific. SynapseAI compilation at `declare_state()` time is correctly modeled.
- §4.2 `FloatCustom`: explicitly covers "Intel FP8 E4M3FNUZ bias variant" in Appendix B. Good precision.
- §11.1: compiled graph freedom — Gaudi's runtime dispatch of pre-compiled kernels fits cleanly.

---

**Johny Srouji proxy — Apple — M4/M4 Max / Metal / CoreML / ANE Lead**

- Appendix B Apple Silicon: thorough and accurate. CoreML Swift/ObjC bridge, Metal compute shaders, unified memory buffer allocation — correctly described.
- §4.5.1: "Metal command buffer completion event" is correct.
- §11.1: ANE can be used as a model-internal optimization. ARIA doesn't need to know.

---

**Ziad Asghar proxy — Qualcomm — Snapdragon X Elite / QNN / Hexagon NPU / Edge Lead**

- Appendix B Qualcomm QNN: accurate. QNN graph execution, state memory via QNN memory API, FloatCustom for Qualcomm formats.
- §4.5.1: QNN completion callbacks covered under NPU entry.
- §13.7: compiled graph model fits QNN's ahead-of-time compilation.

---

## STANDARDS / COMPANION INTELLIGENCE

---

**HaiberDyn Agent — HCS Expert — Companion Intelligence Architecture**

*(Green signals from ARIA toward HCS — what ARIA gets right for companion deployments)*

- §6.2 MPA: directly solves a companion deployment problem. The user selects their model (Qwen, Claude, future model) — the MPA ensures the HCS harness extracts tool invocations consistently regardless of model family. This supports HCS's user-choice principle without requiring harness changes per model.
- §10 Model Invariant 5: "model MUST NOT allocate persistent per-request state outside StateHandle" — a companion privacy property. Models that leak per-request state create cross-session data exposure vectors. ARIA's strict encapsulation is a companion sovereignty guarantee.
- §13.3: Custom tag aggregation for architectural confidentiality enables proprietary companion model deployment while remaining ARIA-conformant. Sovereign companion models can protect their architecture.

---

## SECURITY

---

**Becky Weiss proxy — AWS — Cloud Security / Multi-Tenant Inference Lead**

- §9 `ModelError.detail` sanitization: correct, well-specified, and actionable. "Printable ASCII, no control chars, max 512 bytes" prevents log injection and header injection.
- §6.2.2 `JsonValue` depth/size limits: prevents JSON bomb attacks via adversarial tool parameters.
- §10 Runtime Invariant 11: resource limits on `declare_state()` before allocation. Prevents adversarial model exhausting the allocator.

---

**Mark Russinovich proxy — Azure — Enterprise Security / Compliance / Data Residency Lead**

- §10 Runtime Invariant 7: "MUST NOT silently suppress or retry errors without consumer's knowledge." Essential for enterprise audit trails and incident response.
- §9 error taxonomy: "Raised by / Fatal / Retriable" maps directly to enterprise incident classification, SLA tiers, and retry policy automation.
- §3.1 immutability at boundaries: "consumer MUST NOT write into model-owned tensor memory" prevents a class of harness bugs that corrupt model state across tenant boundaries.

---

**Ivan Krstić proxy — Apple — Device Security / Secure Enclave / Private Compute Lead**

- §10 Model Invariant 5: "model MUST NOT allocate persistent per-request state outside StateHandle" — critical for Private Compute isolation. A model caching user data outside ARIA-managed memory could persist across Secure Enclave boundaries.
- §3.1 encapsulation: "StateHandle is opaque. Consumer has no access to slot memory." Correct for private computation — the harness must not be able to inspect model activations.

---

**Asaf Shen proxy — Qualcomm — Edge Security / TrustZone / TEE / Model IP Protection Lead**

- §13.3: Custom tag aggregation for architectural confidentiality enables proprietary model IP protection on edge devices deployed in adversarial physical environments.
- §14.3: Apache 2.0 reference implementations — critical for commercial edge deployments where GPL creates legal risk.

---

## SUMMARY — What the Spec Gets Right

Across 28 panelists, strong consensus on:

1. **Semantic tag design** — AttentionKV/SSMState distinction is the right abstraction. Window-size eviction semantics are precise.
2. **Two-level batch Result** — outer catastrophic / inner per-item failure semantics are production-correct (despite ABI inconsistency in §16).
3. **Compiled graph compatibility** — `declare_state()` as graph compilation trigger works for XLA, SynapseAI, QNN, CANN, CoreML.
4. **Innovation freedom invariants** (§11) — neither model nor runtime is over-constrained. Hardware dispatch, paging, sampling remain fully internal.
5. **Patent non-assertion pledge + CC BY-SA 4.0** — essential IP posture for ecosystem adoption.
6. **ModelError.detail sanitization** — correct, actionable, prevents live injection vectors.
7. **Model Invariant 5** (no out-of-handle state) — the critical multi-tenant isolation and privacy property.
8. **KV dtype negotiation** (§8) — ordered preference mechanism is well-designed.
9. **Multi-call prefill** (§7.4) — clean chunked prefill and multi-turn extension semantics.
10. **FloatCustom{e_bits, m_bits}** — covers exotic quantization formats (CN chips, quantization formats, non-standard scaling) without spec revision.
