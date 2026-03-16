# Contributing to ARIA

Welcome. ARIA is a community spec — its value comes from broad participation by model
authors, runtime maintainers, hardware vendors, and operators. This document explains
how to contribute effectively and the intellectual property (IP) rules that govern it.

**Status**: ARIA 0.7-draft

---

## 1. Intellectual Property Policy

ARIA follows a strict "Open Spec / Open Code" policy to prevent fragmentation and
patent aggression.

### 1.1 Licenses
*   **Specification Text**: **CC BY-SA 4.0** (Creative Commons Attribution-ShareAlike 4.0).
    *   *Why?* You are free to share and adapt the text, but **derivations of the spec must remain open source** under the same terms. This prevents corporations from creating proprietary forks of the standard document itself.
*   **Reference Code & Headers**: **Apache 2.0**.
    *   *Why?* Permissive use (commercial/proprietary runtimes) is allowed, but **patent grants are automatic**. If you contribute code, you grant a patent license. If you sue users for patent infringement, your license terminates.

### 1.2 Patent Non-Assertion Pledge
By contributing to this repository, you agree to the **ARIA Patent Non-Assertion Pledge**:

> "I promise not to assert any patent claims against any implementation of the ARIA specification, insofar as those claims would be infringed by a compliant implementation of the ARIA specification."

This pledge is binding on you and your organization.

### 1.3 Developer Certificate of Origin (DCO)
To legally enforce the above, every contribution MUST be signed off.
We use the standard DCO 1.1.

**By adding a `Signed-off-by` line to your commit, you certify that:**
1.  You have the right to submit this contribution.
2.  You agree to the licenses (CC BY-SA 4.0 for text, Apache 2.0 for code).
3.  You agree to the Patent Non-Assertion Pledge in §1.2.

**How to sign off:**
```bash
git commit -s -m "fix: update kv cache layout"
```
Commits without a sign-off will be rejected by CI.

---

## 2. Types of Contributions

| Type | How to contribute | Decision process |
|------|------------------|-----------------|
| Bug report (spec ambiguity, typo) | GitHub Issue | Editor fixes within 7 days |
| Clarification (spec is unclear) | GitHub Issue → PR | Editor review; 7-day objection window |
| New semantic tag | Issue with `[TAG PROPOSAL]` | Tag Registry process (§4.2) |
| Non-breaking spec addition | PR with proposal document | 14-day review; 2 implementation requirement |
| Breaking change proposal | Issue + design doc | Steering Committee vote |
| Language binding | PR to `bindings/` directory | Editor review; no implementation requirement |
| Conformance test | PR to `tests/conformance/` | Editor review; encouraged for all implementers |
| Reference implementation | PR or link as issue | Editor review |

---

## 3. Spec Change Format

Spec changes SHOULD use RFC 2119 language (MUST, SHOULD, MAY, MUST NOT, SHOULD NOT).

For new sections or significant additions, follow this structure:
1. **Problem statement**: What fails without this change?
2. **Proposed spec text**: Exact normative language.
3. **Reference implementation**: Pseudocode or link showing how model/runtime implements it.
4. **Compatibility impact**: Does this break existing conforming implementations?

---

## 4. Community and Semantic Tag Registry

### 4.1 Discussion and Proposals
For open-ended questions, architectural discussions, or early-stage proposals, please use [GitHub Discussions](https://github.com/HaiberDyn/ARIA/discussions).

For concrete bug reports or feature requests that are ready for implementation, please use [GitHub Issues](https://github.com/HaiberDyn/ARIA/issues).

### 4.2 Semantic Tag Registry

ARIA uses an IANA-style registry for semantic tags. New tags can be registered
**without a full spec revision**, enabling rapid adoption of new model architectures.

### 4.3 Registry Status Levels

| Status | Meaning | Where defined |
|--------|---------|---------------|
| **Standard** | Part of the core ARIA spec; all conforming runtimes must support | `spec/ARIA-x.y-draft.md` §6 |
| **Registered** | Community-reviewed; runtimes may advertise support | `spec/EXTENSIONS.md` |
| **Experimental** | Vendor-specific; use `Custom` with extension ID | Inline in model code |

### 4.4 How to Register a Tag

New Semantic Tags or Capability strings follow the [Extension Proposal Process](spec/EXTENSIONS.md#process).

To propose a new tag, create a GitHub Issue or Discussion titled `[TAG PROPOSAL] YourTagName` using the template below:

```
## Tag Name
`YourTagName { field1: type, field2: type }`

## Architecture
Which model architecture(s) use this state type?

## State Semantics
- Is state append-only? Fully overwritten? Partially updated?
- Does size grow with sequence length?
- Can state be evicted or must it be preserved exactly?

## Runtime Management Obligations
What MUST a runtime do to manage this state type correctly?
(Model this on §6 of the spec — AttentionKV and SSMState entries)

## Custom Fallback
How should a runtime that does not recognize this tag handle it?
(Usually: allocate as declared, preserve exactly, treat as Custom)

## Reference Model
Link to or description of a model that uses this tag.

## Reference Implementation
Pseudocode showing how the model reads/writes this state.
```

The tag is added to `spec/EXTENSIONS.md` once:
- The proposal is complete (all fields above answered)
- No unresolved technical objections after 7 days
- At least one reference implementation exists

### 4.5 Promotion to Standard

A Registered tag may be promoted to Standard in a minor spec revision when:
- Two independent implementations support it
- The tag description is stable (no changes in 60 days)
- Steering consensus

---

## 5. Language Bindings

ARIA's core spec is language-agnostic pseudocode. Bindings translate the spec into
specific languages. Bindings live in `bindings/<language>/`.

Each binding MUST document:
- The concrete types used for `TensorView` (e.g., `torch.Tensor`, `*mut f16`)
- Lifetime management (how views are scoped to calls)
- Memory coherence mechanism for the target platform
- Any language-specific adaptations (e.g., Python async wrapper)

Bindings are community-maintained. A binding that passes the conformance test suite
(see `CONFORMANCE.md`) may be marked as "conformance-tested."

---

## 6. Conformance Tests

Conformance tests live in `tests/conformance/`. Adding tests for new spec behaviors is
one of the highest-value contributions. See `CONFORMANCE.md` for the test categories
and how to run them.

---

## 7. Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/)
Code of Conduct. Be respectful. Critique spec text and technical approaches, not people.
