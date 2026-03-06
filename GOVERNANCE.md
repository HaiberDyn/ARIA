# ARIA Governance

**Version**: 0.1 (Draft — Pre-Working-Group)  
**Status**: Single-author draft. Seeking co-authors and working group formation.

---

## 1. Current Status

ARIA 0.4 is currently an initial specification authored by HaiberDyn. It has not yet been
adopted by a standards body or working group. This governance document describes the
intended multi-stakeholder structure and how to participate.

We are actively seeking co-authors, early implementers, and working group participants.
See [§6 How to Join](#6-how-to-join).

---

## 2. Goals

1. **Multi-stakeholder**: No single company controls the spec. Decisions require consensus
   across model authors, runtime maintainers, and hardware vendors.
2. **Lightweight**: Minimize process overhead. Proposals should move from idea to
   implementation in weeks, not years.
3. **Implementation-first**: No extension is merged into the spec without at least two
   independent implementations.
4. **Stable**: Backward compatibility within a major version is a hard requirement.

---

## 3. Roles

### 3.1 Editors

Editors are responsible for maintaining the specification text, merging approved proposals,
and resolving editorial disputes. The editor list is maintained in this file.

**Current editors**:
- HaiberDyn (initial author)

**Becoming an editor**: Any participant with 3+ accepted proposals may be nominated as
an editor by any existing editor. Acceptance requires no objections from Steering members
within 14 days.

### 3.2 Steering Committee

The Steering Committee makes binding decisions on: major version planning, working group
charter changes, editor appointments, and standards body submissions. Decisions are made
by lazy consensus (silence = consent; 7-day objection window).

**Current steering members**:
- HaiberDyn (founding member)

**Target composition**: 5–7 members representing at minimum: one runtime maintainer,
one model author (major model lab), one hardware vendor, and one cloud operator.

### 3.3 Contributors

Anyone who submits a proposal (via GitHub Issues or Pull Requests) is a contributor.
No formal membership required. Contributors agree to the DCO (see `IP-POLICY.md`).

### 3.4 Observers

Organizations that have indicated interest but not yet made contributions. No obligations;
listed in `PARTICIPANTS.md` when that file is created.

---

## 4. Decision Process

### 4.1 Spec Changes (Non-Breaking)

1. Open a GitHub Issue describing the problem and proposed change.
2. Discussion period: minimum 14 days, or until two independent implementations exist.
3. Editor merges if no unresolved objections from Steering members.
4. Changes are tagged as `minor` (no conformance break) or `patch` (editorial only).

### 4.2 New Semantic Tags

New semantic tags can be registered without a full spec revision:

1. Open an Issue with tag: `[TAG PROPOSAL] TagName{...}`.
2. Provide: (a) reference model that uses this state type, (b) description of runtime
   management obligations (analogous to §6 in the spec), (c) any Opaque fallback behavior.
3. A registered tag becomes part of the **Semantic Tag Registry** (see `CONTRIBUTING.md`).
4. Tags in the registry are `Registered` status; tags merged into the core spec are
   `Standard` status. Runtimes that implement `Registered` tags are advertising enhanced
   support; it is not required for conformance.

### 4.3 Breaking Changes (Major Version)

1. Requires explicit Steering Committee approval.
2. A migration guide and backward compatibility bridge MUST be provided.
3. The previous major version receives security and critical bug fixes for 24 months.

---

## 5. Intended Standards Body Submission

ARIA's long-term home is a neutral standards body. Current candidates under evaluation:

| Body | Fit | Notes |
|------|-----|-------|
| **LF AI & Data Foundation** | High | Neutral home for ML infrastructure; existing ML model serving working groups |
| **MLCommons** | Medium | Strong inference benchmarking track record; broader scope needed |
| **WASI Working Group** | Medium | ARIA as a WASI-NN Stateful Inference Profile is architecturally natural |
| **W3C** | Low | Web-focus; not the right home for systems-level ML serving |

The current plan is to approach LF AI & Data with a working group proposal once ARIA has:
- 2+ independent framework implementations
- 1+ model author publishing an ARIA-conformant model
- A passing conformance test suite (see `CONFORMANCE.md`)

---

## 6. How to Join

**As a contributor**: Open a GitHub Issue or Pull Request. Read `CONTRIBUTING.md`.

**As a Steering member**: Email the current editors expressing interest, your organization,
and your use case. Steering membership requires active participation (at minimum, reviewing
major proposals) and IP commitment (see `IP-POLICY.md`).

**As a co-author on the initial spec**: We are actively seeking co-authors from framework
and model author communities. Contact: open a GitHub Issue tagged `[CO-AUTHOR]`.

---

## 7. Amendments to This Document

This governance document may be amended by Steering consensus following the same process
as non-breaking spec changes (§4.1), with a minimum 21-day discussion period.
