# ARIA Reading Guide

The ARIA spec is split into modules. Read them in any order — all content is in the assembled `spec/ARIA-0.7-draft.md`. This guide suggests where to start based on your role.

## If you are implementing a model (ILanguageModel)

Start here:
1. ARIA-MASTER — motivation, scope, what ARIA is and isn't
2. MOD-C1-TYPES — the type system you'll use throughout
3. MOD-C2-CONTRACT — the interface you must implement
4. MOD-C3-STATE — semantic tags (which to declare) + StateHandle (how to access state)
5. MOD-C4-MPA — Model Protocol Adapter (if your model emits tool calls)
6. MOD-C5-NEGOTIATION — how KV dtype negotiation works from your side
7. MOD-C6-ERRORS — errors you must raise
8. MOD-C7-OBLIGATIONS — §10 Model invariants + §11.1 Model freedom
9. MOD-C8-ABI — how to export your model as a shared library

## If you are implementing a runtime / harness

Start here:
1. ARIA-MASTER — motivation, scope, what ARIA is and isn't
2. MOD-C1-TYPES — the type system you'll use throughout
3. MOD-C2-CONTRACT — the interface you must call correctly
4. MOD-C3-STATE — semantic tags (what each means for memory management) + StateHandle (what accessors you must provide)
5. MOD-C4-MPA — Model Protocol Adapter (if you want tool-call extraction)
6. MOD-C5-NEGOTIATION — how to select and allocate KV dtypes
7. MOD-C6-ERRORS — errors you must propagate
8. MOD-C7-OBLIGATIONS — §10 Runtime invariants + §11.2 Runtime freedom
9. MOD-C8-ABI — how to load models via dlopen

## Reference

- ARIA-TAIL — design rationale, versioning, future extensions, framework compatibility notes, reference implementation
