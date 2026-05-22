# First Phase Checklist

Goal: reshape the repository to the planned Rust + Zig monorepo layout, then complete the first executable Rust foundation tasks.

## 0) Baseline and guardrails

- [x] Confirm current status (`git status --short --branch`) and keep existing files untouched unless explicitly migrated.
- [x] Add a root `.gitignore` for Rust, Zig, Python, and output artifacts.
- [x] Keep the current architecture docs (`pnn_implementation_plan.html`, `pnn_mono_repo_architecture.html`) as planning references.

## 1) Create monorepo folder structure

- [x] Create `rust/`, `zig/`, `data/`, `benchmarks/`, `.github/workflows/` at repo root.
- [x] Add root `NOTES.md` (cross-language diary placeholder).
- [ ] Add `data/README.md` describing shared CSV conventions.
- [ ] Add `benchmarks/README.md` describing shared JSON output schema and parity rule (`max |rust - zig| < 1e-4`).

## 2) Rust workspace scaffold (no algorithm yet)

- [x] Create `rust/Cargo.toml` workspace with members: `pnn-core`, `pnn-cli`, `pnn-py`.
- [x] Create crates:
  - [x] `rust/pnn-core` (library)
  - [x] `rust/pnn-cli` (binary)
  - [x] `rust/pnn-py` (PyO3 bindings placeholder)
- [x] In `rust/pnn-core/src/`, add stubs:
  - [x] `lib.rs`
  - [x] `types.rs`
  - [x] `knn.rs`
  - [x] `model.rs`
  - [x] `inference.rs`
  - [x] `predict.rs`
- [x] Wire module exports in `lib.rs` so the crate compiles.
- [x] Run `cargo check --workspace` from `rust/` and fix all compile issues.

## 3) Shared contracts before implementation depth

- [ ] Define CSV field expectations for train/test files (features + label conventions) in `data/README.md`.
- [ ] Define JSON output contract (required keys, value types, and example payload) in `benchmarks/README.md`.
- [ ] Ensure both contracts are language-agnostic and explicitly shared by Rust and Zig.

## 4) First real implementation task (Rust only)

- [ ] Implement `types.rs` with core data structures (`DataMatrix`, `Labels`, `ModelParams`, `PnnModel`).
- [ ] Implement initial `knn.rs` neighbor search API (can start simple, optimize later).
- [ ] Add `rust/pnn-core/tests/knn_toy.rs` with a 5-point hand-validated nearest-neighbor test.
- [ ] Run `cargo test -p pnn-core` and ensure toy kNN test passes.

## 5) Minimal CLI contract check

- [ ] Implement a minimal `pnn-cli` flow: read CSV paths + write JSON output path.
- [ ] Emit schema-valid JSON even if predictions are temporary placeholders.
- [ ] Verify end-to-end run on a tiny sample file in `data/`.

## 6) Parity harness bootstrap (pre-Zig)

- [ ] Create `benchmarks/compare.py` that loads two JSON outputs with the shared schema.
- [ ] Implement numeric comparison utility with tolerance target (`1e-4`).
- [ ] Add usage notes now; full Rust-vs-Zig comparison will be activated once Zig output exists.

## Exit criteria for first phase

- [ ] Repository layout matches the planned monorepo architecture.
- [ ] Rust workspace and crates compile cleanly.
- [ ] Shared CSV/JSON contracts are documented and stable.
- [ ] First algorithmic checkpoint is complete: Rust kNN + toy correctness test.
