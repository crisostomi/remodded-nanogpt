# remodded-nanogpt

A modular **feature-search framework** for `modded-nanogpt`. Record-winning
changes are represented as composable **features**, activated through config and
compiled into a *static* training script — so the hot path stays
`torch.compile(..., dynamic=False, fullgraph=True)`-friendly with no runtime
feature branching.

```
feature YAML  ->  build context  ->  generated static train script
              ->  torchrun        ->  local manifest/logs  ->  (optional) Flywheel
```

GitHub is the source of truth for code and configs. Every run is reproducible
from GitHub + local artifacts; Flywheel is only a mirror for logging and
comparison.

> **Layout note:** the spec's `modded_nanogpt.*` package was adapted to this
> repo's `src/nano/` uv template, so CLI entry points are `python -m nano.*`.
> Tests live under `src/tests/` (the template's `testpaths`).

---

## 🚀 Installation

```sh
uv sync --extra test
```

The vendored baseline (`baseline/train_gpt.py` + `triton_kernels.py`, pinned in
`baseline/SOURCE_SHA.txt`) is the *current record*; the codegen template is
derived from it by `baseline/build_template.py`.

---

## 🧩 Concepts

- **Feature** — a unit of change that declares every surface it touches (model
  params/buffers, forward graph, optimizer table, schedule, data, distributed
  broadcasts, ...) via a `FeatureSpec`, and `apply`s its effects to a
  `BuildContext`. See `src/nano/features/`.
- **Feature set / preset** — a named collection of features (+ schedule/model
  overrides, tracking metadata) in `configs/feature_sets/*.yaml`. A set may
  extend a `base` preset; the final enabled set is `base ∪ enable − disable`.
- **A PR is *not* a feature.** A PR is a *collection* of features (e.g. PR #299
  = sign-trick + 15× vocab + 192-dim + sliced injection), each independently
  searchable.

Eight features are **template-toggleable** today (their model construction,
optimizer table *and* hot forward path are guarded): `sparse_attention_gate`,
`xsa`, `xsa_lowering_rewrite`, `bigram_sign_trick`, `bigram_dim_192`,
`bigram_vocab_15x`, `residual_slice_bigram_injection`, `sparse_bigram_comms`.
The rest are structural (always-on) for the MVP; the builder refuses to disable a
structural feature rather than emit a broken script.

---

## 🛠️ CLI

### Build a static train script

```sh
python -m nano.builder.codegen --preset current_record --out generated/train_current_record.py
```

Writes the script plus `*.features.yaml` and `*.manifest.json` sidecars and
copies `triton_kernels.py` alongside it.

### Build from a preset with overrides

```sh
python -m nano.builder.codegen \
  --preset current_record \
  --disable sparse_attention_gate \
  --enable xsa_lowering_rewrite \
  --out generated/train_current_record_minus_sag.py
```

### Run a combo and write a local manifest

```sh
python -m nano.search.run_combo \
  --preset current_record \
  --disable sparse_attention_gate \
  --enable xsa_lowering_rewrite \
  --run-name current_minus_sag_xsa_rewrite_s1385_seed0 \
  --nproc-per-node 8
```

`--dry-run` builds, validates and writes all artifacts (`manifest.json`,
`features.yaml`, `train_generated.py`) without launching `torchrun`. Each run
lands in `experiments/runs/<run_id>/`.

### Upload an existing run

```sh
python -m nano.runtime.flywheel --manifest experiments/runs/<run_id>/manifest.json --backend noop
```

Backends: `noop` (default), `local` (JSONL mirror), `flywheel` (stub until
creds/docs land). A failed upload never marks the training run failed.

### Compare runs

```sh
python -m nano.search.analyze_results
```

---

## 🧪 Running Tests

```sh
uv run pytest src/tests
```

Covers dependency/conflict/optimizer-consistency validation, build-context
assembly, codegen (render compiles; toggles change the right code), manifest
generation and the log parser.

---

## 📂 Structure

```
baseline/            vendored record train_gpt.py + triton_kernels.py + build_template.py
configs/feature_sets/ current_record.yaml, pr_299.yaml, pr_264.yaml, pr_259.yaml, pr_317.yaml, ...
generated/           generated train scripts (gitignored)
experiments/runs/    per-run manifest/logs/artifacts (gitignored)
src/nano/
  config/            feature-set schema, loading, preset resolution
  features/          FeatureSpec + registry + the feature catalog
  builder/           BuildContext, validation, render, codegen, templates/
  runtime/           manifest, log parser, tracking backends
  search/            run_combo, candidate_space, analyze_results
src/tests/
```

---

## 👤 Maintainers

- **Donato Crisostomi** - [donatocrisostomi@gmail.com](mailto:donatocrisostomi@gmail.com)

## 📜 License

MIT — see [LICENSE](LICENSE).
