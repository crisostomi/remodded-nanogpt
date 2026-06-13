# HANDOFF — remodded-nanogpt gene-search framework

> Authoritative current-state + continuation guide for the next agent/engineer.
> Last updated: 2026-06-13. Branch: `feat/gene-search-framework` (head `614e5be`,
> off `main`). Pushed to `origin`. Not yet merged / no PR opened.

---

## 1. What this project is

A **gene-combinatorial search repo** for the `modded-nanogpt` speedrun. A *record*
is just an instantiation of a set of **genes** (modular, composable changes). The
product is: **pick any valid combination of genes → generate a runnable static
training script → measure it → search for the next record (#84).**

Reproducing historical records is *not* the goal — records are known-good points
and a quarry of genes. (If exact historical reproduction is ever wanted, it's
cheap and separate: `docs/GENE_MAP.md` pins the commit sha of all 83 records, so
each is one `git show <sha>:train_gpt.py` away.)

The training script must stay fast and `torch.compile(dynamic=False,
fullgraph=True)`-friendly: **no runtime feature branching in the hot path.** Genes
are resolved at *generation* time into a static script.

---

## 2. Current status

- **Engine: complete and tested.** 49 tests pass (`uv run --extra test pytest src/tests`).
- **26 genes registered**; **15 template-toggleable**, **11 structural** (see §4).
- **Tuning genes** and the **full curriculum** are searchable config dimensions.
  Global optimizer overrides: `optim.adam` = {`lr`, `eps`, `weight_decay`},
  `optim.normuon` = {`lr`, `momentum`, `beta2`, `weight_decay`}. Per-parameter:
  `optim.params.<label>` = {`lr_mul`, `wd_mul`, `adam_betas`, `eps`, `comms`, `optim`}
  (so **Adam betas are tunable per-parameter only**, momentum is NorMuon-only). Curriculum:
  `schedule.training_stages` (batch/window/seq-len ramps, per-stage LR, MTP-weight schedule).
- **`current_record` renders behavior-equivalent to the vendored record**: the
  model/forward/attention/optimizer-step code is **byte-identical** to
  `baseline/train_gpt.py`; the config blocks (param_table, work_order, optim
  defaults, curriculum, bigram dims) are rendered from context and **value-identical**.
- The 3 findings from an adversarial review are **fixed** (override desync,
  run_combo upload ordering, silent unknown-override sections).

### ⚠️ The validation gap (read this)

**This dev environment has no GPU and no `torch`.** Every gene is verified by
**compile** (`compile(text, "<x>", "exec")`), **equivalence-diff** vs the baseline,
and **structural greps** — *never by actually training*. For architecture toggles
(which remove/rearrange known-good record code) that's a solid guarantee. For
**optimizer changes** (the remaining alleles, cautious-WD) compile-only is weak: a
numeric bug wouldn't surface until a real 8×H100 run. Sequence optimizer-internal
genes against a real GPU checkout.

---

## 3. How it works (architecture)

```
feature YAML ─► resolve_feature_set ─► build_context ─► render_train_script ─► static .py + manifest
 (configs/)      (config/)             (builder/)        (builder/ + Jinja)      (runtime/)
```

Package is `nano` under `src/nano/` (src-layout, uv template). CLI entry points are
`python -m nano.*`.

- **`features/`** — `FeatureSpec` (declares everything a gene touches: params,
  buffers, deps/conflicts/soft-conflicts, `modifies_*`, `template_toggleable`),
  the `registry` (global `FEATURES` dict; `@feature(spec)` decorator), and the gene
  catalog (`bigram.py`, `xsa.py`, `sparse_attention_gate.py`, `mudd.py`,
  `optimizer.py`, `schedule.py`, `kernels.py`). Importing `nano.features` registers
  all genes.
- **`builder/context.py`** — `BuildContext` + sub-configs (`ModelConfig`,
  `ScheduleConfig`, `OptimConfig`, ...). Holds the seeded baseline param table,
  work order, optimizer defaults and curriculum. **The single source of truth the
  template reads from.**
- **`builder/validate.py`** — the 9 spec rules + `validate_renderable` (structural
  genes must be present) + `validate_semantics` (e.g. narrow bigram dim ⇒ sliced
  injection required).
- **`builder/render.py`** — `build_context()` (validate → apply genes in topo
  order → `apply_overrides` → prune disabled-owner params → validate), and the
  Jinja render + the `format_*` helpers that turn context into the rendered config
  blocks.
- **`builder/codegen.py`** — `python -m nano.builder.codegen` CLI + the shared
  `generate()` used by the search runner.
- **`builder/templates/train_gpt.py.j2`** — see §5. **Generated; do not hand-edit.**
- **`runtime/`** — `manifest.py`, `parse_logs.py`, `tracking.py` (NoOp / LocalJsonl
  / Flywheel-stub backends), `flywheel.py` (upload CLI).
- **`config/`** — feature-set schema, YAML loading, preset resolution.
- **`search/`** — `run_combo.py` (resolve→generate→torchrun→parse→manifest;
  `--dry-run` stops before torchrun), `candidate_space.py` (`leave_one_out`,
  `enable_one`, `pairwise_toggles`, `hyperparameter_sweep`), `analyze_results.py`.

---

## 4. The gene library

**Toggleable (15)** — fully guarded across model construction, optimizer table and
hot forward; any subset renders a coherent, compile-valid script:

`adam_every_other_step`, `bigram_dim_192`, `bigram_sign_trick`, `bigram_vocab_15x`,
`fp8_lm_head`, `paired_head_attention`, `partial_key_offset`,
`residual_slice_bigram_injection`, `skip_gate`, `smear`, `sparse_attention_gate`,
`sparse_bigram_comms`, `untie_embed_at_2_3`, `xsa`, `xsa_lowering_rewrite`.

**Structural (11)** — always-on for now; the builder refuses to disable them
(`validate_renderable`). Categorized:

- **Already covered by curriculum search** (redundant as toggles —
  `TRAINING_STAGES` is fully searchable): `batch_size_schedule`,
  `max_seq_len_schedule`, `yarn_window_schedule`.
- **Entangled architecture genes** (next work, §7a): `bigram_hash` (root),
  `value_embeds`, `value_embed_gates`, `mudd_last_layers`, `mtp_loss`. Their "off"
  states are real pre-feature behaviors woven through the last-layer / post-loop
  forward (e.g. the MUDD branch consumes value-embeds). Structurally verifiable.
- **Optimizer alleles** (§7b, needs GPU validation): `normuon`, `polar_express`
  (the optimizer / orthogonalizer slot), `cautious_weight_decay`. Need the
  **allele mechanism** + alternative implementations harvested from git history.

---

## 5. The template is *derived*, not hand-written (critical)

`src/nano/builder/templates/train_gpt.py.j2` is **generated** from the vendored
record `baseline/train_gpt.py` (pinned @ `b534dfd`, see `baseline/SOURCE_SHA.txt`)
by **`baseline/build_template.py`**, which applies a small set of audited, surgical
replacements (each asserted to match *exactly once*) so every untouched line stays
byte-identical to the record.

**Never hand-edit the `.j2`.** To change template behavior, edit
`baseline/build_template.py` and regenerate:

```sh
python baseline/build_template.py     # rewrites the .j2; must be committed
```

The builder has two replacement kinds:
- **exact** `(old, new)` tuples in the `EXACT` list — `old` must occur exactly once.
- **region** replacements (param_table, work_order, TRAINING_STAGES) — start/end
  anchors. **Regions run before EXACT** (some EXACT edits rewrite region end-anchors).

Config blocks (param_table, work_order, optim defaults, TRAINING_STAGES) are NOT
guarded in the template — they're replaced with `{{ ... }}` placeholders and
rendered per-build from the context by `render.py`'s `format_*` functions.

---

## 6. How to add a new gene (the recipe)

1. **Declare it** — add `@feature(FeatureSpec(name=..., requires=..., conflicts=...,
   owns_params=..., modifies_*=True, template_toggleable=True))` in the right
   `src/nano/features/*.py`. **Add the boolean field it controls to the relevant
   `*Config` dataclass in `builder/context.py`** (e.g. `use_x: bool = False` on
   `ModelConfig`) — this field MUST exist or the template's `{% if model.use_x %}`
   raises at render time (StrictUndefined). The `apply(ctx)` body sets that flag
   (`ctx.model.use_x` / `ctx.optim.use_x` / `ctx.schedule.use_x` / `ctx.loss.use_x`).
2. **Owned params** — if the gene introduces a *new* parameter, add its row to
   `BASELINE_PARAM_TABLE` + `BASELINE_WORK_ORDER` in `builder/context.py` and list it
   in the spec's `owns_params`. The prune step drops it automatically when the gene
   is disabled (owner-map in `render._param_owner_map`). If it reuses an existing
   param, just list it in `owns_params`.
3. **Guard the template** — add `guard("model.use_x", body, else_body)` replacements
   to `baseline/build_template.py` (exact or region) for every site the gene touches:
   model `__init__`, the hot forward, bf16 casts, `dist.broadcast`, etc.
4. **Wire the preset FIRST (before verifying)** — if the gene guards code that is
   *present in the record* (the usual case — it's currently always-on), add it to
   `configs/feature_sets/current_record.yaml` **before** the equivalence check.
   Otherwise the rendered `current_record` correctly drops the guarded line and the
   equivalence check fails. ⚠️ The gene-on script still **compiles** in that state, so
   the compile check gives false confidence — only the byte-diff catches the dropped line.
5. **Regenerate** — `python baseline/build_template.py`.
6. **Verify** (no GPU, so):
   - `current_record` (with the gene now in the preset) still compiles AND stays
     byte-equivalent to the record — run the equivalence check in §8;
   - the gene-OFF script compiles AND the off-state is correct (grep the removed/added
     code);
   - add tests under `src/tests/` (mirror `test_build_context.py` / `test_codegen.py`),
     then `uv run --extra test pytest src/tests`.

> If you work in a **second checkout or git worktree**, see the editable-install
> gotcha in §9 — you must `uv sync --extra test` *inside that tree* (or set
> `PYTHONPATH=<tree>/src`) or your edits won't be the `nano` that gets imported.

Worked examples to copy: `sparse_attention_gate` (param + forward + table), `smear`
(localized forward), `partial_key_offset` (single forward guard), `untie_embed_at_2_3`
(single optimizer-manager guard).

---

## 7. Next steps (prioritized)

**(a) Entangled architecture genes** — `bigram_hash` root, `value_embeds`,
`value_embed_gates`, `mudd_last_layers`, `mtp_loss`. Harvest each "off" state from
the pre-feature commit (shas in `docs/GENE_MAP.md`), guard the (multiple) forward
sites, handle the MUDD/value-embed interaction. Compile/structure-verifiable.

**(b) Allele mechanism + optimizer alleles** — implement "exactly-one-of" slots
(see `GENE_MAP.md` §3: optimizer, orthogonalizer, attention_backend, cross_entropy,
mlp_kernel). Add the alternative implementations (Muon-without-NorMuon,
Newton-Schulz, plain WD, ...) harvested from git. **Validate on a real GPU.** This
is the densest remaining value and the highest risk.

**(c) Reframe the 3 schedule genes** as curriculum aliases (or drop them) — they're
subsumed by `schedule.training_stages` search.

**(d) Curriculum ergonomics** — a `curriculum_sweep` helper (batch/window ramps) on
top of the `schedule.training_stages` override.

**(e) Flywheel backend** — currently a stub (`runtime/tracking.py::FlywheelBackend`).
Wire to the SDK/API when creds/docs land. `upload_run` already swallows failures so
training is never affected.

**(f) Optional: sha-backed faithful reproduction** of all 83 records (separate from
search) — trivial given `GENE_MAP.md`'s sha index.

---

## 8. Runbook

```sh
uv sync --extra test                       # install (jinja2, pyyaml, numpy, pytest)
uv run --extra test pytest src/tests       # 49 tests (the `test` extra carries pytest)
python baseline/build_template.py          # regenerate the .j2 after editing the builder
python -m nano.builder.codegen --preset current_record --out generated/train_current_record.py
python -m nano.search.run_combo --preset current_record --disable xsa --dry-run
```

**Equivalence check** (the core regression — model/forward must stay byte-identical):

```python
from nano.config.presets import resolve_feature_set
from nano.builder.render import build_context, render_train_script_text
fs = resolve_feature_set(preset="current_record")
ctx = build_context(fs.enabled, overrides=fs.overrides)
text = render_train_script_text(ctx, header={})
compile(text, "<cr>", "exec")
# Diff `text` (minus the generated `# Generated by ...` header) vs
# baseline/train_gpt.py. The invariant: the FIRST byte-difference is at
# ~baseline L1690, and every diff hunk is a config block that is *rendered* from
# context -- bigram dims (~L1690), TRAINING_STAGES (~L1762), cooldown_frac (the
# TrainingSchedule line ~L1775), param_table (~L1809), work_order (~L1838) --
# plus the header. These hunks differ in FORMATTING (single-quoted dict values,
# one-line work_order, collapsed TRAINING_STAGES) but are value-identical.
# Everything ABOVE L1690 -- the optimizer step (~L367-790) and the whole
# model/forward/attention (~L949-1499) -- stays byte-identical. The check that
# matters: NO diff hunk in that model/forward/optimizer region. (Line numbers
# drift when the baseline is re-vendored; trust "no hunk before the first
# config block" over the exact numbers.)
```

**Search demo** (architecture × tuning → distinct valid scripts):

```python
from nano.search.candidate_space import leave_one_out, hyperparameter_sweep
# leave_one_out(base, [genes...]) and hyperparameter_sweep(base, {"optim.adam.lr":[...]})
# -> FeatureSets; build_context + render each; they all compile and differ.
```

---

## 9. Key references & gotchas

- **`docs/GENE_MAP.md`** — all 83 records → sha → gene(s), the 10 allele groups, the
  16 substrate eras, coverage vs current genes, and an era-by-era expansion plan.
  (Sha resolution: #5 and #27 are flagged **low** confidence; #7/#15/#17/#25/#66 are
  **medium** — re-confirm these against the record logs before relying on them.)
- **`SPECS.md`** — the original MVP spec + a status addendum at the top.
- **`baseline/`** — the vendored record (template ground truth) + `build_template.py`.
- **Gotchas:**
  - `model.use_*` overrides are **forbidden** — feature membership is the only source
    of truth for both model construction and the param table.
  - Unknown override sections / feature-set keys **error** (no silent no-ops).
  - The bigram-dim/sliced-injection semantic constraint: `bigram_dim < model_dim`
    requires `residual_slice_bigram_injection`.
  - `param_table`/`work_order` order is load-bearing (`scatter_order = list(param_table)`);
    keep `BASELINE_PARAM_TABLE` insertion order matching the record.
  - Records form a chain; genes are tied to a *substrate*. Cross-substrate
    combinations require porting a gene forward (see GENE_MAP eras). The current
    code is era **E15**.
  - **Editable-install / worktree gotcha:** `nano` is an editable install whose
    `.pth` hardcodes *this* repo's `src/`. If you work in a second checkout or a
    git worktree, the main `.venv`'s `python` still imports `nano` from the
    original tree, so your edits aren't exercised (tests fail with confusing
    `Unknown feature` errors). Fix: `uv sync --extra test` **inside that tree**
    (builds `nano` from it), or prefix commands with `PYTHONPATH=<tree>/src`.
