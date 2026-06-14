# HANDOFF — remodded-nanogpt gene-search framework

> Authoritative current-state + continuation guide for the next agent/engineer.
> Last updated: 2026-06-14. Base framework + the entangled-gene toggle work
> (§4, §7a) are committed on `main` (head `0a3b218`). In the working tree, not
> yet committed: the curriculum closeout (§7c retire the 3 schedule genes + §7d
> `curriculum_sweep`); `mtp_loss` made toggleable (§7a); `cautious_weight_decay`
> made a real toggle; the **allele mechanism** + the first real allele
> (orthogonalizer `newton_schulz`) (§7b).

---

## 1. What this project is

A **gene-combinatorial search repo** for the `modded-nanogpt` speedrun. A *record*
is just an instantiation of a set of **genes** (modular, composable changes). The
product is: **pick any valid combination of genes → generate a runnable static
training script → measure it → search for the next record (#84).**

Reproducing historical records is *not* the goal — records are known-good points
and a quarry of genes. (If exact historical reproduction is ever wanted, it's
cheap and separate: `docs/GENE_MAP.md` pins the commit sha of all 83 records. Those
are **upstream** shas, not in this repo — fetch them via the blobless-clone recipe
in §9, then `git -C /tmp/mnng show <sha>:<path>`.)

The training script must stay fast and `torch.compile(dynamic=False,
fullgraph=True)`-friendly: **no runtime feature branching in the hot path.** Genes
are resolved at *generation* time into a static script.

---

## 2. Current status

- **Engine: complete and tested.** 84 tests pass (`uv run --extra test pytest src/tests`;
  if the local `uv.lock` fails to parse with an older `uv`, use
  `PYTHONPATH=src <python> -m pytest src/tests` — see §9 for the interpreter gotcha).
- **24 genes registered**: **23 template-toggleable** (21 additive + 2 orthogonalizer
  allele members), **1 structural** (`normuon`), across **1 allele slot** (see §4).
- **Tuning genes** and the **full curriculum** are searchable config dimensions.
  Global optimizer overrides: `optim.adam` = {`lr`, `eps`, `weight_decay`},
  `optim.normuon` = {`lr`, `momentum`, `beta2`, `weight_decay`}. Per-parameter:
  `optim.params.<label>` = {`lr_mul`, `wd_mul`, `adam_betas`, `eps`, `comms`, `optim`}
  (so **Adam betas are tunable per-parameter only**, momentum is NorMuon-only). Curriculum:
  `schedule.training_stages` (batch/window/seq-len ramps, per-stage LR, MTP-weight schedule),
  swept ergonomically via `search.candidate_space.curriculum_sweep` / `make_curriculum`.
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
**optimizer-internal changes** compile-only is weak: a numeric bug wouldn't surface
until a real 8×H100 run.

**GPU-VALIDATION TODO (for the agent with the 8×H100 nodes).** These off-states /
alleles are structurally verified here (compile + grep + on-state byte-equivalence)
but their *numeric* end-to-end behaviour is unverified — train them and compare to
the record before trusting:
- `cautious_weight_decay` **off** — plain decoupled WD (the sign-agreement mask is
  dropped in both the Adam and NorMuon paths). Standard decoupled-WD form, but
  unvalidated on this stack.
- orthogonalizer `newton_schulz` allele — the `polar_express` scaffold with the
  canonical Muon quintic `(3.4445, -4.7750, 2.0315)` ×5 in place of the tuned Polar
  Express coefficients. The coefficients are faithful (published Muon NS); the
  end-to-end training behaviour on the E15 fused scaffold is unverified.

`mtp_loss` is **not** in this list: its off-state (single-token, `mtp_weights=[1.0]`)
runs the same fused kernel the record already exercises in its last two stages, so
n_predict=1 is not new code (see §4).

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
  `enable_one`, `pairwise_toggles`, `hyperparameter_sweep`, `curriculum_sweep`),
  `analyze_results.py`.

---

## 4. The gene library

**Additive toggleable (21)** — fully guarded across model construction, optimizer
table and hot forward; any subset (respecting the dependency lattice below) renders
a coherent, compile-valid script:

`adam_every_other_step`, `bigram_dim_192`, `bigram_hash`, `bigram_sign_trick`,
`bigram_vocab_15x`, `cautious_weight_decay`, `fp8_lm_head`, `mtp_loss`,
`mudd_last_layers`, `paired_head_attention`, `partial_key_offset`,
`residual_slice_bigram_injection`, `skip_gate`, `smear`, `sparse_attention_gate`,
`sparse_bigram_comms`, `untie_embed_at_2_3`, `value_embeds`, `value_embed_gates`,
`xsa`, `xsa_lowering_rewrite`.

Two of these are **config/optimizer genes** added most recently (not architecture):
- `mtp_loss` (§7a, done) — multi-token prediction is purely the loss objective on
  E15: the fused softcapped-CE kernel reads `n_predict = mtp_weights.shape[0]`
  future tokens per position from the same plain next-token `target_seq`. Off-state
  is single-token (`mtp_weights == [1.0]` every stage), realized at build time by
  `render._finalize_loss_schedule` — **no template branch**. n_predict=1 already
  runs in the record's last two stages, so the off path is not new GPU code.
- `cautious_weight_decay` (done) — off-state drops the sign-agreement `mask` in both
  the Adam (`mask = (update * p_slice) > 0`) and NorMuon (`mask = (grad * p_precise)
  >= 0`) paths → plain decoupled WD. Numeric behaviour GPU-validation-pending (§2).

The four **entangled architecture genes** made toggleable most recently
(`value_embeds`, `value_embed_gates`, `mudd_last_layers`, `bigram_hash`) form a
dependency lattice — the builder rejects an enabled set that violates it:

- `value_embed_gates` **requires** `value_embeds` (off-state: ungated
  `aux_v = ve_view.view(B, T, -1)`).
- `mudd_last_layers` **requires** `value_embeds` (its last-layer branch fuses
  `ve_view` and its post-loop mixer reads `ve[1]`). Off-state: the last layer
  degrades from the MUDD branch to the normal per-layer residual path
  (`elif ve[i]` → `if ve[i]`), the post-loop mixer + `cache[9]` + `mu` sentinel +
  the `init_mudd`/`forward_mudd` methods are dropped. MUDD does **not** require
  `bigram_hash`: its `mu[11] * x0_bigram` injection is guarded independently, so
  MUDD composes with bigram on **or** off.
- the whole bigram family (`bigram_sign_trick`, `bigram_vocab_15x`,
  `bigram_dim_192`, `residual_slice_bigram_injection`, `sparse_bigram_comms`)
  **requires** `bigram_hash`, so dropping `bigram_hash` cascades them off.
  Bigram-off also drops the `x0` re-injection (`x0_lambdas` is bundled into this
  gene's `owns_params`), zeroes the data-loader hash to a placeholder, and turns
  the TrainingManager sparse-grad-comms methods into no-ops (the optimizer's
  `sharded_sparse` path self-gates on the `comms` field — no param carries it
  once `bigram_embed` is pruned, so `model.bigram_embed` is never dereferenced).

> **Retired (§7c, done):** `batch_size_schedule`, `max_seq_len_schedule`,
> `yarn_window_schedule` were no-op manifest tags — their `use_*` flags were read
> by nothing; the batch/seq-len/window ramps live entirely in the searchable
> `TRAINING_STAGES`. Removed (gene defs, dead `ScheduleConfig` flags, preset
> entries) and replaced by the `curriculum_sweep` config-search helper (§7d).

**Allele slots (exactly-one-of)** — a `FeatureSpec.allele_group` declares a set of
mutually-exclusive members occupying one named slot; the builder rejects enabling
two members (`validate_alleles`) and rendering requires exactly one
(`validate_renderable`). The selected member drives a context field the template
reads. Implemented so far:

- **`orthogonalizer`** = `polar_express` | `newton_schulz` (§7b, done). Both share
  the *same* fused-momentum / bf16 / Triton-kernel scaffold (the `polar_express`
  function); they differ **only** in the odd-polynomial iteration coefficient table
  (`for a, b, c in polar_express_coeffs`). `polar_express` = 5 per-step tuned rows
  (the record default); `newton_schulz` = the canonical Muon quintic
  `(3.4445, -4.7750, 2.0315)` ×5. The selection is `ctx.optim.orthogonalizer`; the
  guard swaps only the coefficient block. NS numerics are GPU-validation-pending (§2).

**Structural (1)** — always-on; the builder refuses to disable it
(`validate_renderable`). `normuon` is the optimizer slot — still single-member
(its `muon`-without-the-variance-estimator alternative is a TODO, §7b).

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

The `.j2` is a **committed generated artifact** — regenerate and commit it in
the *same* change as any `build_template.py` edit. The test
`test_codegen.py::test_committed_template_matches_build_script` calls
`build_template.build_template_text()` and byte-compares it to the committed
`.j2`, so stale-template drift fails loudly in CI instead of silently shipping a
broken off-state to the GPU. (`build_template_text()` is the pure, no-I/O core;
`main()` is the thin file-writing wrapper.)

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
(single optimizer-manager guard), `mtp_loss` (config-only gene, no template guard),
`cautious_weight_decay` (two optimizer-path guards).

**Adding an allele to a slot:** give the new feature `allele_group="<slot>"` +
`template_toggleable=True`, set a context field in its `apply` (e.g.
`ctx.optim.<slot> = "<member>"`), and guard the differing region on that field
(`guard('optim.<slot> == "<member>"', new_body, default_body)` — default_body must
be byte-identical to the record so the default member preserves equivalence). The
`validate_alleles`/`validate_renderable` "exactly-one" rules are automatic. Worked
example: the `orthogonalizer` slot (`polar_express` | `newton_schulz`) — a single
coefficient-table guard.

---

## 7. Next steps (prioritized)

**(a) Entangled architecture genes + `mtp_loss`** — ✅ **DONE (all 5)**:
`value_embeds`, `value_embed_gates`, `mudd_last_layers`, `bigram_hash` (§4 lattice)
and now `mtp_loss`. The earlier "`mtp_loss` needs the external kernel" worry was
wrong: reading `triton_kernels.py` (it **is** vendored at `baseline/`) + the data
loader shows MTP is purely the loss objective — the fused kernel reads `n_predict =
mtp_weights.shape[0]` future tokens from the same plain next-token `target_seq`, so
off = `mtp_weights==[1.0]` every stage (a build-time collapse, no template branch;
n_predict=1 already runs in the record). See `render._finalize_loss_schedule`.

**(b) Allele mechanism + optimizer alleles** — ✅ **mechanism + first allele DONE**;
remaining alleles are TODO. The `FeatureSpec.allele_group` "exactly-one-of" slot
machinery is implemented (`validate_alleles` rejects two members; `validate_renderable`
requires one; the selected member drives a context field the template reads). The
first real allele — **orthogonalizer `polar_express | newton_schulz`** — is wired
faithfully (a coefficient-table swap; §4). Numeric validation of `newton_schulz` →
the §2 GPU TODO.

  **Remaining allele slots — TODO (the densest remaining value).** Harvest the real
  bodies from the upstream record commits (now reachable — see the harvest recipe in
  §9) and adapt them to the E15 fused substrate; **validate each on the 8×H100 nodes**:
  - **`optimizer`** slot — add `muon` (NorMuon minus the Adafactor-style variance
    estimator) alongside `normuon`. Touches `_normuon_update` (drop
    `_apply_normuon_variance_reduction`). `normuon` is currently the single
    structural member; convert it to an `allele_group="optimizer"` member when you add `muon`.
  - **`attention_backend`** slot — `flex_attention` | `flash_attention_3` (records
    29+). A substrate-level swap; bigger lift.
  - **`cross_entropy`** slot — the fused softcapped MTP CE kernel (record 60) vs
    earlier CE kernels; entangled with `fp8_lm_head`.
  - **`mlp_kernel`** slot — the `ReLUSqrdMLP` / `FusedLinearReLUSquareFunction`
    Triton kernel vs a plain implementation.

**(c) Retire the 3 schedule genes** — ✅ **DONE**: `batch_size_schedule`,
`max_seq_len_schedule`, `yarn_window_schedule` were no-op tags subsumed by
`schedule.training_stages` search; removed (see §4 retired-note). Verified by the
equivalence-diff (`current_record` render unchanged — they guarded nothing) + tests.

**(d) Curriculum ergonomics** — ✅ **DONE**: `search.candidate_space.curriculum_sweep`
(cartesian-product over per-stage batch/seq-len/window/LR ramps → `FeatureSet`s
carrying `schedule.training_stages` overrides) + `make_curriculum` (build one
curriculum). Scalars broadcast across stages; per-stage lists ramp; strict on bad
shapes. See `test_search.py`.

**(e) Flywheel backend** — currently a stub (`runtime/tracking.py::FlywheelBackend`).
Wire to the SDK/API when creds/docs land. `upload_run` already swallows failures so
training is never affected.

**(f) Optional: sha-backed faithful reproduction** of all 83 records (separate from
search) — trivial given `GENE_MAP.md`'s sha index.

---

## 8. Runbook

```sh
uv sync --extra test                       # install (jinja2, pyyaml, numpy, pytest)
uv run --extra test pytest src/tests       # 84 tests (the `test` extra carries pytest)
python baseline/build_template.py          # regenerate the .j2 after editing the builder
python -m nano.builder.codegen --preset current_record --out generated/train_current_record.py
python -m nano.search.run_combo --preset current_record --disable xsa --dry-run
```

On *this* box `uv` can't parse the lockfile and the shell's `python3` is flaky
(§9). The reliable interpreter with all deps + pytest is
`/home/donato/miniconda3/bin/python3`; run tests/codegen as
`PYTHONPATH=src /home/donato/miniconda3/bin/python3 -m pytest src/tests`.

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
from nano.search.candidate_space import leave_one_out, hyperparameter_sweep, curriculum_sweep
# leave_one_out(base, [genes...]) and hyperparameter_sweep(base, {"optim.adam.lr":[...]})
# curriculum_sweep(base, {"batch_size":[8*2048*8, 16*2048*8], "window_sizes":[[(1,3),(3,7),(5,11),(6,13)]]})
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
- **Harvesting allele bodies / old genes from upstream history (the recipe).** The
  record shas in `GENE_MAP.md` are **upstream** `KellerJordan/modded-nanogpt`
  commits — they are **not** in this repo (it has 6 commits, no upstream remote), so
  `git show <sha>:...` fails here until you fetch them. Network egress works; clone
  the upstream blobless (commit graph now, file blobs on demand) and read from it
  without touching this repo's git:
  ```sh
  git clone --filter=blob:none --no-checkout \
      https://github.com/KellerJordan/modded-nanogpt.git /tmp/mnng
  # the vendored baseline is byte-identical to upstream b534dfd (verified):
  git -C /tmp/mnng show b534dfd:train_gpt.py    | diff - baseline/train_gpt.py     # empty
  git -C /tmp/mnng show b534dfd:triton_kernels.py | diff - baseline/triton_kernels.py  # empty
  ```
  So `/tmp/mnng` is a faithful source for the remaining allele bodies (§7b). Old
  records live under `records/track_1_short/<date>_<name>/train_gpt2.py`; recent ones
  at root `train_gpt.py`. NOTE: `git log -S`/pickaxe over the blobless clone is slow
  (a blob fetch per commit) — prefer `git show <sha>:<path>` on a known commit.
- **Gotchas:**
  - **Flaky `python3` + incomplete `.venv`.** The shell's `python3` sometimes
    resolves to the project `.venv/bin/python3` (missing `pytest` **and** GitPython,
    which `nano/__init__.py` imports) and sometimes to `/usr/bin/python3` (missing
    `pytest`). The one interpreter with everything is
    **`/home/donato/miniconda3/bin/python3`** — pin it for tests, codegen and the
    equivalence check (`PYTHONPATH=src /home/donato/miniconda3/bin/python3 ...`).
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
  - **`uv.lock` may not parse with an older `uv`:** the committed `uv.lock` was
    written by a newer `uv` (the editable root has no `version` field), so
    `uv 0.5.4` errors with `missing field version` and `uv run --extra test ...`
    fails before running anything. Fallback that needs no `uv`: the repo deps
    (jinja2, pyyaml, numpy, pytest) are import-only, so
    `PYTHONPATH=src python3 -m pytest src/tests` and
    `PYTHONPATH=src python3 -m nano.builder.codegen ...` work directly. The codegen
    path needs no GPU/torch.
  - **Don't run git-mutating commands in the shared working tree when fanning out
    review agents.** The off-state work lives as *uncommitted* edits (incl. the
    regenerated `.j2`); an audit agent that runs `git checkout`/`restore` on the
    `.j2` will silently revert it to the stale committed version and make the
    off-states look broken. Tell review agents to be read-only (render variants to
    `/tmp`, never regenerate into the repo, no git mutations). Worktree isolation
    is *wrong* for auditing uncommitted work — a fresh worktree won't contain it.
  - **Editable-install / worktree gotcha:** `nano` is an editable install whose
    `.pth` hardcodes *this* repo's `src/`. If you work in a second checkout or a
    git worktree, the main `.venv`'s `python` still imports `nano` from the
    original tree, so your edits aren't exercised (tests fail with confusing
    `Unknown feature` errors). Fix: `uv sync --extra test` **inside that tree**
    (builds `nano` from it), or prefix commands with `PYTHONPATH=<tree>/src`.
