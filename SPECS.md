Use this as the implementation spec for the coding agent.

# Project spec: modular feature-search framework for `modded-nanogpt`

## Goal

Refactor `modded-nanogpt` so record-winning changes can be represented as modular **features**, activated or disabled through config, then executed as reproducible experiments.

The training code must remain fast. The final training script for a run should be static and compatible with the current `torch.compile(..., dynamic=False, fullgraph=True)` style. Avoid runtime feature branching inside hot forward paths.

GitHub remains the source of truth for code and configs. Flywheel is only for logging, visualization, comparison, and artifact browsing.

Current repo facts to preserve:

The current `train_gpt.py` logs its own source plus `triton_kernels.py` before training, then logs environment info, initializes distributed training, compiles the model, warms up selected transition steps, resets state, trains, validates, and writes timing/loss logs. Preserve that auditability and warmup/reset discipline.  

The current optimizer already uses a parameter table with labels, optimizer type, communication mode, Adam betas, LR multipliers, WD multipliers, scatter order, and work order. The modular system should build that table from enabled features instead of hardcoding it in one place. 

## Core design principle

A “feature” is not just a flag in the forward pass.

A feature may touch:

```text
model parameters
buffers
forward graph
optimizer param table
data loader
schedule
warmup shapes
distributed broadcasts
logging metadata
artifact outputs
```

So each feature must declare all surfaces it modifies.

Do not implement this as:

```python
if flags.xsa:
    ...
if flags.bigram_sign_trick:
    ...
if flags.mudd:
    ...
```

inside the hot compiled model.

Instead:

```text
feature YAML -> build context -> generated/static train script -> torchrun -> local manifest/logs -> Flywheel upload
```

## Non-goals for MVP

Do not fully rewrite all of `train_gpt.py` immediately.

Do not modularize every historical record at once.

Do not optimize search algorithms yet.

Do not depend on Flywheel to reconstruct a run. Every run must be reproducible from GitHub plus local artifacts.

Do not remove the ability to run a flat `train_gpt.py` style script.

## Desired repo structure

Create this structure:

```text
modded_nanogpt/
  __init__.py

  config/
    base.py
    schema.py
    presets.py

  features/
    __init__.py
    registry.py
    base.py
    bigram.py
    xsa.py
    sparse_attention_gate.py
    mudd.py
    schedule.py
    optimizer.py
    kernels.py

  builder/
    __init__.py
    context.py
    validate.py
    render.py
    codegen.py

  runtime/
    __init__.py
    logging.py
    manifest.py
    flywheel.py
    parse_logs.py

  search/
    __init__.py
    run_combo.py
    candidate_space.py
    analyze_results.py

configs/
  feature_sets/
    current_record.yaml
    record_83.yaml
    pr_299.yaml
    pr_264.yaml
    pr_259.yaml
    pr_317.yaml

generated/
  .gitkeep

experiments/
  runs/
    .gitkeep
```

Keep existing files working where possible.

## Primary CLI commands

Implement these commands.

### 1. Build a static train script

```bash
python -m modded_nanogpt.builder.codegen \
  --feature-set configs/feature_sets/current_record.yaml \
  --out generated/train_current_record.py
```

### 2. Build from preset with overrides

```bash
python -m modded_nanogpt.builder.codegen \
  --preset current_record \
  --disable sparse_attention_gate \
  --enable xsa_lowering_rewrite \
  --out generated/train_current_record_minus_sag.py
```

### 3. Run a combo and create local manifest

```bash
python -m modded_nanogpt.search.run_combo \
  --preset current_record \
  --disable sparse_attention_gate \
  --enable xsa_lowering_rewrite \
  --run-name current_minus_sag_xsa_rewrite_s1385_seed0 \
  --nproc-per-node 8
```

This should internally:

```text
resolve feature set
validate dependencies/conflicts
generate static train script
compute generated file hash
launch torchrun
tee raw log
parse final metrics
write manifest.json
optionally upload to Flywheel
```

### 4. Upload existing run to Flywheel

```bash
python -m modded_nanogpt.runtime.flywheel \
  --manifest experiments/runs/<run_id>/manifest.json
```

Flywheel API details are unknown. Implement a backend interface with a no-op backend and a Flywheel adapter stub.

## Config schema

Use YAML for feature sets.

Example:

```yaml
name: current_record_minus_sag_xsa_rewrite
base: current_record

enable:
  - xsa_lowering_rewrite

disable:
  - sparse_attention_gate

schedule:
  num_scheduled_iterations: 1375
  num_extension_iterations: 10

tracking:
  project: modded-nanogpt-feature-search
  tags:
    - h100x8
    - ablation
    - xsa
    - no_sag
```

Preset example:

```yaml
name: pr_299
description: Sign trick on bigram embeddings
enable:
  - bigram_hash
  - bigram_sign_trick
  - bigram_vocab_15x
  - bigram_dim_192
  - residual_slice_bigram_injection
```

Current record example:

```yaml
name: current_record
enable:
  - fp8_lm_head
  - polar_express
  - normuon
  - cautious_weight_decay
  - adam_every_other_step
  - mtp_loss
  - untie_embed_at_2_3
  - bigram_hash
  - bigram_sign_trick
  - bigram_vocab_15x
  - bigram_dim_192
  - residual_slice_bigram_injection
  - sparse_bigram_comms
  - value_embeds
  - value_embed_gates
  - smear
  - skip_gate
  - sparse_attention_gate
  - xsa
  - mudd_last_layers
  - paired_head_attention
  - partial_key_offset
  - yarn_window_schedule
  - batch_size_schedule
  - max_seq_len_schedule
```

Do not worry if this list is incomplete in MVP. The point is to make it easy to grow.

## Feature model

Create `modded_nanogpt/features/base.py`.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str = ""

    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    soft_conflicts: tuple[str, ...] = ()

    owns_params: tuple[str, ...] = ()
    owns_buffers: tuple[str, ...] = ()

    modifies_model: bool = False
    modifies_forward: bool = False
    modifies_optimizer: bool = False
    modifies_schedule: bool = False
    modifies_data: bool = False
    modifies_warmup: bool = False
    modifies_distributed: bool = False
    modifies_loss: bool = False
    modifies_logging: bool = False


class Feature(Protocol):
    spec: FeatureSpec

    def apply(self, ctx: "BuildContext") -> None:
        ...
```

Every feature module exports one or more `Feature` objects.

## Build context

Create `modded_nanogpt/builder/context.py`.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    num_layers: int = 11
    num_heads: int = 6
    head_dim: int = 128
    model_dim: int = 768

    use_fp8_lm_head: bool = True
    use_xsa: bool = False
    use_sparse_attention_gate: bool = False
    use_mudd: bool = False
    use_bigram_hash: bool = False
    use_bigram_sign_trick: bool = False

    bigram_vocab_size: int | None = None
    bigram_dim: int | None = None
    bigram_sign_table_rows: int | None = None


@dataclass
class ScheduleConfig:
    num_scheduled_iterations: int = 1375
    num_extension_iterations: int = 10
    cooldown_frac: float = 0.60
    split_embed_stage: int = 2
    ws_post_yarn_ext: int = 20
    training_stages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimParamConfig:
    optim: str
    comms: str
    adam_betas: list[float] | None = None
    lr_mul: float = 1.0
    wd_mul: float = 1.0


@dataclass
class OptimConfig:
    param_table: dict[str, OptimParamConfig] = field(default_factory=dict)
    work_order: list[str] = field(default_factory=list)
    scatter_order: list[str] = field(default_factory=list)
    adam_defaults: dict[str, Any] = field(default_factory=dict)
    normuon_defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    needs_bigram_inputs: bool = False
    align_train_to_bos: bool = True


@dataclass
class DistributedConfig:
    broadcast_buffers: list[str] = field(default_factory=list)


@dataclass
class LossConfig:
    use_mtp: bool = False
    use_softcapped_ce: bool = True


@dataclass
class RenderConfig:
    template_name: str = "train_gpt.py.j2"
    include_feature_comments: bool = True


@dataclass
class BuildContext:
    enabled_features: set[str] = field(default_factory=set)

    model: ModelConfig = field(default_factory=ModelConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    metadata: dict[str, Any] = field(default_factory=dict)
```

## Registry

Create `modded_nanogpt/features/registry.py`.

```python
FEATURES: dict[str, Feature] = {}


def register(feature: Feature) -> Feature:
    name = feature.spec.name
    if name in FEATURES:
        raise ValueError(f"Duplicate feature: {name}")
    FEATURES[name] = feature
    return feature


def get_feature(name: str) -> Feature:
    try:
        return FEATURES[name]
    except KeyError:
        raise KeyError(f"Unknown feature: {name}") from None
```

Import all feature modules in `features/__init__.py` so registration happens.

## Feature validation

Create `modded_nanogpt/builder/validate.py`.

Rules:

1. Every enabled feature must exist.
2. Every `requires` feature must be enabled.
3. No `conflicts` pair may be enabled.
4. Emit warnings for `soft_conflicts`.
5. No two enabled features may own the same param unless explicitly allowed.
6. No two enabled features may own the same buffer unless explicitly allowed.
7. Every model parameter expected by the generated model must exist in optimizer `param_table`.
8. Every optimizer `param_table` entry must correspond to an actual generated model parameter.
9. `work_order` and `scatter_order` must contain exactly the same labels as `param_table`.

The current optimizer asserts param table consistency. Preserve this style, but catch obvious problems before launching GPU runs. 

## Feature application flow

Implement:

```python
def build_context(feature_names: set[str], overrides: dict | None = None) -> BuildContext:
    validate_feature_names(feature_names)

    ctx = BuildContext(enabled_features=set(feature_names))

    # Deterministic application order.
    ordered = topo_sort_features(feature_names)

    for name in ordered:
        FEATURES[name].apply(ctx)

    apply_overrides(ctx, overrides)
    validate_context(ctx)

    return ctx
```

Topological sort must respect `requires`.

## MVP features to implement first

### Feature: `bigram_hash`

File: `modded_nanogpt/features/bigram.py`

Spec:

```python
FeatureSpec(
    name="bigram_hash",
    requires=(),
    owns_params=("bigram_embed", "x0_lambdas", "bigram_lambdas"),
    modifies_model=True,
    modifies_forward=True,
    modifies_optimizer=True,
    modifies_data=True,
)
```

Effects:

```text
ctx.model.use_bigram_hash = True
ctx.model.bigram_vocab_size default = 50304 * 5 unless another feature overrides
ctx.model.bigram_dim default = ctx.model.model_dim unless another feature overrides
ctx.data.needs_bigram_inputs = True
add bigram_embed to optimizer table
add x0_lambdas to optimizer table
add bigram_lambdas to optimizer table
include get_bigram_hash in generated script
include bigram_inputs from data loader
inject bigram embeddings into residual stream
```

Historical context: PR #201 added bigram hash embeddings, reduced steps, adjusted cooldown, and described CPU-side bigram hash generation. 

### Feature: `bigram_sign_trick`

Spec:

```python
FeatureSpec(
    name="bigram_sign_trick",
    requires=("bigram_hash",),
    owns_buffers=("bigram_sign_table",),
    modifies_model=True,
    modifies_forward=True,
    modifies_distributed=True,
)
```

Effects:

```text
ctx.model.use_bigram_sign_trick = True
ctx.model.bigram_sign_table_rows default = 8192
ctx.distributed.broadcast_buffers includes "bigram_sign_table"
generated model registers bigram_sign_table buffer
forward computes sign_idx
forward multiplies bigram embedding by sign table
```

Historical context: PR #299 implemented sign trick compression and changed bigram table/injection behavior. 

### Feature: `bigram_vocab_15x`

Spec:

```python
FeatureSpec(
    name="bigram_vocab_15x",
    requires=("bigram_hash",),
    modifies_model=True,
)
```

Effects:

```text
ctx.model.bigram_vocab_size = 50304 * 15
```

### Feature: `bigram_dim_192`

Spec:

```python
FeatureSpec(
    name="bigram_dim_192",
    requires=("bigram_hash",),
    modifies_model=True,
)
```

Effects:

```text
ctx.model.bigram_dim = 192
```

### Feature: `residual_slice_bigram_injection`

Spec:

```python
FeatureSpec(
    name="residual_slice_bigram_injection",
    requires=("bigram_hash", "bigram_dim_192"),
    modifies_forward=True,
)
```

Effects:

```python
x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + x0_bigram * bigram_lambdas[i]
```

This is necessary when `bigram_dim < model_dim`. PR #299’s patch moved from full residual addition to sliced residual updates. 

### Feature: `xsa`

File: `modded_nanogpt/features/xsa.py`

Spec:

```python
FeatureSpec(
    name="xsa",
    owns_params=("xsa_alphas",),
    soft_conflicts=("sparse_attention_gate",),
    modifies_model=True,
    modifies_forward=True,
    modifies_optimizer=True,
)
```

Effects:

```text
ctx.model.use_xsa = True
add xsa_alphas param
add xsa_alphas to optimizer table
apply XSA only to non-paired attention layers
```

Historical context: PR #264 added zero-initialized per-layer, per-head XSA gates for non-paired attention layers and used the loss headroom for a shorter schedule. 

### Feature: `xsa_lowering_rewrite`

Spec:

```python
FeatureSpec(
    name="xsa_lowering_rewrite",
    requires=("xsa",),
    modifies_forward=True,
)
```

Effects:

Generate XSA as:

```python
dot = (y * v).sum(-1, keepdim=True)
denom = v.square().sum(-1, keepdim=True).clamp_min(1e-8)
alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)
y = y - alpha * (dot / denom) * v
```

instead of:

```python
vn = F.normalize(v, dim=-1, eps=1e-4)
proj = (y * vn).sum(-1, keepdim=True)
alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)
y = y - alpha * proj * vn
```

Historical context: PR #317 proposed removing sparse attention gates and rewriting XSA lowering. 

### Feature: `sparse_attention_gate`

File: `modded_nanogpt/features/sparse_attention_gate.py`

Spec:

```python
FeatureSpec(
    name="sparse_attention_gate",
    owns_params=("attn_gate_bank",),
    soft_conflicts=("xsa",),
    modifies_model=True,
    modifies_forward=True,
    modifies_optimizer=True,
)
```

Effects:

```text
ctx.model.use_sparse_attention_gate = True
add attn_gate_bank parameter
add attn_gate_bank optimizer entry
forward multiplies attention output by sparse gate
```

The feature should be removable. This is the first target for ablation because PR #317 argues it may be redundant with XSA. 

### Feature: `mudd_last_layers`

File: `modded_nanogpt/features/mudd.py`

Spec:

```python
FeatureSpec(
    name="mudd_last_layers",
    owns_params=("mudd_w1", "mudd_w2", "mudd_b2"),
    modifies_model=True,
    modifies_forward=True,
    modifies_optimizer=True,
    modifies_schedule=True,
)
```

Effects:

```text
ctx.model.use_mudd = True
add mudd params
add optimizer entries for mudd_w1, mudd_w2, mudd_b2
include MUDD forward code
include MUDD residual/value routing
allow schedule to be reduced by config
```

Historical context: PR #259 added trimmed MUDD connections, extra parameters, optimizer entries, dynamic residual/value routing, and schedule reduction. 

For MVP, it is acceptable to keep current MUDD implementation intact behind a static generation switch rather than decomposing every subcomponent.

### Feature: `mtp_loss`

File: `modded_nanogpt/features/schedule.py`

Spec:

```python
FeatureSpec(
    name="mtp_loss",
    modifies_loss=True,
    modifies_schedule=True,
)
```

Effects:

```text
ctx.loss.use_mtp = True
schedule creates mtp_weights
training forward passes mtp_weights
loss uses multi-token prediction weights
```

Historical context: PR #178 added multi-token prediction and untied LM head/embed at two-thirds training, plus other minor changes. Treat this PR as multiple features, not one toggle. 

### Feature: `untie_embed_at_2_3`

Spec:

```python
FeatureSpec(
    name="untie_embed_at_2_3",
    modifies_optimizer=True,
    modifies_schedule=True,
)
```

Effects:

```text
training_schedule.split_step is set from stage boundary
optimizer starts with embed tied to lm_head
at split step, optimizer copies lm_head state to embed and marks split
```

## Code generation approach

MVP can use template rendering.

Use Jinja2 or Python string templates.

Input:

```text
BuildContext
```

Output:

```text
generated/train_<name>.py
generated/features_<name>.yaml
generated/manifest_seed.json
```

The generated train script should be a single runnable Python file, similar to current `train_gpt.py`.

It should include only code required by the enabled feature set.

For MVP, it is acceptable to copy the current `train_gpt.py` into a template and guard sections using Jinja conditionals:

```jinja2
{% if model.use_xsa %}
# xsa_alphas param init
{% endif %}
```

Generated output must not include Jinja remnants.

Generated output should include a header:

```python
# Generated by modded_nanogpt.builder.codegen
# Feature set: current_record_minus_sag_xsa_rewrite
# Source git sha: <sha>
# Generated at: <iso timestamp>
# Enabled features:
#   - xsa
#   - xsa_lowering_rewrite
# Disabled features:
#   - sparse_attention_gate
```

## Runtime branch policy

Acceptable:

```python
if cfg.use_xsa:
    # during generation or model construction
```

Avoid:

```python
if self.flags.use_xsa:
    # inside compiled forward
```

Generated final code should ideally contain only the active path.

For MVP, if some construction-time `if`s remain in `__init__`, that is fine. Hot forward path branches should be removed or minimized.

## Local manifest

Every run writes:

```text
experiments/runs/<run_id>/
  manifest.json
  features.yaml
  train_generated.py
  raw.log
  summary.json
```

Manifest schema:

```json
{
  "run_id": "current_minus_sag_xsa_rewrite_s1385_seed0",
  "project": "modded-nanogpt-feature-search",
  "feature_set": "current_record_minus_sag_xsa_rewrite",
  "enabled_features": [],
  "disabled_features": [],
  "base_preset": "current_record",

  "git": {
    "repo": "KellerJordan/modded-nanogpt",
    "sha": "...",
    "dirty": false
  },

  "generated": {
    "train_script": "train_generated.py",
    "sha256": "..."
  },

  "hardware": {
    "gpu_type": "H100",
    "n_gpus": 8,
    "world_size": 8
  },

  "software": {
    "python": "...",
    "torch": "...",
    "cuda": "...",
    "triton": "..."
  },

  "schedule": {
    "num_scheduled_iterations": 1375,
    "num_extension_iterations": 10,
    "total_steps": 1385,
    "cooldown_frac": 0.6
  },

  "metrics": {
    "train_time_ms": null,
    "step_avg_ms": null,
    "val_loss": null,
    "peak_memory_allocated_mib": null,
    "peak_memory_reserved_mib": null,
    "p_value_vs_3_28": null
  },

  "status": "pending",
  "created_at": "..."
}
```

After parsing logs, update:

```json
"status": "completed"
```

or:

```json
"status": "failed"
```

with error info.

## Log parser

Implement `modded_nanogpt/runtime/parse_logs.py`.

It should parse lines like current script emits:

```text
step:<step>/<train_steps> val_loss:<loss> train_time:<ms>ms step_avg:<ms>
peak memory allocated: <mib> MiB reserved: <mib> MiB
```

Output:

```json
{
  "final_step": 1385,
  "train_steps": 1385,
  "val_loss": 3.2791,
  "train_time_ms": 84429,
  "step_avg_ms": 60.96,
  "peak_memory_allocated_mib": 12345,
  "peak_memory_reserved_mib": 23456
}
```

## Flywheel integration

Implement backend interface, not direct hard dependency everywhere.

File: `modded_nanogpt/runtime/tracking.py`

```python
from typing import Protocol, Mapping, Any


class TrackingBackend(Protocol):
    def create_run(self, manifest: Mapping[str, Any]) -> str:
        ...

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        ...

    def upload_artifact(self, run_id: str, path: str, artifact_type: str | None = None) -> None:
        ...

    def finish(self, run_id: str, status: str) -> None:
        ...
```

Implement:

```text
NoOpBackend
LocalJsonlBackend
FlywheelBackend
```

`FlywheelBackend` can be a stub initially:

```python
class FlywheelBackend:
    def __init__(self, api_key: str | None = None, project: str | None = None):
        ...

    def create_run(self, manifest):
        raise NotImplementedError("Wire this to Flywheel SDK/API once credentials/docs are available")
```

Do not block the rest of the refactor on Flywheel.

Flywheel config in YAML:

```yaml
tracking:
  backend: flywheel
  project: modded-nanogpt-feature-search
  entity: paradigma
  tags:
    - h100x8
    - feature-search
```

## Search runner

Implement `modded_nanogpt/search/run_combo.py`.

Pseudo-flow:

```python
def main():
    args = parse_args()

    feature_set = resolve_feature_set(
        preset=args.preset,
        feature_set_file=args.feature_set,
        enable=args.enable,
        disable=args.disable,
    )

    ctx = build_context(feature_set.enabled, overrides=feature_set.overrides)

    run_dir = make_run_dir(args.run_name)

    generated_script = run_dir / "train_generated.py"
    render_train_script(ctx, generated_script)

    write_features_yaml(feature_set, run_dir / "features.yaml")
    manifest = create_initial_manifest(ctx, feature_set, generated_script)
    write_json(manifest, run_dir / "manifest.json")

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(generated_script),
    ]

    status = run_subprocess_tee(cmd, run_dir / "raw.log")

    summary = parse_log(run_dir / "raw.log")
    update_manifest(manifest, summary, status)
    write_json(summary, run_dir / "summary.json")
    write_json(manifest, run_dir / "manifest.json")

    if args.upload:
        upload_to_tracking_backend(manifest, run_dir)
```

CLI args:

```text
--preset
--feature-set
--enable, repeatable
--disable, repeatable
--run-name
--nproc-per-node
--data-path
--upload
--tracking-backend
--dry-run
```

`--dry-run` should build and validate but not launch GPU run.

## Candidate search generator

Implement simple generators only.

File: `modded_nanogpt/search/candidate_space.py`.

Functions:

```python
def leave_one_out(base_features: set[str], candidates: list[str]) -> list[FeatureSet]:
    ...

def pairwise_toggles(base_features: set[str], pairs: list[tuple[str, str]]) -> list[FeatureSet]:
    ...

def enable_one(base_features: set[str], candidates: list[str]) -> list[FeatureSet]:
    ...
```

Initial candidate list:

```yaml
leave_one_out:
  - sparse_attention_gate
  - xsa
  - mudd_last_layers
  - bigram_sign_trick
  - bigram_dim_192
  - residual_slice_bigram_injection
  - sparse_bigram_comms

pairwise:
  - [xsa, sparse_attention_gate]
  - [xsa, xsa_lowering_rewrite]
  - [mudd_last_layers, xsa]
  - [bigram_sign_trick, bigram_dim_192]
  - [mtp_loss, batch_size_schedule]
  - [untie_embed_at_2_3, bigram_hash]
```

## Acceptance criteria for MVP

### A. Build and validate current record

Command:

```bash
python -m modded_nanogpt.builder.codegen \
  --preset current_record \
  --out generated/train_current_record.py
```

Must:

```text
write generated/train_current_record.py
write generated/train_current_record.features.yaml
write generated/train_current_record.manifest.json
validate all dependencies
validate optimizer param table consistency
```

### B. Dry-run ablation

Command:

```bash
python -m modded_nanogpt.search.run_combo \
  --preset current_record \
  --disable sparse_attention_gate \
  --enable xsa_lowering_rewrite \
  --run-name dryrun_current_minus_sag_xsa_rewrite \
  --nproc-per-node 8 \
  --dry-run
```

Must:

```text
resolve features
reject invalid configs
generate run directory
generate train script
write manifest
not call torchrun
```

### C. Invalid dependency test

This config:

```yaml
name: invalid_sign_only
enable:
  - bigram_sign_trick
```

Must fail with:

```text
bigram_sign_trick requires bigram_hash
```

### D. Optimizer consistency test

If a feature creates a parameter but fails to add optimizer config, validation must fail before GPU launch.

Example failure:

```text
Parameter xsa_alphas exists but is missing from optimizer param_table
```

### E. Local tracking always works

Even with no Flywheel credentials, every run must write:

```text
manifest.json
features.yaml
train_generated.py
raw.log, unless dry-run
summary.json, unless dry-run
```

### F. Flywheel does not affect training

If Flywheel upload fails after training, mark upload failure in manifest but do not mark the training run itself as failed.

## Unit tests

Add tests under:

```text
tests/test_feature_validation.py
tests/test_build_context.py
tests/test_codegen.py
tests/test_manifest.py
tests/test_log_parser.py
```

Required tests:

```python
def test_sign_trick_requires_bigram_hash():
    ...

def test_xsa_lowering_requires_xsa():
    ...

def test_sparse_attention_gate_soft_conflicts_with_xsa():
    ...

def test_disable_sparse_attention_gate_removes_attn_gate_bank():
    ...

def test_enable_xsa_adds_xsa_alphas_optimizer_entry():
    ...

def test_generated_manifest_has_code_hash():
    ...

def test_parse_final_val_loss_from_log():
    ...
```

## Implementation phases

### Phase 1: Infrastructure only

Deliver:

```text
FeatureSpec
BuildContext
registry
YAML loader
dependency validation
manifest writer
log parser
NoOp tracking backend
CLI dry-run
```

No model refactor yet.

### Phase 2: Template current script

Deliver:

```text
Jinja template based on current train_gpt.py
current_record preset
generated train_current_record.py
hashing and artifacts
```

Generated script should be behavior-equivalent to current script when all current features are enabled.

### Phase 3: First real toggles

Implement these toggles:

```text
sparse_attention_gate
xsa
xsa_lowering_rewrite
bigram_sign_trick
bigram_dim_192
bigram_vocab_15x
residual_slice_bigram_injection
```

Target first ablation:

```bash
python -m modded_nanogpt.search.run_combo \
  --preset current_record \
  --disable sparse_attention_gate \
  --enable xsa_lowering_rewrite \
  --run-name current_minus_sag_xsa_rewrite
```

### Phase 4: MUDD and schedule features

Implement:

```text
mudd_last_layers
mtp_loss
untie_embed_at_2_3
batch_size_schedule
max_seq_len_schedule
yarn_window_schedule
```

### Phase 5: Flywheel adapter

Once Flywheel API details are available, wire:

```text
create run
log config
log scalar metrics
upload artifacts
finish run
```

Required artifacts to upload:

```text
manifest.json
features.yaml
train_generated.py
raw.log
summary.json
```

## Coding style requirements

Prefer clear dataclasses and validation errors.

Avoid magical global state except the feature registry.

Keep feature application deterministic.

Generated files should be readable.

Use explicit names like:

```text
bigram_sign_trick
sparse_attention_gate
xsa_lowering_rewrite
mudd_last_layers
```

not abbreviations like:

```text
bst
sag2
xsa2
muddx
```

## Important design warning

Do not make PRs the primitive.

Use this:

```text
PR preset = collection of features
```

not this:

```text
PR = feature
```

Reason: PR #178 bundled MTP, untie embed/head, optimizer tweak, init change, and LR schedule change. PR #299 bundled sign trick, bigram dimension change, vocab multiplier, sliced injection, and buffer broadcast. PR #259 bundled MUDD architecture, optimizer additions, routing changes, and schedule reduction. These need to be searchable below the PR level.    

## First PR the coding agent should open

Title:

```text
Add feature-set builder, manifest logging, and static train script generation
```

Scope:

```text
No ML behavior changes
No actual feature ablations yet
Add BuildContext, FeatureSpec, registry, YAML feature-set loading
Add current_record preset
Add codegen that can emit train_current_record.py
Add local manifest/log parser/tracking stubs
Add dry-run CLI
Add tests for dependency validation and manifest generation
```

Definition of done:

```text
python -m modded_nanogpt.builder.codegen --preset current_record --out generated/train_current_record.py
python -m modded_nanogpt.search.run_combo --preset current_record --disable sparse_attention_gate --dry-run
pytest tests/test_feature_validation.py tests/test_manifest.py tests/test_log_parser.py
```

The first PR should not try to prove a new record. It should create the machinery that makes record-search sane.
