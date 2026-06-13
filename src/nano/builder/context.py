"""The ``BuildContext`` and its sub-configs.

Features mutate a ``BuildContext``; the renderer turns a fully-applied context
into a static training script. The context is the single source of truth that
the Jinja template reads from -- every ``{{ ... }}`` and ``{% if ... %}`` in
``templates/train_gpt.py.j2`` resolves against fields defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Baseline backbone (vendored from modded-nanogpt @ baseline/SOURCE_SHA.txt).
#
# The transformer backbone params below are *always* present regardless of the
# feature set; toggleable, feature-owned entries (attn_gate_bank, xsa_alphas,
# bigram_embed, mudd_*, ...) are seeded here too and pruned by the builder when
# their owning feature is disabled. Insertion order is preserved exactly so the
# generated ``scatter_order = list(param_table)`` matches the record script.
# ---------------------------------------------------------------------------

#: Params with no owning feature: the transformer backbone, always present.
CORE_PARAMS: tuple[str, ...] = (
    "qk_bank",
    "vo_bank",
    "mlp_bank",
    "scalars",
    "lm_head",
    "embed",
    "post_lambdas",
    "resid_lambdas",
)

#: Baseline param table in record order (dict-of-rows, consumed verbatim by the
#: generated ``NorMuonAndAdam``). ``bigram_embed`` is seeded with the dense
#: ``"sharded"`` comms; the ``sparse_bigram_comms`` feature upgrades it.
BASELINE_PARAM_TABLE: dict[str, dict[str, Any]] = {
    "qk_bank":        {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
    "vo_bank":        {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
    "mlp_bank":       {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
    "scalars":        {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 5.0,  "wd_mul": 0.0},
    "smear_gate":     {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.01, "wd_mul": 0.0},
    "skip_gate":      {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.05, "wd_mul": 0.0},
    "attn_gate_bank": {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99]},
    "ve_gate_bank":   {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99]},
    "lm_head":        {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.0},
    "bigram_embed":   {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.75, 0.95], "lr_mul": 75.0, "wd_mul": 5.0},
    "post_lambdas":   {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
    "x0_lambdas":     {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
    "bigram_lambdas": {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
    "resid_lambdas":  {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.95], "lr_mul": 5.0,  "wd_mul": 0.0},
    "xsa_alphas":     {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
    "value_embeds":   {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.75, 0.95], "lr_mul": 75.0, "wd_mul": 5.0},
    "embed":          {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.0},
    # MUDD overrides (kept as a trailing group, matching the record script).
    "mudd_w1":        {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.25},
    "mudd_w2":        {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.25},
    "mudd_b2":        {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.25, "wd_mul": 0.0},
}

#: Labels that belong to the trailing ``.update({...})`` MUDD block.
MUDD_PARAMS: tuple[str, ...] = ("mudd_w1", "mudd_w2", "mudd_b2")

#: Baseline ``work_order`` (update/gather scheduling order).
BASELINE_WORK_ORDER: list[str] = [
    "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank", "mudd_b2", "xsa_alphas",
    "post_lambdas", "x0_lambdas", "bigram_lambdas", "resid_lambdas",
    "mudd_w2",
    "value_embeds", "bigram_embed",
    "mudd_w1",
    "lm_head", "embed",
    "qk_bank", "vo_bank", "mlp_bank",
]

ADAM_DEFAULTS: dict[str, Any] = dict(lr=0.008, eps=1e-10, weight_decay=0.005)
NORMUON_DEFAULTS: dict[str, Any] = dict(lr=0.023, momentum=0.95, beta2=0.9, weight_decay=1.2)

#: Baseline training curriculum (rendered into ``TRAINING_STAGES``). Each stage's
#: numeric genes -- duration, batch size, window sizes, max seq len, LR mul and
#: the MTP-weight schedule -- are searchable via a ``schedule.training_stages``
#: override. The final (extension) stage has no ``duration``.
BASELINE_TRAINING_STAGES: list[dict[str, Any]] = [
    {"duration": 1 / 3, "train_max_seq_len": 896,  "batch_size": 8 * 2048 * 8,  "window_sizes": [1, 3],
     "lr_mul": 1.0,  "mtp_weights_start": [1.0, 0.5, 0.25], "mtp_weights_end": [1.0, 0.5, 0.0]},
    {"duration": 1 / 3, "train_max_seq_len": 2048, "batch_size": 16 * 2048 * 8, "window_sizes": [3, 7],
     "lr_mul": 1.52, "mtp_weights_start": [1.0, 0.5],       "mtp_weights_end": [1.0, 0.0]},
    {"duration": 1 / 3, "train_max_seq_len": 2048, "batch_size": 24 * 2048 * 8, "window_sizes": [5, 11],
     "lr_mul": 1.73, "mtp_weights_start": [1.0],             "mtp_weights_end": [1.0]},
    {"duration": None,  "train_max_seq_len": 2048, "batch_size": 24 * 2048 * 8, "window_sizes": [6, 13],
     "lr_mul": 1.0,  "mtp_weights_start": [1.0],             "mtp_weights_end": [1.0]},
]


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    num_layers: int = 11
    num_heads: int = 6
    head_dim: int = 128
    model_dim: int = 768

    # precision / lm head
    use_fp8_lm_head: bool = False

    # attention sparsity
    use_xsa: bool = False
    use_xsa_lowering_rewrite: bool = False
    use_sparse_attention_gate: bool = False
    use_paired_head_attention: bool = False
    use_partial_key_offset: bool = False

    # value embeddings / misc gates
    use_value_embeds: bool = False
    use_value_embed_gates: bool = False
    use_smear: bool = False
    use_skip_gate: bool = False

    # dynamic dense connections
    use_mudd: bool = False

    # bigram-hash family
    use_bigram_hash: bool = False
    use_bigram_sign_trick: bool = False
    use_residual_slice_bigram_injection: bool = False
    use_sparse_bigram_comms: bool = False

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

    use_mtp_schedule: bool = False
    use_yarn_window_schedule: bool = False
    use_batch_size_schedule: bool = False
    use_max_seq_len_schedule: bool = False
    use_untie_embed: bool = False

    training_stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return self.num_scheduled_iterations + self.num_extension_iterations


@dataclass
class OptimConfig:
    param_table: dict[str, dict[str, Any]] = field(default_factory=dict)
    work_order: list[str] = field(default_factory=list)
    scatter_order: list[str] = field(default_factory=list)
    adam_defaults: dict[str, Any] = field(default_factory=dict)
    normuon_defaults: dict[str, Any] = field(default_factory=dict)

    # structural optimizer features (always-on in the MVP, recorded for the manifest)
    use_normuon: bool = False
    use_polar_express: bool = False
    use_cautious_weight_decay: bool = False
    use_adam_every_other_step: bool = False


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

    # ---- non-template warnings collected during build (e.g. soft conflicts) ----
    warnings: list[str] = field(default_factory=list)

    def seed_baseline(self) -> "BuildContext":
        """Seed the always-present backbone param table / defaults.

        Toggleable feature-owned entries are seeded too and pruned later by
        :func:`nano.builder.render.prune_disabled_params` once the enabled set
        is known. Returns ``self`` for chaining.
        """
        import copy

        self.optim.param_table = copy.deepcopy(BASELINE_PARAM_TABLE)
        self.optim.work_order = list(BASELINE_WORK_ORDER)
        self.optim.adam_defaults = dict(ADAM_DEFAULTS)
        self.optim.normuon_defaults = dict(NORMUON_DEFAULTS)
        self.optim.scatter_order = list(self.optim.param_table)
        self.schedule.training_stages = copy.deepcopy(BASELINE_TRAINING_STAGES)
        return self
