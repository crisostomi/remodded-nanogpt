#!/usr/bin/env python3
"""Derive the codegen Jinja template from the vendored baseline train script.

The template (``src/nano/builder/templates/train_gpt.py.j2``) is produced by
applying a small, audited set of surgical replacements to ``train_gpt.py`` so
that every line *not* touched by a toggle stays byte-identical to the vendored
record. Re-run this whenever the baseline is re-vendored::

    python baseline/build_template.py

Each exact replacement asserts it matched exactly once, so a baseline change
that moves a guarded region fails loudly instead of silently mis-templating.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASELINE = HERE / "train_gpt.py"
TEMPLATE = REPO / "src" / "nano" / "builder" / "templates" / "train_gpt.py.j2"


def guard(flag: str, body: str, else_body: str | None = None) -> str:
    out = f"{{% if {flag} %}}\n{body}\n"
    if else_body is not None:
        out += f"{{% else %}}\n{else_body}\n"
    out += "{% endif %}"
    return out


# (exact_old, new) -- each old must occur exactly once.
EXACT: list[tuple[str, str]] = [
    # ---- schedule / hyperparameters values from context ----
    (
        "    num_scheduled_iterations: int = 1375  # number of steps to complete lr and ws schedule",
        "    num_scheduled_iterations: int = {{ schedule.num_scheduled_iterations }}  # number of steps to complete lr and ws schedule",
    ),
    (
        "    num_extension_iterations: int = 10  # number of steps to continue training at final lr and ws",
        "    num_extension_iterations: int = {{ schedule.num_extension_iterations }}  # number of steps to continue training at final lr and ws",
    ),
    (
        "    bigram_vocab_size: int = 50304 * 15",
        "    bigram_vocab_size: int = {{ model.bigram_vocab_size }}",
    ),
    (
        "    bigram_dim: int = 192",
        "    bigram_dim: int = {{ model.bigram_dim }}",
    ),
    (
        "    bigram_sign_table_rows: int = 8192  # prefer a power of 2 (values ~500-15000 gave similar results)",
        guard(
            "model.use_bigram_sign_trick",
            "    bigram_sign_table_rows: int = {{ model.bigram_sign_table_rows }}  # prefer a power of 2 (values ~500-15000 gave similar results)",
        ),
    ),
    (
        "training_schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations, cooldown_frac=0.60)",
        "training_schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations, cooldown_frac={{ schedule.cooldown_frac }})",
    ),
    # ---- fp8 lm head ----
    (
        '        use_fp8 = not os.environ.get("DISABLE_FP8", False)',
        guard(
            "model.use_fp8_lm_head",
            '        use_fp8 = not os.environ.get("DISABLE_FP8", False)',
            "        use_fp8 = False",
        ),
    ),
    # ---- MUDD construction (structural; guarded for table consistency) ----
    (
        "        self.init_mudd(num_layers, model_dim)",
        guard("model.use_mudd", "        self.init_mudd(num_layers, model_dim)"),
    ),
    # ---- value embeddings / gates ----
    (
        "        self.value_embeds = nn.Parameter(0.01 * torch.randn(5 * self.vocab_size, model_dim, dtype=torch.bfloat16))",
        guard(
            "model.use_value_embeds",
            "        self.value_embeds = nn.Parameter(0.01 * torch.randn(5 * self.vocab_size, model_dim, dtype=torch.bfloat16))",
        ),
    ),
    (
        "        self.attn_gate_bank = nn.Parameter(torch.zeros(10, num_heads, 12)) # 10 layers",
        guard(
            "model.use_sparse_attention_gate",
            "        self.attn_gate_bank = nn.Parameter(torch.zeros(10, num_heads, 12)) # 10 layers",
        ),
    ),
    (
        "        self.ve_gate_bank = nn.Parameter(torch.zeros(5, num_heads, 12)) # 5 unique gates",
        guard(
            "model.use_value_embed_gates",
            "        self.ve_gate_bank = nn.Parameter(torch.zeros(5, num_heads, 12)) # 5 unique gates",
        ),
    ),
    # ---- smear / skip gates ----
    (
        "        self.smear_gate = nn.Linear(12, 1, bias=False)\n        nn.init.zeros_(self.smear_gate.weight)",
        guard(
            "model.use_smear",
            "        self.smear_gate = nn.Linear(12, 1, bias=False)\n        nn.init.zeros_(self.smear_gate.weight)",
        ),
    ),
    (
        "        self.skip_gate = nn.Linear(12, 1, bias=False)\n        nn.init.zeros_(self.skip_gate.weight)",
        guard(
            "model.use_skip_gate",
            "        self.skip_gate = nn.Linear(12, 1, bias=False)\n        nn.init.zeros_(self.skip_gate.weight)",
        ),
    ),
    # ---- bigram embed + sign table buffer ----
    (
        "        self.bigram_embed = nn.Embedding(args.bigram_vocab_size, args.bigram_dim)\n"
        "        nn.init.zeros_(self.bigram_embed.weight)\n"
        "        bigram_sign_table = torch.randn(args.bigram_sign_table_rows, args.bigram_dim).sign().to(torch.bfloat16)\n"
        "        self.register_buffer('bigram_sign_table', bigram_sign_table)",
        guard(
            "model.use_bigram_hash",
            "        self.bigram_embed = nn.Embedding(args.bigram_vocab_size, args.bigram_dim)\n"
            "        nn.init.zeros_(self.bigram_embed.weight)",
        )
        + "\n"
        + guard(
            "model.use_bigram_sign_trick",
            "        bigram_sign_table = torch.randn(args.bigram_sign_table_rows, args.bigram_dim).sign().to(torch.bfloat16)\n"
            "        self.register_buffer('bigram_sign_table', bigram_sign_table)",
        ),
    ),
    (
        "        # Per-layer injection coefficients for x0 and bigram\n"
        "        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))\n"
        "        self.bigram_lambdas = nn.Parameter(0.05 * torch.ones(num_layers))",
        guard(
            "model.use_bigram_hash",
            "        # Per-layer injection coefficients for x0 and bigram\n"
            "        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))\n"
            "        self.bigram_lambdas = nn.Parameter(0.05 * torch.ones(num_layers))",
        ),
    ),
    # ---- xsa_alphas parameter ----
    (
        "        # Per-(layer, head) learnable XSA gate; zero-init -> tanh(0)=0 disables XSA at step 0\n"
        "        self.xsa_alphas = nn.Parameter(torch.zeros(num_layers, self.num_heads))",
        guard(
            "model.use_xsa",
            "        # Per-(layer, head) learnable XSA gate; zero-init -> tanh(0)=0 disables XSA at step 0\n"
            "        self.xsa_alphas = nn.Parameter(torch.zeros(num_layers, self.num_heads))",
        ),
    ),
    # ---- forward: attn-gate / ve-gate unbind ----
    (
        "        ag = self.attn_gate_bank.unbind(0)\n"
        "        veg = self.ve_gate_bank.unbind(0)\n"
        "        attn_gates = [*ag[:6], None, *ag[6:]]",
        guard("model.use_sparse_attention_gate", "        ag = self.attn_gate_bank.unbind(0)")
        + "\n        veg = self.ve_gate_bank.unbind(0)\n"
        + guard(
            "model.use_sparse_attention_gate",
            "        attn_gates = [*ag[:6], None, *ag[6:]]",
            "        attn_gates = [None] * self.num_layers",
        ),
    ),
    # ---- forward: xsa_alphas unbind ----
    (
        "        # XSA on non-paired attn layers only; paired {0,2,5,9} and MLP-only layer 6 skipped\n"
        "        xsa_alpha_per_layer = self.xsa_alphas.unbind(0)\n"
        "        xsa_alphas = [xsa_alpha_per_layer[j] if j in {1, 3, 4, 7, 8, 10} else None for j in range(self.num_layers)]",
        guard(
            "model.use_xsa",
            "        # XSA on non-paired attn layers only; paired {0,2,5,9} and MLP-only layer 6 skipped\n"
            "        xsa_alpha_per_layer = self.xsa_alphas.unbind(0)\n"
            "        xsa_alphas = [xsa_alpha_per_layer[j] if j in {1, 3, 4, 7, 8, 10} else None for j in range(self.num_layers)]",
            "        xsa_alphas = [None] * self.num_layers",
        ),
    ),
    # ---- forward: bigram sign-trick ----
    (
        "        # Use sign-trick to better compress multiple bigrams into a shared bigram embedding row\n"
        "        # (details in https://github.com/KellerJordan/modded-nanogpt/pull/299 by @trianxy)\n"
        "        sign_idx = torch.zeros_like(input_seq)\n"
        "        sign_idx[1:] = (input_seq[:-1] ^ input_seq[1:]) % self.bigram_sign_table.shape[0]  # (8192,)\n"
        "        bigram_signs = self.bigram_sign_table[sign_idx]                                    # (seq, bigram_dim)\n"
        "        x0_bigram = (self.bigram_embed(bigram_input_seq) * bigram_signs)[None]             # (1, seq, bigram_dim)",
        guard(
            "model.use_bigram_sign_trick",
            "        # Use sign-trick to better compress multiple bigrams into a shared bigram embedding row\n"
            "        # (details in https://github.com/KellerJordan/modded-nanogpt/pull/299 by @trianxy)\n"
            "        sign_idx = torch.zeros_like(input_seq)\n"
            "        sign_idx[1:] = (input_seq[:-1] ^ input_seq[1:]) % self.bigram_sign_table.shape[0]  # (8192,)\n"
            "        bigram_signs = self.bigram_sign_table[sign_idx]                                    # (seq, bigram_dim)\n"
            "        x0_bigram = (self.bigram_embed(bigram_input_seq) * bigram_signs)[None]             # (1, seq, bigram_dim)",
            "        x0_bigram = self.bigram_embed(bigram_input_seq)[None]",
        ),
    ),
    # ---- forward: sliced residual injection (3 sites) ----
    (
        "        # Initialize residual stream with pre-layer-0 bigram injection\n"
        "        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + x0_bigram * bigram_lambdas[0]",
        "        # Initialize residual stream with pre-layer-0 bigram injection\n"
        + guard(
            "model.use_residual_slice_bigram_injection",
            "        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + x0_bigram * bigram_lambdas[0]",
            "        x = x + x0_bigram * bigram_lambdas[0]",
        ),
    ),
    (
        "                    x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + mu[11] * x0_bigram",
        guard(
            "model.use_residual_slice_bigram_injection",
            "                    x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + mu[11] * x0_bigram",
            "                    x = x + mu[11] * x0_bigram",
        ),
    ),
    (
        "                    if bg_inject[i] is not None:\n"
        "                        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + bg_inject[i]",
        "                    if bg_inject[i] is not None:\n"
        + guard(
            "model.use_residual_slice_bigram_injection",
            "                        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + bg_inject[i]",
            "                        x = x + bg_inject[i]",
        ),
    ),
    # ---- attention forward: XSA lowering + sparse attention gate ----
    (
        "        # Gated XSA (arXiv:2603.09078) with learnable strength: subtract per-head fraction tanh(α)\n"
        "        # of y aligned with v̂. Non-paired only (v shape doesn't line up for paired layers).\n"
        "        if attn_args.xsa_alpha is not None and not self.paired:\n"
        "            vn = F.normalize(v, dim=-1, eps=1e-4)\n"
        "            proj = (y * vn).sum(-1, keepdim=True)\n"
        "            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)\n"
        "            y = y - alpha * proj * vn\n"
        "        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)",
        guard(
            "model.use_xsa",
            "        # Gated XSA (arXiv:2603.09078) with learnable strength: subtract per-head fraction tanh(α)\n"
            "        # of y aligned with v̂. Non-paired only (v shape doesn't line up for paired layers).\n"
            "        if attn_args.xsa_alpha is not None and not self.paired:\n"
            + guard(
                "model.use_xsa_lowering_rewrite",
                "            dot = (y * v).sum(-1, keepdim=True)\n"
                "            denom = v.square().sum(-1, keepdim=True).clamp_min(1e-8)\n"
                "            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)\n"
                "            y = y - alpha * (dot / denom) * v",
                "            vn = F.normalize(v, dim=-1, eps=1e-4)\n"
                "            proj = (y * vn).sum(-1, keepdim=True)\n"
                "            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)\n"
                "            y = y - alpha * proj * vn",
            ),
        )
        + "\n"
        + guard(
            "model.use_sparse_attention_gate",
            "        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)",
        ),
    ),
    # ---- bf16 casts at module setup ----
    (
        "model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()\n"
        "model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()\n"
        "model.qk_bank.data = model.qk_bank.data.bfloat16()\n"
        "model.vo_bank.data = model.vo_bank.data.bfloat16()\n"
        "model.mlp_bank.data = model.mlp_bank.data.bfloat16()\n"
        "model.mudd_w1.data = model.mudd_w1.data.bfloat16()\n"
        "model.mudd_w2.data = model.mudd_w2.data.bfloat16()\n"
        "model.mudd_b2.data = model.mudd_b2.data.bfloat16()",
        guard("model.use_sparse_attention_gate", "model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()")
        + "\n"
        + guard("model.use_value_embed_gates", "model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()")
        + "\n"
        "model.qk_bank.data = model.qk_bank.data.bfloat16()\n"
        "model.vo_bank.data = model.vo_bank.data.bfloat16()\n"
        "model.mlp_bank.data = model.mlp_bank.data.bfloat16()\n"
        + guard(
            "model.use_mudd",
            "model.mudd_w1.data = model.mudd_w1.data.bfloat16()\n"
            "model.mudd_w2.data = model.mudd_w2.data.bfloat16()\n"
            "model.mudd_b2.data = model.mudd_b2.data.bfloat16()",
        ),
    ),
    # ---- bigram sign-table broadcast ----
    (
        "dist.broadcast(model.bigram_sign_table, 0)  # buffer, not in parameters()",
        guard(
            "model.use_bigram_sign_trick",
            "dist.broadcast(model.bigram_sign_table, 0)  # buffer, not in parameters()",
        ),
    ),
    # ---- partial key offset (induction key shift on long-window layers) ----
    (
        "            if key_offset:\n"
        "                # shift keys forward for the stationary head dims. Enables 1-layer induction.\n"
        "                k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]",
        guard(
            "model.use_partial_key_offset",
            "            if key_offset:\n"
            "                # shift keys forward for the stationary head dims. Enables 1-layer induction.\n"
            "                k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]",
        ),
    ),
    # ---- paired-head attention layers ----
    (
        "        self.paired_head_layers = [0, 2, 5, 9]",
        guard(
            "model.use_paired_head_attention",
            "        self.paired_head_layers = [0, 2, 5, 9]",
            "        self.paired_head_layers = []",
        ),
    ),
    # ---- smear token embedding forward one position ----
    (
        "        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))\n"
        "        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])",
        guard(
            "model.use_smear",
            "        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))\n"
            "        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])",
        ),
    ),
    # ---- learned skip connection into the attention-free layer 6 ----
    (
        "        skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(self.skip_gate(x0[..., :self.skip_gate.weight.size(-1)]))",
        guard(
            "model.use_skip_gate",
            "        skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(self.skip_gate(x0[..., :self.skip_gate.weight.size(-1)]))",
        ),
    ),
    (
        "            if i == 6:\n"
        "                x = x + skip_gate_out * cache[3]",
        "            if i == 6:\n"
        + guard(
            "model.use_skip_gate",
            "                x = x + skip_gate_out * cache[3]",
            "                pass",
        ),
    ),
    # ---- Adam every-other-step cadence ----
    (
        '    def _is_adam_step(self, step: int):\n'
        '        """Adam params are only updated on odd steps."""\n'
        "        return step % 2 == 1",
        '    def _is_adam_step(self, step: int):\n'
        '        """Adam params are only updated on odd steps."""\n'
        + guard(
            "optim.use_adam_every_other_step",
            "        return step % 2 == 1",
            "        return True",
        ),
    ),
    # ---- tunable optimizer defaults (rendered from BuildContext) ----
    (
        "        adam_defaults = dict(\n"
        "            lr=0.008,\n"
        "            eps=1e-10,\n"
        "            weight_decay=0.005,\n"
        "        )",
        "{{ adam_defaults_block }}",
    ),
    (
        "        normuon_defaults = dict(\n"
        "            lr=0.023,\n"
        "            momentum=0.95,\n"
        "            beta2=0.9,\n"
        "            weight_decay=1.2,\n"
        "        )",
        "{{ normuon_defaults_block }}",
    ),
]


def replace_region(text: str, start: str, end: str, placeholder: str) -> str:
    si = text.index(start)
    ei = text.index(end, si)
    return text[:si] + placeholder + text[ei:]


def main() -> int:
    text = BASELINE.read_text(encoding="utf-8")

    # Region replacements run FIRST: param_table dict (+ MUDD update) and
    # work_order list. The work_order region's end-anchor is the adam_defaults
    # dict, which the exact pass below rewrites -- so regions must go first.
    text = replace_region(
        text,
        "        self.param_table = {\n",
        "\n        # - Process smaller/faster params first",
        "{{ param_table_block }}",
    )
    text = replace_region(
        text,
        "        self.work_order = [\n",
        "\n\n        adam_defaults = dict(",
        "{{ work_order_block }}",
    )

    for old, new in EXACT:
        count = text.count(old)
        if count != 1:
            sys.stderr.write(
                f"ERROR: expected exactly 1 match, found {count} for:\n{old[:120]!r}\n"
            )
            return 1
        text = text.replace(old, new)

    # Header comment, logged with the rest of the source for auditability.
    text = "{{ header_comment }}\n\n" + text

    TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE.write_text(text, encoding="utf-8")
    sys.stderr.write(f"Wrote {TEMPLATE} ({text.count(chr(10)) + 1} lines)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
