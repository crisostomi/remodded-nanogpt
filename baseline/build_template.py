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


# The MUDD method definitions (init_mudd + forward_mudd), guarded as a block so
# the off-state drops the dead code (and its docstrings) entirely. Triple-quoted
# so the embedded docstring """ does not need escaping; must match the baseline
# byte-for-byte (asserted exactly-once by the EXACT pass).
MUDD_METHODS = (
'''    def init_mudd(self, num_layers: int, model_dim: int):
        """
        Multiway Dynamic Dense Connections @lishengping. https://arxiv.org/abs/2502.12170
        Expressive and efficient mechanism for data dependent skip connections.
        Given current activation x, return n skip coefficients computed via ~mlp(x).
        Trimmed for speedrun: invoked at start of last layer and post-loop only.

        Start of last layer produces 14 coefficients:
          mu[0..2]  = v_mudd source coefs  (cache[0], cache[7], x)   -> added into V
          mu[3..5]  = residual source coefs (cache[0], cache[7], x)  -> residual recombination
          mu[6..7]  = per-pair ve_gate (2 channels, tiled to num_heads)
          mu[8..9]  = resid_attn / post_attn lambdas (dynamic)
          mu[10..11]= x0 / bigram injection lambdas (dynamic)
          mu[12..13]= resid_mlp / post_mlp lambdas (dynamic)

        Post-loop produces 5 residual coefs over
          {cache[0], cache[7], cache[9], ve_bank0, cache[3]}.
        """
        num_mudd_layers = 2
        self._mudd_scale = 0.1
        mudd_dim = 64
        max_num_coef = 14

        self.mudd_w1 = nn.Parameter(torch.empty(num_mudd_layers, mudd_dim, model_dim))
        for j in range(num_mudd_layers):
            nn.init.kaiming_uniform_(self.mudd_w1.data[j], a=math.sqrt(5))

        self.mudd_w2 = nn.Parameter(torch.zeros(num_mudd_layers, max_num_coef, mudd_dim))

        # Bias init in pre-scaled domain (effective = bias * _mudd_scale).
        bs_init = torch.zeros(num_mudd_layers, max_num_coef)
        # Per-pair ve_gate baseline (matches max of `2*sigmoid` used at other layers):
        bs_init[0, 6]  = 2.0 / self._mudd_scale       # ve_gate lane 0
        bs_init[0, 7]  = 2.0 / self._mudd_scale       # ve_gate lane 1
        # Layer-0 layer-10 dynamic lambdas (effective values match per-layer defaults):
        bs_init[0, 8]  = 1.1**0.5 / self._mudd_scale  # resid_attn[10]
        bs_init[0, 9]  = 1.0 / self._mudd_scale       # post_attn[10]
        bs_init[0, 10] = 0.0                          # x0_lambda[10] (init 0)
        bs_init[0, 11] = 0.05 / self._mudd_scale      # bigram_lambda[10]
        bs_init[0, 12] = 1.1**0.5 / self._mudd_scale  # resid_mlp[10]
        bs_init[0, 13] = 1.0 / self._mudd_scale       # post_mlp[10]
        # Layer-1 (post-loop): -0.5 backout absorbed into residual h7 coef.
        bs_init[1, 1]  = -0.5 / self._mudd_scale      # post-loop residual h7 coef
        self.mudd_b2 = nn.Parameter(bs_init)

    def forward_mudd(self, x, id, num_coef):
        """Returns `num_coef` per-token MUDD coefficients from block `id` (0 or 1)."""
        x = F.gelu(F.linear(x, self.mudd_w1[id]))
        x = (F.linear(x, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        return x.split(1, dim=-1)'''
)


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
        guard(
            "model.use_bigram_hash",
            "    bigram_vocab_size: int = {{ model.bigram_vocab_size }}",
            "    bigram_vocab_size: int = 50304 * 15",
        ),
    ),
    (
        "    bigram_dim: int = 192",
        guard(
            "model.use_bigram_hash",
            "    bigram_dim: int = {{ model.bigram_dim }}",
            "    bigram_dim: int = 192",
        ),
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
    # ---- MUDD method definitions (dropped wholesale when MUDD is off) ----
    (MUDD_METHODS, guard("model.use_mudd", MUDD_METHODS)),
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
        + "\n"
        + guard("model.use_value_embed_gates", "        veg = self.ve_gate_bank.unbind(0)")
        + "\n"
        + guard(
            "model.use_sparse_attention_gate",
            "        attn_gates = [*ag[:6], None, *ag[6:]]",
            "        attn_gates = [None] * self.num_layers",
        ),
    ),
    # ---- forward: ve_gates list + its length assert (value_embed_gates) ----
    (
        "        ve_gates = [None, veg[0], veg[1], *self.gate_filler_nones, veg[2], veg[3], veg[4]]",
        guard(
            "model.use_value_embed_gates",
            "        ve_gates = [None, veg[0], veg[1], *self.gate_filler_nones, veg[2], veg[3], veg[4]]",
        ),
    ),
    (
        "        assert len(ve_gates) == self.num_layers",
        guard("model.use_value_embed_gates", "        assert len(ve_gates) == self.num_layers"),
    ),
    # ---- forward: value-embed table gather + per-layer list (value_embeds) ----
    (
        "        # Value embeddings - always computed (not precomputed)\n"
        "        ve = self.value_embeds.view(5, self.vocab_size, -1)[:, input_seq]\n"
        "        # Shifted .01 ... 234 structure on token value embeddings by @photomz\n"
        "        ve = [None, ve[0], ve[1], *self.gate_filler_nones, ve[2], ve[3], ve[4]]\n"
        "        assert len(ve) == self.num_layers",
        guard(
            "model.use_value_embeds",
            "        # Value embeddings - always computed (not precomputed)\n"
            "        ve = self.value_embeds.view(5, self.vocab_size, -1)[:, input_seq]\n"
            "        # Shifted .01 ... 234 structure on token value embeddings by @photomz\n"
            "        ve = [None, ve[0], ve[1], *self.gate_filler_nones, ve[2], ve[3], ve[4]]\n"
            "        assert len(ve) == self.num_layers",
        ),
    ),
    # ---- forward: last-layer MUDD branch + value-embed aux_v dispatch ----
    # MUDD (use_mudd) owns the `if i == last:` arm and turns the `elif ve[i]`
    # into a plain `if ve[i]` when off. The value-embed branch (use_value_embeds)
    # wraps the whole dispatch; its gate body (use_value_embed_gates) falls back
    # to ungated value embeds. MUDD requires value_embeds, so the all-off arm of
    # the value-embed guard is the only branch when value embeds are dropped.
    (
        "                if i == self.num_layers - 1:\n"
        "                    cache[9] = x\n"
        "                    mu = self.forward_mudd(x, id=0, num_coef=14)\n"
        "                    v_mudd = (mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * x).view(B, T, self.num_heads, self.head_dim)\n"
        "                    x = (1 + mu[5]) * x + mu[3] * cache[0] + mu[4] * cache[7]\n"
        "                    ve_gate = torch.cat([mu[6], mu[7]], dim=-1).repeat_interleave(\n"
        "                        self.num_heads // 2, dim=-1\n"
        "                    ).unsqueeze(-1)\n"
        "                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)\n"
        "                    aux_v = (ve_gate * ve_view + v_mudd).view(B, T, -1)\n"
        "                elif ve[i] is not None:\n"
        "                    # gate pattern g(x[:6] + ve[:6]) by @photomz\n"
        "                    gate_in = torch.cat([attn_in_normed[..., :6], ve[i][None, ..., :6]], dim=-1)\n"
        "                    ve_gate_out = 2 * torch.sigmoid(F.linear(gate_in, ve_gates[i])).view(B, T, self.num_heads, 1)\n"
        "                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)\n"
        "                    aux_v = (ve_gate_out * ve_view).view(B, T, -1)\n"
        "                else:\n"
        "                    aux_v = None",
        "{% if model.use_value_embeds %}\n"
        "{% if model.use_mudd %}\n"
        "                if i == self.num_layers - 1:\n"
        "                    cache[9] = x\n"
        "                    mu = self.forward_mudd(x, id=0, num_coef=14)\n"
        "                    v_mudd = (mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * x).view(B, T, self.num_heads, self.head_dim)\n"
        "                    x = (1 + mu[5]) * x + mu[3] * cache[0] + mu[4] * cache[7]\n"
        "                    ve_gate = torch.cat([mu[6], mu[7]], dim=-1).repeat_interleave(\n"
        "                        self.num_heads // 2, dim=-1\n"
        "                    ).unsqueeze(-1)\n"
        "                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)\n"
        "                    aux_v = (ve_gate * ve_view + v_mudd).view(B, T, -1)\n"
        "                elif ve[i] is not None:\n"
        "{% else %}\n"
        "                if ve[i] is not None:\n"
        "{% endif %}\n"
        "{% if model.use_value_embed_gates %}\n"
        "                    # gate pattern g(x[:6] + ve[:6]) by @photomz\n"
        "                    gate_in = torch.cat([attn_in_normed[..., :6], ve[i][None, ..., :6]], dim=-1)\n"
        "                    ve_gate_out = 2 * torch.sigmoid(F.linear(gate_in, ve_gates[i])).view(B, T, self.num_heads, 1)\n"
        "                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)\n"
        "                    aux_v = (ve_gate_out * ve_view).view(B, T, -1)\n"
        "{% else %}\n"
        "                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)\n"
        "                    aux_v = ve_view.view(B, T, -1)\n"
        "{% endif %}\n"
        "                else:\n"
        "                    aux_v = None\n"
        "{% else %}\n"
        "                aux_v = None\n"
        "{% endif %}",
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
        "{% if model.use_bigram_hash %}\n"
        + guard(
            "model.use_bigram_sign_trick",
            "        # Use sign-trick to better compress multiple bigrams into a shared bigram embedding row\n"
            "        # (details in https://github.com/KellerJordan/modded-nanogpt/pull/299 by @trianxy)\n"
            "        sign_idx = torch.zeros_like(input_seq)\n"
            "        sign_idx[1:] = (input_seq[:-1] ^ input_seq[1:]) % self.bigram_sign_table.shape[0]  # (8192,)\n"
            "        bigram_signs = self.bigram_sign_table[sign_idx]                                    # (seq, bigram_dim)\n"
            "        x0_bigram = (self.bigram_embed(bigram_input_seq) * bigram_signs)[None]             # (1, seq, bigram_dim)",
            "        x0_bigram = self.bigram_embed(bigram_input_seq)[None]",
        )
        + "\n{% endif %}",
    ),
    # ---- forward: sliced residual injection site 1 (pre-layer-0) ----
    (
        "        # Initialize residual stream with pre-layer-0 bigram injection\n"
        "        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + x0_bigram * bigram_lambdas[0]",
        "{% if model.use_bigram_hash %}\n"
        "        # Initialize residual stream with pre-layer-0 bigram injection\n"
        + guard(
            "model.use_residual_slice_bigram_injection",
            "        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + x0_bigram * bigram_lambdas[0]",
            "        x = x + x0_bigram * bigram_lambdas[0]",
        )
        + "\n{% endif %}",
    ),
    # ---- forward: post-attention residual recombination (MUDD vs normal) ----
    # MUDD (use_mudd) owns the dynamic `if mu is not None:` recombination; off it
    # falls back to the normal residual path (de-indented). The bigram injection
    # lines nest the residual-slice guard (sites 2 & 3 of the sliced injection).
    (
        "                if mu is not None:\n"
        "                    x = mu[8] * x + mu[9] * attn_out + mu[10] * cache[0] \n"
        "                    x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + mu[11] * x0_bigram\n"
        "                else:\n"
        "                    x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0_inject[i]\n"
        "                    if bg_inject[i] is not None:\n"
        "                        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + bg_inject[i]",
        "{% if model.use_mudd %}\n"
        "                if mu is not None:\n"
        "                    x = mu[8] * x + mu[9] * attn_out + mu[10] * cache[0] \n"
        "{% if model.use_bigram_hash %}\n"
        "{% if model.use_residual_slice_bigram_injection %}\n"
        "                    x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + mu[11] * x0_bigram\n"
        "{% else %}\n"
        "                    x = x + mu[11] * x0_bigram\n"
        "{% endif %}\n"
        "{% endif %}\n"
        "                else:\n"
        "{% if model.use_bigram_hash %}\n"
        "                    x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0_inject[i]\n"
        "                    if bg_inject[i] is not None:\n"
        "{% if model.use_residual_slice_bigram_injection %}\n"
        "                        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + bg_inject[i]\n"
        "{% else %}\n"
        "                        x = x + bg_inject[i]\n"
        "{% endif %}\n"
        "{% else %}\n"
        "                    x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out\n"
        "{% endif %}\n"
        "{% else %}\n"
        "{% if model.use_bigram_hash %}\n"
        "                x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0_inject[i]\n"
        "                if bg_inject[i] is not None:\n"
        "{% if model.use_residual_slice_bigram_injection %}\n"
        "                    x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + bg_inject[i]\n"
        "{% else %}\n"
        "                    x = x + bg_inject[i]\n"
        "{% endif %}\n"
        "{% else %}\n"
        "                x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out\n"
        "{% endif %}\n"
        "{% endif %}",
    ),
    # ---- forward: bigram-owned lambda unbinds (x0_lambdas / bigram_lambdas) ----
    (
        "        x0_lambdas = self.x0_lambdas.bfloat16().unbind(0)\n"
        "        bigram_lambdas = self.bigram_lambdas.bfloat16().unbind(0)",
        guard(
            "model.use_bigram_hash",
            "        x0_lambdas = self.x0_lambdas.bfloat16().unbind(0)\n"
            "        bigram_lambdas = self.bigram_lambdas.bfloat16().unbind(0)",
        ),
    ),
    # ---- forward: x0 / bigram per-layer injection precompute ----
    (
        "        # Precompute x0/bigram injection (added to attention output each layer)\n"
        "        # Layer 0: bigram already injected above, so only x0 component\n"
        "        x0_inject = tuple(x0 * x0_lambdas[i] for i in range(self.num_layers))\n"
        "        bg_inject = (None,) + tuple(x0_bigram * bigram_lambdas[i] for i in range(1, self.num_layers))",
        guard(
            "model.use_bigram_hash",
            "        # Precompute x0/bigram injection (added to attention output each layer)\n"
            "        # Layer 0: bigram already injected above, so only x0 component\n"
            "        x0_inject = tuple(x0 * x0_lambdas[i] for i in range(self.num_layers))\n"
            "        bg_inject = (None,) + tuple(x0_bigram * bigram_lambdas[i] for i in range(1, self.num_layers))",
        ),
    ),
    # ---- forward: `mu` sentinel + post-MLP recombination + post-loop MUDD ----
    (
        "            mu = None",
        guard("model.use_mudd", "            mu = None"),
    ),
    (
        "            if mu is not None:\n"
        "                x = mu[12] * x + mu[13] * ReLUSqrdMLP(norm(x), c_fc, c_proj)\n"
        "            else:\n"
        "                x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * ReLUSqrdMLP(norm(x), c_fc, c_proj)",
        guard(
            "model.use_mudd",
            "            if mu is not None:\n"
            "                x = mu[12] * x + mu[13] * ReLUSqrdMLP(norm(x), c_fc, c_proj)\n"
            "            else:\n"
            "                x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * ReLUSqrdMLP(norm(x), c_fc, c_proj)",
            "            x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * ReLUSqrdMLP(norm(x), c_fc, c_proj)",
        ),
    ),
    (
        "        # Post-loop MUDD: 5 residual coefs over {cache[0], cache[7], cache[9], ve_bank0, cache[3]}.\n"
        "        mu = self.forward_mudd(x, id=1, num_coef=5)\n"
        "        ve_bank0 = ve[1][None].to(dtype=x.dtype)  # (1, T, D), same VE as layer-1 attn\n"
        "        x = x + mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * cache[9] + mu[3] * ve_bank0 + mu[4] * cache[3]",
        guard(
            "model.use_mudd",
            "        # Post-loop MUDD: 5 residual coefs over {cache[0], cache[7], cache[9], ve_bank0, cache[3]}.\n"
            "        mu = self.forward_mudd(x, id=1, num_coef=5)\n"
            "        ve_bank0 = ve[1][None].to(dtype=x.dtype)  # (1, T, D), same VE as layer-1 attn\n"
            "        x = x + mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * cache[9] + mu[3] * ve_bank0 + mu[4] * cache[3]",
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
    # ---- data loader: bigram-hash inputs (off -> reuse token inputs as a no-op placeholder) ----
    (
        "        _bigram_inputs = get_bigram_hash(_inputs)",
        guard(
            "data.needs_bigram_inputs",
            "        _bigram_inputs = get_bigram_hash(_inputs)",
            "        _bigram_inputs = _inputs  # bigram_hash disabled: forward ignores this",
        ),
    ),
    # ---- TrainingManager: sparse bigram-grad comms machinery (needs bigram_embed) ----
    (
        "        if _sparse_comms_active():\n"
        "            self.row_update_mask = np.zeros(args.bigram_vocab_size, dtype=np.uint8)\n"
        "            self.sparse_counts_state = None\n"
        "            # buffer we use for fast GPU uploads of send indexes\n"
        "            self.send_idxes_buffer = torch.empty(args.bigram_vocab_size, dtype=torch.int32, pin_memory=True)",
        guard(
            "model.use_bigram_hash",
            "        if _sparse_comms_active():\n"
            "            self.row_update_mask = np.zeros(args.bigram_vocab_size, dtype=np.uint8)\n"
            "            self.sparse_counts_state = None\n"
            "            # buffer we use for fast GPU uploads of send indexes\n"
            "            self.send_idxes_buffer = torch.empty(args.bigram_vocab_size, dtype=torch.int32, pin_memory=True)",
        ),
    ),
    (
        "        if not _sparse_comms_active():\n"
        "            return\n"
        "\n"
        "        self.row_update_mask[bigram_indexes] = 1\n"
        "\n"
        "        if self._is_adam_step(step):\n"
        "            with torch.no_grad():\n"
        "                bigram_idx_np = np.flatnonzero(self.row_update_mask).astype(np.int32)\n"
        "                send_idxes, send_counts, recv_counts, recv_counts_fut = sparse_comms_start(\n"
        "                    bigram_idx_np, args.bigram_vocab_size, rank, world_size, self.send_idxes_buffer\n"
        "                )\n"
        "                self.sparse_counts_state = (send_idxes, send_counts, recv_counts, recv_counts_fut)",
        guard(
            "model.use_bigram_hash",
            "        if not _sparse_comms_active():\n"
            "            return\n"
            "\n"
            "        self.row_update_mask[bigram_indexes] = 1\n"
            "\n"
            "        if self._is_adam_step(step):\n"
            "            with torch.no_grad():\n"
            "                bigram_idx_np = np.flatnonzero(self.row_update_mask).astype(np.int32)\n"
            "                send_idxes, send_counts, recv_counts, recv_counts_fut = sparse_comms_start(\n"
            "                    bigram_idx_np, args.bigram_vocab_size, rank, world_size, self.send_idxes_buffer\n"
            "                )\n"
            "                self.sparse_counts_state = (send_idxes, send_counts, recv_counts, recv_counts_fut)",
            "        return",
        ),
    ),
    (
        "        if not _sparse_comms_active() or not self._is_adam_step(step):\n"
        "            return\n"
        "\n"
        "        send_idxes, send_counts, recv_counts, recv_counts_fut = self.sparse_counts_state\n"
        "        self.sparse_counts_state = None\n"
        "\n"
        "        recv_counts_fut.wait()\n"
        "        recv_idxes, sparse_state, idxes_fut = sparse_comms_share_indexes(send_idxes, send_counts, recv_counts)\n"
        "        self.optimizer._reduce_futures[model.bigram_embed.weight] = [idxes_fut, recv_idxes]\n"
        "        self.optimizer._sparse_async_data[model.bigram_embed.weight] = sparse_state\n"
        "\n"
        "        self.row_update_mask.fill(0)",
        guard(
            "model.use_bigram_hash",
            "        if not _sparse_comms_active() or not self._is_adam_step(step):\n"
            "            return\n"
            "\n"
            "        send_idxes, send_counts, recv_counts, recv_counts_fut = self.sparse_counts_state\n"
            "        self.sparse_counts_state = None\n"
            "\n"
            "        recv_counts_fut.wait()\n"
            "        recv_idxes, sparse_state, idxes_fut = sparse_comms_share_indexes(send_idxes, send_counts, recv_counts)\n"
            "        self.optimizer._reduce_futures[model.bigram_embed.weight] = [idxes_fut, recv_idxes]\n"
            "        self.optimizer._sparse_async_data[model.bigram_embed.weight] = sparse_state\n"
            "\n"
            "        self.row_update_mask.fill(0)",
            "        return",
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
    # ---- untie embed from lm_head at 2/3 of training ----
    (
        "        if step == self.split_step:\n"
        "            self.optimizer.copy_lm_state_to_embed()",
        guard(
            "schedule.use_untie_embed",
            "        if step == self.split_step:\n"
            "            self.optimizer.copy_lm_state_to_embed()",
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


class TemplateBuildError(RuntimeError):
    """An EXACT replacement did not match exactly once (baseline drifted)."""


def build_template_text(baseline_text: str | None = None) -> str:
    """Derive the full Jinja template text from the baseline (pure, no I/O).

    Exposed (separately from :func:`main`) so a test can assert the committed
    ``train_gpt.py.j2`` byte-matches a fresh regeneration -- catching the case
    where ``build_template.py`` is edited but the generated artifact is not
    regenerated and committed alongside it.
    """
    text = BASELINE.read_text(encoding="utf-8") if baseline_text is None else baseline_text

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
    text = replace_region(
        text,
        "TRAINING_STAGES = [\n",
        "\n\n# TODO - Confirm.",
        "{{ training_stages_block }}",
    )

    for old, new in EXACT:
        count = text.count(old)
        if count != 1:
            raise TemplateBuildError(
                f"expected exactly 1 match, found {count} for:\n{old[:120]!r}"
            )
        text = text.replace(old, new)

    # Header comment, logged with the rest of the source for auditability.
    return "{{ header_comment }}\n\n" + text


def main() -> int:
    try:
        text = build_template_text()
    except TemplateBuildError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1

    TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE.write_text(text, encoding="utf-8")
    sys.stderr.write(f"Wrote {TEMPLATE} ({text.count(chr(10)) + 1} lines)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
