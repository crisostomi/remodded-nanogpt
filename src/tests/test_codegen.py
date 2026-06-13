"""Codegen / rendering tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from nano.builder.codegen import generate
from nano.builder.render import build_context, render_train_script_text
from nano.config.presets import resolve_feature_set

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = REPO_ROOT / "src/nano/builder/templates/train_gpt.py.j2"


def _load_build_template_module():
    spec = importlib.util.spec_from_file_location(
        "_build_template", REPO_ROOT / "baseline" / "build_template.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render(preset="current_record", enable=None, disable=None) -> str:
    fs = resolve_feature_set(preset=preset, enable=enable or [], disable=disable or [])
    ctx = build_context(fs.enabled, overrides=fs.overrides)
    return render_train_script_text(
        ctx, header={"feature_set": fs.name, "enabled_features": sorted(fs.enabled)}
    )


def test_committed_template_matches_build_script():
    """The committed train_gpt.py.j2 must byte-match a fresh regeneration.

    Guards against the (silent, GPU-only) failure mode where build_template.py is
    edited to add/remove guards but the generated artifact is not regenerated and
    committed alongside it -- leaving the rendered off-states broken.
    """
    expected = _load_build_template_module().build_template_text()
    committed = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert committed == expected, (
        "src/nano/builder/templates/train_gpt.py.j2 is stale. "
        "Run `python baseline/build_template.py` and commit the regenerated template."
    )


def test_render_current_record_is_valid_python():
    text = _render()
    compile(text, "<train_current_record>", "exec")  # syntax check, no execution


def test_render_current_record_keeps_record_code():
    text = _render()
    # all toggles on -> record hot-path code present verbatim
    assert "y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w))" in text
    assert "self.xsa_alphas = nn.Parameter(torch.zeros(num_layers, self.num_heads))" in text
    assert "bigram_signs = self.bigram_sign_table[sign_idx]" in text
    assert "vn = F.normalize(v, dim=-1, eps=1e-4)" in text  # original XSA lowering


def test_ablation_removes_gate_and_rewrites_xsa():
    text = _render(disable=["sparse_attention_gate"], enable=["xsa_lowering_rewrite"])
    compile(text, "<abl>", "exec")
    # sparse attention gate gone
    assert "torch.sigmoid(F.linear(x[..., :12], attn_gate_w))" not in text
    assert "self.attn_gate_bank = nn.Parameter" not in text
    assert "attn_gates = [None] * self.num_layers" in text
    # XSA lowering rewritten
    assert "y = y - alpha * (dot / denom) * v" in text
    assert "vn = F.normalize(v, dim=-1, eps=1e-4)" not in text
    # xsa itself still present
    assert "self.xsa_alphas = nn.Parameter" in text


def test_disable_sign_trick_uses_plain_bigram_injection():
    text = _render(disable=["bigram_sign_trick"])
    compile(text, "<no_sign>", "exec")
    assert "self.bigram_sign_table[sign_idx]" not in text
    assert "x0_bigram = self.bigram_embed(bigram_input_seq)[None]" in text
    assert "dist.broadcast(model.bigram_sign_table" not in text


def test_disable_mudd_falls_back_to_normal_residual_path():
    text = _render(disable=["mudd_last_layers"])
    compile(text, "<no_mudd>", "exec")
    # All MUDD machinery gone (methods, calls, params, post-loop mixer).
    for sym in ("forward_mudd", "init_mudd", "mudd_w1", "mudd_b2", "cache[9]", "ve_bank0", "mu["):
        assert sym not in text, sym
    # The last-layer dispatch degrades from `elif ve[i]` to a plain `if ve[i]`.
    assert "                if ve[i] is not None:" in text
    assert "elif ve[i] is not None:" not in text
    # Normal per-layer residual recombination is used everywhere.
    assert "                x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0_inject[i]" in text
    assert "            x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * ReLUSqrdMLP(norm(x), c_fc, c_proj)" in text
    # value embeds + their gates are untouched.
    assert "ve_gate_out = 2 * torch.sigmoid" in text
    assert "self.value_embeds = nn.Parameter" in text


def test_disable_value_embed_gates_uses_ungated_value_embeds():
    text = _render(disable=["value_embed_gates"])
    compile(text, "<no_ve_gates>", "exec")
    for sym in ("ve_gate_bank", "veg = self.ve_gate_bank", "ve_gates["):
        assert sym not in text, sym
    # Ungated fallback: value embeds added directly as aux_v.
    assert "                    aux_v = ve_view.view(B, T, -1)" in text
    # MUDD's own dynamic gate (mu[6], mu[7]) is independent of ve_gate_bank.
    assert "ve_gate = torch.cat([mu[6], mu[7]]" in text
    assert "self.forward_mudd(" in text


def test_disable_value_embeds_drops_aux_v_and_dependents():
    # value_embeds is required by value_embed_gates and mudd_last_layers, so the
    # whole cluster comes off together.
    text = _render(disable=["value_embeds", "value_embed_gates", "mudd_last_layers"])
    compile(text, "<no_ve>", "exec")
    for sym in ("self.value_embeds", "ve = self.value_embeds", "ve_view", "ve[i]",
                "ve_gate_bank", "forward_mudd", "ve_bank0"):
        assert sym not in text, sym
    assert "                aux_v = None" in text


BIGRAM_FAMILY = [
    "bigram_hash", "bigram_sign_trick", "bigram_vocab_15x", "bigram_dim_192",
    "residual_slice_bigram_injection", "sparse_bigram_comms",
]


def test_disable_bigram_family_removes_all_bigram_machinery():
    text = _render(disable=BIGRAM_FAMILY)
    compile(text, "<no_bigram>", "exec")
    # No executable bigram references survive (construction, forward, comms).
    for sym in ("self.bigram_embed", "x0_bigram", "self.bigram_sign_table", "bigram_signs",
                "x0_inject", "bg_inject", "self.x0_lambdas", "self.bigram_lambdas",
                "dist.broadcast(model.bigram_sign_table",
                "_reduce_futures[model.bigram_embed.weight]"):
        assert sym not in text, sym
    # Data loader passes a harmless placeholder; the hash is not computed.
    assert "_bigram_inputs = _inputs" in text
    assert "_bigram_inputs = get_bigram_hash(_inputs)" not in text
    # Sparse-comms TrainingManager methods degrade to no-ops.
    assert "    def sparse_index_update(self, step, bigram_indexes):\n        return" in text
    assert "    def sparse_index_share(self, step):\n        return" in text
    # Hyperparameters keep valid literal int defaults (not None).
    assert "    bigram_vocab_size: int = 50304 * 15" in text
    assert "    bigram_dim: int = 192" in text
    # get_bigram_hash stays defined (the optional eval path still imports it).
    assert "def get_bigram_hash(x):" in text


def test_disable_bigram_keeps_mudd_but_drops_bigram_injection():
    # MUDD requires only value_embeds, so it survives a bigram drop; its
    # mu[11] * x0_bigram injection is guarded off.
    text = _render(disable=BIGRAM_FAMILY)
    assert "self.forward_mudd(" in text          # MUDD intact
    assert "mu[11] * x0_bigram" not in text       # bigram injection gone
    # Normal residual line loses the x0_inject suffix.
    assert "attn_out + x0_inject[i]" not in text


def test_generated_manifest_has_code_hash(tmp_path):
    fs = resolve_feature_set(preset="current_record")
    out = tmp_path / "train.py"
    result = generate(
        fs,
        out,
        features_path=tmp_path / "train.features.yaml",
        manifest_path=tmp_path / "train.manifest.json",
    )
    manifest = result["manifest"]
    assert manifest["generated"]["sha256"]
    assert len(manifest["generated"]["sha256"]) == 64
    # hash must match the actually-written file
    import hashlib

    assert manifest["generated"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert (tmp_path / "triton_kernels.py").exists()  # copied alongside
    assert result["features_path"].exists()
