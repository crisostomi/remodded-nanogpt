"""Codegen / rendering tests."""

from __future__ import annotations

from nano.builder.codegen import generate
from nano.builder.render import build_context, render_train_script_text
from nano.config.presets import resolve_feature_set


def _render(preset="current_record", enable=None, disable=None) -> str:
    fs = resolve_feature_set(preset=preset, enable=enable or [], disable=disable or [])
    ctx = build_context(fs.enabled, overrides=fs.overrides)
    return render_train_script_text(
        ctx, header={"feature_set": fs.name, "enabled_features": sorted(fs.enabled)}
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
