"""GR00T projection inventory shared by integer and floating-point adapters."""

import torch


def inventory(model, expanded=False):
    names = {id(m): n for n, m in model.named_modules()}
    sites, groups, activations, text = {}, [], [], []

    def add(m, role, scope):
        assert isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.bfloat16
        name = names[id(m)]
        sites[name] = (m, role, scope)
        return name

    for name, m in model.named_modules():
        kind = type(m).__name__
        if kind == "Qwen3VLVisionAttention":
            add(m.qkv, "AttnQKV", "vision")
            add(m.proj, "AttnO", "vision")
        elif kind in ("Qwen3VLVisionMLP", "Qwen3VLVisionPatchMerger"):
            add(m.linear_fc1, "FFNup", "vision")
            down = add(m.linear_fc2, "FFNdown", "vision")
            act_kind = type(m.act_fn).__name__
            assert act_kind in ("GELUTanh", "GELU"), act_kind
            approximate = "tanh" if act_kind == "GELUTanh" else m.act_fn.approximate
            activations.append((m.act_fn, "forward", down, approximate))
        elif kind == "Qwen3VLTextAttention":
            groups.append(
                [
                    add(getattr(m, a), r, "text")
                    for a, r in [
                        ("q_proj", "AttnQ"),
                        ("k_proj", "AttnK"),
                        ("v_proj", "AttnV"),
                    ]
                ]
            )
            add(m.o_proj, "AttnO", "text")
        elif kind == "Qwen3VLTextMLP":
            assert m.config.hidden_act == "silu"
            groups.append(
                [add(m.gate_proj, "FFNgate", "text"), add(m.up_proj, "FFNup", "text")]
            )
            down = add(m.down_proj, "FFNdown", "text")
            text.append((m, down))
        elif kind == "BasicTransformerBlock":
            a = m.attn1
            scope = (
                "dit"
                if name.startswith("action_head.model.transformer_blocks.")
                else "vl_transformer"
            )
            qkv = [
                add(getattr(a, attr), role, scope)
                for attr, role in [
                    ("to_q", "AttnQ"),
                    ("to_k", "AttnK"),
                    ("to_v", "AttnV"),
                ]
            ]
            groups.append(qkv[1:] if a.is_cross_attention else qkv)
            add(a.to_out[0], "AttnO", scope)
            assert type(m.ff.net[0]).__name__ == "GELU"
            add(m.ff.net[0].proj, "FFNup", scope)
            down = add(m.ff.net[2], "FFNdown", scope)
            activations.append((m.ff.net[0], "gelu", down, m.ff.net[0].approximate))
    if expanded:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and name not in sites:
                add(m, "AdditionalLinear", "additional")
    return sites, groups, activations, text
