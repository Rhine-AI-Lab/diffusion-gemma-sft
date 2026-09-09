"""Torch-only verification of the GRPO pluggable SFT probability term.

Run inside the container venv:
  srun -ul --environment=cscs/gdsdv2-pt.toml \
      bash -c 'source .venv/bin/activate && python cscs/verify_grpo_sft_term.py'

Checks:
  1. diffusion_sft_logps(...) / n == -diffusion_sft_loss(...)  for every variant.
  2. The encoder-loss span toggle masks the expected label positions.
No model / trl / transformers needed: imports sft_loss.py directly by path.
"""

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_LOSS_PATH = os.path.join(HERE, "..", "DiffGemma", "diffgemma_trl", "sft_loss.py")
spec = importlib.util.spec_from_file_location("sft_loss", SFT_LOSS_PATH)
sft_loss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft_loss)


def test_equivalence():
    torch.manual_seed(0)
    B, L, K = 4, 16, 64
    logits = torch.randn(B, L, K)
    x0 = torch.randint(0, K, (B, L))
    xt = torch.randint(0, K, (B, L))
    t = torch.rand(B).clamp(0.05, 0.95)
    # full-completion mask with a few padded tails
    canvas_mask = torch.ones(B, L)
    canvas_mask[0, 12:] = 0.0
    canvas_mask[1, 8:] = 0.0
    n = canvas_mask.sum(-1).clamp(min=1.0)

    cases = [
        ("base-sft", None),
        ("reweighted-ce", None),
        ("reweighted-ce", 50.0),  # with weight clip
        ("loo-ce", None),
    ]
    for variant, clip in cases:
        loss = sft_loss.diffusion_sft_loss(
            logits, x0, xt, t, loss_mask=canvas_mask,
            variant=variant, vocab_size=K, diffusion_weight_clip=clip,
        )
        logps = sft_loss.diffusion_sft_logps(
            logits, x0, xt, t, loss_mask=canvas_mask,
            variant=variant, vocab_size=K, diffusion_weight_clip=clip,
        )
        recovered = logps / n  # objective divides summed logp by norm == n
        ok = torch.allclose(recovered, -loss, atol=1e-5, rtol=1e-4)
        maxerr = (recovered - (-loss)).abs().max().item()
        print(f"[equiv] variant={variant:<13} clip={str(clip):<5} maxerr={maxerr:.2e} -> {'OK' if ok else 'FAIL'}")
        assert ok, f"equivalence failed for {variant} clip={clip}"


def span_label_count(prompt_mask, canvas_mask, target):
    """Replicates trainer._encoder_ar_loss comp-mask + next-token shift label count."""
    seq_prompt = prompt_mask
    seq_canvas = canvas_mask.to(prompt_mask.dtype)
    zeros_p = torch.zeros_like(seq_prompt)
    zeros_c = torch.zeros_like(seq_canvas)
    if target == "completion":
        comp = torch.cat([zeros_p, seq_canvas], dim=1).bool()
    elif target == "prompt":
        comp = torch.cat([seq_prompt, zeros_c], dim=1).bool()
    elif target == "prompt_completion":
        comp = torch.cat([seq_prompt, seq_canvas], dim=1).bool()
    else:
        raise ValueError(target)
    labels = torch.zeros(comp.shape, dtype=torch.long)
    labels[~comp] = -100
    # next-token shift: labels[:, 1:]
    shift_labels = labels[:, 1:]
    return (shift_labels != -100).sum().item()


def test_span_toggle():
    B = 3
    prompt_mask = torch.ones(B, 5, dtype=torch.long)
    prompt_mask[0, :2] = 0  # left padding example
    canvas_mask = torch.ones(B, 7)
    canvas_mask[1, 4:] = 0.0
    # completion-only scores canvas tokens (minus the boundary token lost to shift),
    # prompt-only scores prompt tokens, prompt_completion scores both spans.
    c = span_label_count(prompt_mask, canvas_mask, "completion")
    p = span_label_count(prompt_mask, canvas_mask, "prompt")
    pc = span_label_count(prompt_mask, canvas_mask, "prompt_completion")
    print(f"[span] completion={c} prompt={p} prompt_completion={pc}")
    assert c > 0 and p > 0 and pc > 0, "each span must score some labels"
    assert pc >= c and pc >= p, "prompt_completion must score at least as many as either span"
    # prompt_completion includes the prompt->completion boundary token, so it can
    # exceed the sum of the individually-shifted spans by at most B.
    assert pc <= p + c + B, "prompt_completion label count out of expected range"


if __name__ == "__main__":
    test_equivalence()
    test_span_toggle()
    print("ALL CHECKS PASSED")
