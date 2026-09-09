"""CPU unit tests for the DiffusionGemma SFT loss variants (no model / GPU needed).

Run:  python -m pytest DiffGemma/diffgemma_trl/tests/test_sft_loss.py
  or: python DiffGemma/diffgemma_trl/tests/test_sft_loss.py
"""

import importlib.util
import os

import numpy as np
import torch

# Load sft_loss.py directly (it has no trl/transformers deps), bypassing the
# package __init__ which imports the RL trainer (and hence trl).
_spec = importlib.util.spec_from_file_location(
    "diffgemma_sft_loss",
    os.path.join(os.path.dirname(__file__), "..", "sft_loss.py"),
)
_sft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sft)
diffusion_sft_loss = _sft.diffusion_sft_loss
diffusion_sft_logps = _sft.diffusion_sft_logps
loo_to_denoiser_logits = _sft.loo_to_denoiser_logits
normalize_variant = _sft.normalize_variant
rf_alpha = _sft.rf_alpha
rf_alpha_dot = _sft.rf_alpha_dot

torch.manual_seed(0)
K = 11
B, L = 3, 5


def _batch():
    logits = torch.randn(B, L, K, dtype=torch.float64)
    x0 = torch.randint(0, K, (B, L))
    xt = torch.randint(0, K, (B, L))
    t = torch.full((B,), 0.3, dtype=torch.float64)
    mask = torch.ones(B, L, dtype=torch.float64)
    return logits, x0, xt, t, mask


def _masked_ce_np(logits, x0, mask):
    lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    ce = -np.take_along_axis(lp, x0[..., None], -1)[..., 0]
    return (ce * mask).sum(-1) / np.clip(mask.sum(-1), 1, None)


def test_base_sft_is_plain_masked_ce():
    logits, x0, xt, t, mask = _batch()
    got = diffusion_sft_loss(logits, x0, xt, t, mask, variant="base-sft", vocab_size=K)
    ref = _masked_ce_np(logits.numpy(), x0.numpy(), mask.numpy())
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-6, atol=1e-6)


def test_reweighted_ce_is_base_times_weight():
    # w(t) = -alpha'/alpha = 1/(1-t) for the RF schedule; no 1/K factor.
    logits, x0, xt, t, mask = _batch()
    base = diffusion_sft_loss(logits, x0, xt, t, mask, variant="base-sft", vocab_size=K)
    rw = diffusion_sft_loss(logits, x0, xt, t, mask, variant="reweighted-ce", vocab_size=K)
    w = (-rf_alpha_dot(t) / rf_alpha(t)).numpy()  # 1/(1-0.3) ~= 1.4286
    np.testing.assert_allclose(rw.numpy(), base.numpy() * w, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(w, 1.0 / (1.0 - 0.3) * np.ones(B), rtol=1e-6)


def test_reweighted_ce_has_no_1_over_K():
    # The fixed weight must NOT depend on K. Compare two vocab sizes with the
    # same probabilities: per-token CE is K-independent, so the loss must match.
    t = torch.full((B,), 0.3, dtype=torch.float64)
    mask = torch.ones(B, L, dtype=torch.float64)
    logits = torch.randn(B, L, 4, dtype=torch.float64)
    x0 = torch.randint(0, 4, (B, L))
    xt = torch.randint(0, 4, (B, L))
    a = diffusion_sft_loss(logits, x0, xt, t, mask, variant="reweighted-ce", vocab_size=4)
    # pad logits to K=400 with -inf-ish so softmax mass is unchanged
    big = torch.full((B, L, 400), -30.0, dtype=torch.float64)
    big[..., :4] = logits
    b = diffusion_sft_loss(big, x0, xt, t, mask, variant="reweighted-ce", vocab_size=400)
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-4)



def test_reweighted_ce_weight_clip_caps_large_weight():
    logits, x0, xt, _, mask = _batch()
    t = torch.full((B,), 0.999, dtype=torch.float64)
    base = diffusion_sft_loss(logits, x0, xt, t, mask, variant="base-sft", vocab_size=K)
    clipped = diffusion_sft_loss(
        logits,
        x0,
        xt,
        t,
        mask,
        variant="reweighted-ce",
        vocab_size=K,
        diffusion_weight_clip=50.0,
    )

    np.testing.assert_allclose(clipped.numpy(), base.numpy() * 50.0, rtol=1e-6, atol=1e-6)


def test_weight_clip_does_not_change_base_sft_when_weight_is_one():
    logits, x0, xt, t, mask = _batch()
    unclipped = diffusion_sft_loss(logits, x0, xt, t, mask, variant="base-sft", vocab_size=K)
    clipped = diffusion_sft_loss(
        logits, x0, xt, t, mask, variant="base-sft", vocab_size=K, diffusion_weight_clip=50.0
    )

    np.testing.assert_allclose(clipped.numpy(), unclipped.numpy(), rtol=1e-6, atol=1e-6)

def test_loo_shift_only_moves_observed_token():
    logits = torch.randn(2, 3, 7, dtype=torch.float64)
    xt = torch.tensor([[0, 1, 2], [6, 5, 4]])
    alpha = torch.full((2, 1), 0.4, dtype=torch.float64)
    out = loo_to_denoiser_logits(logits, xt, alpha, 7)
    diff = (out - logits).numpy()
    expected = np.log1p(7 * 0.4 / (1 - 0.4))
    for b in range(2):
        for l in range(3):
            tok = int(xt[b, l])
            assert abs(diff[b, l, tok] - expected) < 1e-9
            moved = np.nonzero(np.abs(diff[b, l]) > 1e-9)[0]
            np.testing.assert_array_equal(moved, np.array([tok]))


def test_loo_ce_equals_ce_on_shifted_logits():
    logits, x0, xt, t, mask = _batch()
    got = diffusion_sft_loss(logits, x0, xt, t, mask, variant="loo-ce", vocab_size=K)
    alpha = rf_alpha(t).unsqueeze(-1)
    shifted = loo_to_denoiser_logits(logits, xt, alpha, K)
    ref = _masked_ce_np(shifted.numpy(), x0.numpy(), mask.numpy())  # w=1
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-6, atol=1e-6)


def test_reweighted_loo_combines_shift_and_time_weight():
    logits, x0, xt, t, mask = _batch()
    got = diffusion_sft_loss(
        logits,
        x0,
        xt,
        t,
        mask,
        variant="reweighted-loo-ce",
        vocab_size=K,
    )
    loo = diffusion_sft_loss(
        logits, x0, xt, t, mask, variant="loo-ce", vocab_size=K
    )
    weight = -rf_alpha_dot(t) / rf_alpha(t)
    np.testing.assert_allclose(
        got.numpy(), (loo * weight).numpy(), rtol=1e-6, atol=1e-6
    )


def test_reweighted_loo_logps_use_the_loo_shift():
    logits, x0, xt, t, mask = _batch()
    got = diffusion_sft_logps(
        logits,
        x0,
        xt,
        t,
        mask,
        variant="reweighted-loo-ce",
        vocab_size=K,
    )
    loss = diffusion_sft_loss(
        logits,
        x0,
        xt,
        t,
        mask,
        variant="reweighted-loo-ce",
        vocab_size=K,
    )
    np.testing.assert_allclose(
        got.numpy(), (-loss * mask.sum(-1)).numpy(), rtol=1e-6, atol=1e-6
    )


def test_early_combined_variant_alias_is_supported():
    assert normalize_variant("loo-reweighted-ce") == "reweighted-loo-ce"


def test_mask_excludes_tokens():
    logits, x0, xt, t, mask = _batch()
    mask[:, 2:] = 0.0
    base = diffusion_sft_loss(logits, x0, xt, t, mask, variant="base-sft", vocab_size=K)
    pert = logits.clone()
    pert[:, 2:] += 5.0
    after = diffusion_sft_loss(pert, x0, xt, t, mask, variant="base-sft", vocab_size=K)
    np.testing.assert_allclose(base.numpy(), after.numpy(), rtol=1e-6, atol=1e-6)


def test_correct_denoiser_beats_uniform():
    _, x0, xt, t, mask = _batch()
    good = torch.nn.functional.one_hot(x0, K).double() * 12.0
    unif = torch.zeros(B, L, K, dtype=torch.float64)
    for v in ("base-sft", "reweighted-ce", "loo-ce", "reweighted-loo-ce"):
        g = diffusion_sft_loss(good, x0, xt, t, mask, variant=v, vocab_size=K).mean()
        u = diffusion_sft_loss(unif, x0, xt, t, mask, variant=v, vocab_size=K).mean()
        assert float(g) < float(u), v


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASS")
