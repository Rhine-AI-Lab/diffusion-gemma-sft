"""Import + config-validation check for the GRPO pluggable term (needs trl/transformers)."""
from DiffGemma.diffgemma_trl.trainer import (
    DiffusionGemmaGRPOTrainer, DiffusionGemmaGDSDTrainer, _DiffusionGemmaCanvasTrainer,
)
from DiffGemma.diffgemma_trl.configs import DiffusionGemmaGRPOConfig, GRPO_PROB_TERMS, ENCODER_LOSS_TARGETS

for v in GRPO_PROB_TERMS:
    DiffusionGemmaGRPOConfig(output_dir="/tmp/x", grpo_prob_term=v)
for tgt in ENCODER_LOSS_TARGETS:
    DiffusionGemmaGRPOConfig(output_dir="/tmp/x", encoder_loss_target=tgt)
DiffusionGemmaGRPOConfig(output_dir="/tmp/x", grpo_prob_term="reweighted-ce", diffusion_weight_clip=20.0)
print("valid configs: OK")

def must_raise(**kw):
    try:
        DiffusionGemmaGRPOConfig(output_dir="/tmp/x", **kw)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {kw}")

must_raise(grpo_prob_term="bogus")
must_raise(encoder_loss_target="bogus")
must_raise(diffusion_weight_clip=-1.0)
print("invalid configs raise: OK")

# GRPO overrides the new seams; GDSD inherits the base no-op _auxiliary_loss.
assert DiffusionGemmaGRPOTrainer._auxiliary_loss is not _DiffusionGemmaCanvasTrainer._auxiliary_loss
assert DiffusionGemmaGDSDTrainer._auxiliary_loss is _DiffusionGemmaCanvasTrainer._auxiliary_loss
assert DiffusionGemmaGRPOTrainer._canvas_logps is not _DiffusionGemmaCanvasTrainer._canvas_logps
assert DiffusionGemmaGDSDTrainer._canvas_logps is _DiffusionGemmaCanvasTrainer._canvas_logps
print("seam wiring (GRPO overrides, GDSD inherits base): OK")
print("ALL IMPORT CHECKS PASSED")
