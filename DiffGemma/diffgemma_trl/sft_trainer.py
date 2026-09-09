"""Supervised fine-tuning Trainer for DiffusionGemma (uniform-state diffusion).

`DiffusionGemmaSFTTrainer` subclasses `transformers.Trainer` and overrides
`compute_loss` with the selectable canvas-denoising objectives in
`sft_loss.py`, with feature parity to the JAX hackable_diffusion
SFT:

  * uniform-state forward corruption with one diffusion time t per sequence,
    drawn from [diffusion_noise_min, diffusion_noise_max];
  * optional self-conditioning (detached first-pass logits fed to a 2nd pass);
  * rectified-flow schedule alpha(t) = 1 - t (used by reweighted-ce / loo-ce);
  * optional encoder AR (next-token) loss over [prompt; x0].

The canvas forward / corruption / self-conditioning helpers mirror the RL
trainer in `trainer.py` so both paths drive the HF model identically.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
import transformers

from .sft_loss import diffusion_sft_loss, rf_alpha

logger = logging.getLogger(__name__)


class DiffusionGemmaSFTTrainer(transformers.Trainer):
    """Canvas-denoising SFT for DiffusionGemma with selectable loss variants."""

    # ---- small accessors -------------------------------------------------
    def _config(self, model):
        return getattr(getattr(model, "config", None), "text_config", None) or getattr(model, "config", None)

    def _canvas_length(self, model) -> int:
        configured = getattr(self.args, "diffusion_canvas_length", None)
        if configured:
            return int(configured)
        cfg = self._config(model)
        return int(getattr(cfg, "canvas_length", 256) or 256)

    def _vocab_size(self, model) -> int:
        return int(getattr(self.args, "diffusion_vocab_size", None)
                   or getattr(self._config(model), "vocab_size", 262144))

    # ---- model forward (mirrors trainer.py::_diffusion_forward) ----------
    def _diffusion_forward(self, model, prompt_ids, prompt_mask, noisy_canvas,
                           self_conditioning_logits=None, self_conditioning_mask=None):
        canvas_mask = torch.ones_like(noisy_canvas, dtype=prompt_mask.dtype, device=noisy_canvas.device)
        decoder_attention_mask = torch.cat([prompt_mask.to(noisy_canvas.device), canvas_mask], dim=1)
        base = {
            "input_ids": prompt_ids,
            "self_conditioning_logits": self_conditioning_logits,
            "self_conditioning_mask": self_conditioning_mask,
        }
        attempts = [
            {**base, "attention_mask": prompt_mask, "decoder_attention_mask": decoder_attention_mask, "decoder_input_ids": noisy_canvas},
            {**base, "attention_mask": prompt_mask, "decoder_attention_mask": decoder_attention_mask, "canvas_ids": noisy_canvas},
            {**base, "attention_mask": prompt_mask, "decoder_input_ids": noisy_canvas},
            {**base, "attention_mask": prompt_mask, "canvas_ids": noisy_canvas},
            {**base, "decoder_input_ids": noisy_canvas},
            {**base, "canvas_ids": noisy_canvas},
        ]
        last = None
        for kw in attempts:
            try:
                return model(**kw)
            except TypeError as exc:
                last = exc
        raise TypeError("Could not call DiffusionGemma forward with decoder_input_ids/canvas_ids.") from last

    def _self_conditioning(self, model, prompt_ids, prompt_mask, noisy_canvas):
        """Returns ``(sc_logits, sc_mask)`` for self-conditioning.

        With probability ``p``, an example uses the (detached) first-pass logits
        as self-conditioning. Crucially we pass a per-example **mask**, not zeroed
        logits: the model computes ``softmax(logits) @ E`` and multiplies by
        ``self_conditioning_mask``, so masked examples get a true zero
        soft-embedding. Zeroing the *logits* instead would give
        ``softmax(0) = uniform`` -> the mean token embedding (a garbage signal).
        """
        prob = float(getattr(self.args, "diffusion_self_conditioning_prob", 0.0))
        if prob <= 0:
            return None, None
        with torch.no_grad():
            first = self._diffusion_forward(model, prompt_ids, prompt_mask, noisy_canvas)
        sc = first.logits.detach().to(getattr(model, "dtype", first.logits.dtype))
        sc_mask = torch.rand((sc.shape[0],), device=sc.device) < prob  # [B] bool
        return sc, sc_mask

    # ---- uniform-state corruption (one t per sequence) -------------------
    def _corrupt_canvas(self, x0, canvas_mask, vocab_size):
        device = x0.device
        b, l = x0.shape
        lo = float(getattr(self.args, "diffusion_noise_min", 1e-3))
        hi = float(getattr(self.args, "diffusion_noise_max", 1.0 - 1e-3))
        t = torch.empty((b,), device=device).uniform_(lo, hi)          # [B]
        valid = canvas_mask.to(device=device, dtype=torch.bool)
        corrupt = (torch.rand((b, l), device=device) < t[:, None]) & valid
        rand = torch.randint(0, vocab_size, x0.shape, device=device)
        xt = torch.where(corrupt, rand, x0)
        return xt, t

    # ---- encoder AR loss over [prompt; x0] -------------------------------
    def _lm_head_and_cap(self, model):
        """Reach DiffusionGemmaForBlockDiffusion's encoder + lm_head + softcap.

        Under multi-GPU, `model` is a DistributedDataParallel wrapper (exposes neither
        get_base_model nor .model) -> unwrap accelerate/DDP FIRST, then unwrap PEFT.
        Returns ``(encoder_or_None, lm_head_or_None, softcap_or_None)``."""
        base = model
        if getattr(self, "accelerator", None) is not None:
            base = self.accelerator.unwrap_model(base)
        else:
            while hasattr(base, "module") and not hasattr(base, "get_base_model"):
                base = base.module
        if hasattr(base, "get_base_model"):
            base = base.get_base_model()                          # unwrap PEFT
        inner = getattr(base, "model", base)                      # DiffusionGemmaModel
        encoder = getattr(inner, "encoder", None) or getattr(inner, "encoder_model", None)
        return encoder, getattr(base, "lm_head", None), getattr(base, "final_logit_softcapping", None)

    def _encoder_ar_loss(self, model, prompt_ids, prompt_mask, x0, canvas_mask,
                         prompt_comp_mask=None, encoder_hidden=None):
        """Next-token CE on the completion region using encoder hidden states.

        Mirrors the JAX SFT encoder loss. Needs the model to expose the lm_head (and, unless
        ``encoder_hidden`` is supplied, the encoder text model); guarded so a missing API
        degrades to 0 (decoder-only) rather than crashing.

        Two spans, selected by ``block_diffusion``:
          * legacy: encoder over ``[prompt; x0]``, score the completion tokens ``x0``.
          * block:  encoder over ``prompt_ids`` (already ``[prompt; clean prior blocks]``),
                    score the prefix-completion tokens flagged by ``prompt_comp_mask``.
                    b=0 (empty prefix -> all-zero mask) returns 0 (avoids all-`-100` NaN).

        ``encoder_hidden`` (block mode): the encoder's last_hidden_state ALREADY computed by
        the decoder's forward (``outputs.encoder_last_hidden_state``). Passing it reuses that
        single encoder pass instead of running a second one -> one activation graph (fits at
        canvas 256) and one combined backward (every param gets a grad -> DDP-clean)."""
        weight = float(getattr(self.args, "encoder_loss_weight", 0.0))
        if weight <= 0:
            return x0.new_zeros((), dtype=torch.float32)
        block_mode = bool(getattr(self.args, "block_diffusion", False))
        if block_mode and (prompt_comp_mask is None or float(prompt_comp_mask.sum()) == 0.0):
            return x0.new_zeros((), dtype=torch.float32)
        encoder, lm_head, cap = self._lm_head_and_cap(model)
        if lm_head is None or (encoder_hidden is None and encoder is None):
            logger.warning("encoder_loss_weight>0 but encoder/lm_head not found; skipping encoder loss.")
            return x0.new_zeros((), dtype=torch.float32)

        if block_mode:
            # encoder input is already [prompt; clean prior blocks]; score the prefix span.
            seq = prompt_ids
            attn = prompt_mask
            comp = prompt_comp_mask.to(prompt_ids.device).bool()
        else:
            seq = torch.cat([prompt_ids, x0], dim=1)
            attn = torch.cat([prompt_mask, canvas_mask.to(prompt_mask.dtype)], dim=1)
            comp = torch.cat([torch.zeros_like(prompt_mask), canvas_mask.to(prompt_mask.dtype)], dim=1).bool()
        if encoder_hidden is not None:
            hidden = encoder_hidden                             # reuse the decoder forward's pass
        else:
            enc_out = encoder(input_ids=seq, attention_mask=attn)
            hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out[0]
        # Match hidden to the lm_head weight dtype: the reused encoder_last_hidden_state can come
        # back fp32 while lm_head is bf16 (mixed precision) -> matmul dtype mismatch otherwise.
        hidden = hidden.to(lm_head.weight.dtype)
        logits = lm_head(hidden).float()                       # [B, S, K]
        # Apply the model's final-logit softcapping so the encoder head is trained
        # on the same (capped) logits generation uses (modeling: logits = c*tanh(logits/c)).
        if cap:
            logits = cap * torch.tanh(logits / cap)
        # next-token targets on the scored (completion) span only
        labels = seq.clone()
        labels[~comp] = -100
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        return weight * F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

    # ---- loss components -------------------------------------------------
    def _move(self, model, inputs):
        device = next(p.device for p in model.parameters() if p.device.type != "meta")
        return (inputs["prompt_ids"].to(device), inputs["prompt_mask"].to(device),
                inputs["canvas"].to(device), inputs["canvas_mask"].to(device))

    def _decoder_loss(self, model, inputs):
        """Weighted canvas-denoising loss (builds only the decoder autograd graph)."""
        prompt_ids, prompt_mask, x0, canvas_mask = self._move(model, inputs)
        K = self._vocab_size(model)
        xt, t = self._corrupt_canvas(x0, canvas_mask, K)
        sc_logits, sc_mask = self._self_conditioning(model, prompt_ids, prompt_mask, xt)
        outputs = self._diffusion_forward(model, prompt_ids, prompt_mask, xt, sc_logits, sc_mask)
        dec = diffusion_sft_loss(
            outputs.logits, x0, xt, t, loss_mask=canvas_mask,
            variant=self.args.sft_variant, vocab_size=K,
            diffusion_weight_clip=getattr(self.args, "diffusion_weight_clip", None),
            alpha_fn=rf_alpha,
        ).mean()
        return float(self.args.decoder_loss_weight) * dec

    def _encoder_loss(self, model, inputs):
        """Weighted encoder AR loss (builds only the encoder autograd graph)."""
        prompt_ids, prompt_mask, x0, canvas_mask = self._move(model, inputs)
        comp = inputs.get("prompt_comp_mask")  # present only in block_diffusion mode
        return self._encoder_ar_loss(model, prompt_ids, prompt_mask, x0, canvas_mask, comp)

    def _block_loss(self, model, inputs):
        """Block-diffusion combined loss from a SINGLE shared forward. The decoder forward
        already runs the encoder over ``[prompt; clean prior blocks]`` to build the KV cache
        the decoder cross-attends to, and returns that ``encoder_last_hidden_state``. We reuse
        it for the encoder AR loss instead of running a second encoder pass -> ONE activation
        graph (fits at canvas 256) and ONE combined backward, so every param gets a gradient
        every step (no DDP reducer mismatch, no no_sync). Returns ``(dec, enc)``."""
        prompt_ids, prompt_mask, x0, canvas_mask = self._move(model, inputs)
        comp = inputs.get("prompt_comp_mask")
        K = self._vocab_size(model)
        xt, t = self._corrupt_canvas(x0, canvas_mask, K)
        sc_logits, sc_mask = self._self_conditioning(model, prompt_ids, prompt_mask, xt)
        outputs = self._diffusion_forward(model, prompt_ids, prompt_mask, xt, sc_logits, sc_mask)
        dec = diffusion_sft_loss(
            outputs.logits, x0, xt, t, loss_mask=canvas_mask,
            variant=self.args.sft_variant, vocab_size=K,
            diffusion_weight_clip=getattr(self.args, "diffusion_weight_clip", None),
            alpha_fn=rf_alpha,
        ).mean()
        dec = float(self.args.decoder_loss_weight) * dec
        enc = self._encoder_ar_loss(model, prompt_ids, prompt_mask, x0, canvas_mask, comp,
                                    encoder_hidden=getattr(outputs, "encoder_last_hidden_state", None))
        return dec, enc

    # ---- main loss (single-backward path: block mode always; legacy when interleaved off) ----
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Record which mutually-exclusive canvas-type bin each sampled canvas fell in
        # (reasoning-only / transition / answer-only) so we can log the portions the model
        # sees. Training only; cumulative counts (see log()).
        cb = inputs.get("canvas_bins")
        if cb is not None and getattr(model, "training", True):
            if not hasattr(self, "_bin_counts"):
                self._bin_counts = torch.zeros(3)
            self._bin_counts += cb.detach().float().sum(0).cpu()
        if bool(getattr(self.args, "block_diffusion", False)):
            dec, enc = self._block_loss(model, inputs)          # one shared forward
        else:
            dec = self._decoder_loss(model, inputs)
            enc = self._encoder_loss(model, inputs)
        loss = dec + enc
        if return_outputs:
            return loss, {"decoder_loss": dec.detach(), "encoder_loss": enc.detach()}
        return loss

    def log(self, logs, *args, **kwargs):
        """Emit cumulative canvas-type fractions alongside the usual training logs."""
        counts = getattr(self, "_bin_counts", None)
        if counts is not None and float(counts.sum()) > 0:
            c = counts.clone()
            acc = getattr(self, "accelerator", None)
            if acc is not None and getattr(acc, "num_processes", 1) > 1:
                c = acc.reduce(c.to(acc.device), reduction="sum").cpu()
            tot = float(c.sum())
            logs["canvas/frac_reasoning_only"] = float(c[0]) / tot
            logs["canvas/frac_transition"] = float(c[1]) / tot
            logs["canvas/frac_answer_only"] = float(c[2]) / tot
        return super().log(logs, *args, **kwargs)

    # ---- interleaved forward/backward (encoder loss on) ------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Interleaved backward: decoder fwd+bwd then encoder fwd+bwd so only ONE activation
        graph is resident at a time (~halves peak memory vs a single combined backward).
        Grads accumulate additively -> identical optimizer step. The first backward runs
        under accelerator.no_sync so DDP all-reduces once (on the second backward) instead of
        erroring on a second grad-ready mark.

        Gated behind ``--interleaved_backward`` (default off) and NEVER used in block mode:
        block mode shares one encoder pass across both losses (`_block_loss`), so a single
        combined backward already fits and is DDP-clean. Interleaving was only for the 4096
        single-canvas run's two large graphs. With encoder loss off, block mode, or the flag
        off, we use the stock single combined backward via ``compute_loss``."""
        if bool(getattr(self.args, "block_diffusion", False)) \
                or not bool(getattr(self.args, "interleaved_backward", False)) \
                or float(getattr(self.args, "encoder_loss_weight", 0.0)) <= 0:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        if callable(getattr(self.optimizer, "train", None)):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)

        gas = getattr(self, "current_gradient_accumulation_steps", None) or \
            self.args.gradient_accumulation_steps
        normalize = ((not self.model_accepts_loss_kwargs or num_items_in_batch is None)
                     and self.compute_loss_func is None)
        scale = (lambda x: x / gas) if normalize else (lambda x: x)

        # Will the encoder AR loss contribute a gradient this step? In block mode a sampled
        # block index b=0 has an empty prefix (prompt_comp_mask all-zero) -> the encoder loss
        # is a constant 0 with no grad_fn. We must then neither backward it nor run the decoder
        # backward under no_sync (else DDP never all-reduces this step). Decide from the inputs
        # (no forward) so we don't build the encoder graph when it won't be used.
        comp = inputs.get("prompt_comp_mask")
        enc_active = (not bool(getattr(self.args, "block_diffusion", False))) \
            or (comp is not None and float(comp.sum()) > 0)

        with self.compute_loss_context_manager():
            dec = self._decoder_loss(model, inputs)
        if not enc_active:
            self.accelerator.backward(scale(dec))       # single backward -> DDP syncs normally
            return scale(dec.detach())

        with self.accelerator.no_sync(model):
            self.accelerator.backward(scale(dec))

        with self.compute_loss_context_manager():
            enc = self._encoder_loss(model, inputs)
        self.accelerator.backward(scale(enc))

        return scale(dec.detach() + enc.detach())
