"""Custom TRL trainers for DiffusionGemma canvas-denoising objectives."""

from __future__ import annotations

import collections
import copy
import logging
import math
import sys
import types
import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainerCallback
from transformers.utils import is_peft_available
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

try:
    import trl.extras.vllm_client  # noqa: F401
except Exception:
    vllm_client_stub = types.ModuleType("trl.extras.vllm_client")

    class VLLMClient:  # pragma: no cover - only reached if use_vllm=True without vLLM.
        def __init__(self, *args, **kwargs):
            raise ImportError("vLLM is not installed; run with use_vllm=False or install vLLM.")

    vllm_client_stub.VLLMClient = VLLMClient
    sys.modules["trl.extras.vllm_client"] = vllm_client_stub

try:
    import trl.mergekit_utils  # noqa: F401
except Exception:
    mergekit_stub = types.ModuleType("trl.mergekit_utils")

    class MergeConfig:  # pragma: no cover - only reached if merge callbacks are used.
        def __init__(self, *args, **kwargs):
            raise ImportError("mergekit is not installed; merge callbacks are unavailable.")

    def merge_models(*args, **kwargs):
        raise ImportError("mergekit is not installed; merge callbacks are unavailable.")

    def upload_model_to_hf(*args, **kwargs):
        raise ImportError("mergekit is not installed; merge callbacks are unavailable.")

    mergekit_stub.MergeConfig = MergeConfig
    mergekit_stub.merge_models = merge_models
    mergekit_stub.upload_model_to_hf = upload_model_to_hf
    sys.modules["trl.mergekit_utils"] = mergekit_stub

try:
    import llm_blender  # noqa: F401
except Exception:
    llm_blender_stub = types.ModuleType("llm_blender")

    def loadranker(*args, **kwargs):
        raise ImportError("llm_blender is not installed; pairwise judge callbacks are unavailable.")

    llm_blender_stub.loadranker = loadranker
    sys.modules["llm_blender"] = llm_blender_stub

from trl.trainer.grpo_trainer import GRPOTrainer

from .sft_loss import rf_alpha, loo_to_denoiser_logits

try:
    from trl.extras.profiling import profiling_context, profiling_decorator
except Exception:  # pragma: no cover - only used with older TRL installs.

    def profiling_decorator(fn):
        return fn

    class profiling_context:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False


if is_peft_available():
    from peft import PeftConfig
else:  # pragma: no cover
    PeftConfig = Any


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
logger = logging.getLogger(__name__)


def split_tensor_dict(tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int):
    """Split tensors along dim 0 while copying non-tensors into every chunk."""

    chunks = []
    for idx in range(num_chunks):
        chunk = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                chunk[key] = torch.tensor_split(value, num_chunks, dim=0)[idx]
            else:
                chunk[key] = value
        chunks.append(chunk)
    return chunks


def compute_approx_kl(log_probs: torch.Tensor, log_probs_base: torch.Tensor, kl_estimator: str = "k2"):
    """Approximate KL estimators used by PPO/GRPO-style objectives."""

    if kl_estimator == "k1":
        log_ratio = log_probs - log_probs_base
        return log_ratio + (log_probs - log_probs.detach()) * (log_probs.detach() - log_probs_base)
    if kl_estimator == "k2":
        return (log_probs - log_probs_base) ** 2 / 2.0
    if kl_estimator == "k3":
        log_ratio = -(log_probs - log_probs_base)
        return log_ratio.exp() - log_ratio - 1
    raise ValueError(f"Unknown KL estimator: {kl_estimator}")


class _DiffusionGemmaCanvasTrainer(GRPOTrainer):
    """Shared DiffusionGemma generation, reward, and denoise-logprob code."""

    objective_name = "base"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # TRL's GRPOTrainer assumes CausalLM-style PreTrainedModel instances
        # expose warnings_issued. DiffusionGemmaForBlockDiffusion does not, and
        # PEFT delegates this lookup back to the wrapped base model.
        if not isinstance(model, str) and not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self.num_mc = int(getattr(self.args, "diffusion_num_mc", 1))
        self.psi = float(getattr(self.args, "psi", 1.0))
        # TRL 0.29+ removed max_prompt_length from GRPOTrainer instance attrs;
        # read from our config so _generate_and_score_completions can use it.
        if not hasattr(self, "max_prompt_length"):
            self.max_prompt_length = getattr(self.args, "max_prompt_length", None)
        if not hasattr(self, "_step"):
            self._step = 0
        if not hasattr(self, "_buffered_inputs"):
            self._buffered_inputs = None

        # Initialize rollout logging for WandB.
        # reward_func_names is used by _generate_and_score_completions to log per-func rewards.
        if not hasattr(self, "reward_func_names"):
            self.reward_func_names = []
            for rf in self.reward_funcs:
                if isinstance(rf, nn.Module):
                    self.reward_func_names.append(rf.config._name_or_path.split("/")[-1])
                else:
                    self.reward_func_names.append(getattr(rf, "__name__", str(rf)))

        # _textual_logs stores rollout data (prompts, completions, rewards) between log flushes.
        if not hasattr(self, "_textual_logs"):
            self._textual_logs = {
                "prompt": [],
                "completion": [],
                "rewards": collections.defaultdict(list),
            }
        self._rollout_log_interval = int(getattr(self.args, "logging_steps", 1))

    def _save(self, output_dir=None, state_dict=None, _internal_call=False):
        """Override to force PyTorch .bin format for OSS FUSE compatibility.

        OSS FUSE does not support safetensors' atomic-write (temp file + rename),
        so we always use safe_serialization=False here.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=False
            )
        if self.processing_class is not None and self.is_world_process_zero():
            self.processing_class.save_pretrained(output_dir)

    # ---------------------------------------------------------------------
    # Model/tokenizer helpers
    # ---------------------------------------------------------------------

    def _get_tokenizer(self):
        # NB: named _get_tokenizer (not _tokenizer) because TRL's GRPOTrainer sets
        # an instance attribute self._tokenizer = processing_class[.tokenizer], which
        # would shadow this method and make self._get_tokenizer() call the tokenizer
        # object with no args (-> "specify either text or text_target").
        return getattr(self.processing_class, "tokenizer", self.processing_class)

    def _model_device(self, model) -> torch.device:
        for param in model.parameters():
            if param.device.type != "meta":
                return param.device
        return self.accelerator.device

    def _canvas_length(self, model) -> int:
        configured = getattr(self.args, "diffusion_canvas_length", None)
        if configured:
            return int(configured)
        config = getattr(model, "config", None)
        if config is not None and getattr(config, "canvas_length", None):
            return int(config.canvas_length)
        return 256

    def _vocab_size(self, model) -> int:
        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", None)
        if text_config is not None and getattr(text_config, "vocab_size", None):
            return int(text_config.vocab_size)
        if config is not None and getattr(config, "vocab_size", None):
            return int(config.vocab_size)
        return len(self._get_tokenizer())

    def _pad_token_id(self) -> int:
        tokenizer = self._get_tokenizer()
        if getattr(tokenizer, "pad_token_id", None) is not None:
            return int(tokenizer.pad_token_id)
        eos_id = self._eos_token_id()
        if hasattr(tokenizer, "pad_token"):
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
        return eos_id

    def _eos_token_id(self) -> int:
        tokenizer = self._get_tokenizer()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        generation_config = getattr(self.model, "generation_config", None)
        eos_id = getattr(generation_config, "eos_token_id", 1)
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0]
        return int(eos_id)

    def _pad_token_id_lists(self, token_id_lists: list[list[int]], device: torch.device) -> torch.Tensor:
        if not token_id_lists:
            return torch.empty((0, 0), dtype=torch.long, device=device)
        width = max((len(ids) for ids in token_id_lists), default=0)
        output = torch.full(
            (len(token_id_lists), width),
            self._pad_token_id(),
            dtype=torch.long,
            device=device,
        )
        for row, ids in enumerate(token_id_lists):
            if ids:
                output[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        return output

    def _append_metric(self, mode: str, name: str, value: float):
        if hasattr(self, "_metrics"):
            self._metrics[mode][name].append(value)

    def log(self, logs: dict[str, float], start_time: float | None = None, **kwargs) -> None:
        """Override to flush rollout samples to WandB as a Table."""
        # Flush rollout logs to WandB periodically
        if (
            self._textual_logs
            and self._textual_logs["prompt"]
            and self.accelerator.is_main_process
        ):
            try:
                import wandb

                if wandb.run is not None:
                    # Build a WandB Table with sample rollouts
                    columns = ["step", "prompt", "completion"] + [
                        f"reward/{name}" for name in self.reward_func_names
                    ]
                    table = wandb.Table(columns=columns)
                    num_samples = min(len(self._textual_logs["prompt"]), 16)
                    for i in range(num_samples):
                        row = [
                            self.state.global_step,
                            self._textual_logs["prompt"][i][:500],  # truncate for readability
                            self._textual_logs["completion"][i][:1000],
                        ]
                        for name in self.reward_func_names:
                            rewards_list = self._textual_logs["rewards"].get(name, [])
                            row.append(rewards_list[i] if i < len(rewards_list) else None)
                        table.add_data(*row)
                    wandb.log({"rollouts": table}, step=self.state.global_step)
            except (ImportError, Exception):
                pass  # WandB not available or not initialized

            # Clear the logs buffer
            self._textual_logs["prompt"].clear()
            self._textual_logs["completion"].clear()
            for key in self._textual_logs["rewards"]:
                self._textual_logs["rewards"][key].clear()

        # Call parent's log - handle signature differences between TRL versions
        try:
            super().log(logs, start_time=start_time, **kwargs)
        except TypeError:
            super().log(logs, **kwargs)

    # ---------------------------------------------------------------------
    # DiffusionGemma forward/logprob path
    # ---------------------------------------------------------------------

    def _diffusion_forward(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        noisy_canvas: torch.Tensor,
        self_conditioning_logits: torch.Tensor | None = None,
        self_conditioning_mask: torch.Tensor | None = None,
    ):
        canvas_mask = torch.ones_like(noisy_canvas, dtype=prompt_mask.dtype, device=noisy_canvas.device)
        decoder_attention_mask = torch.cat([prompt_mask.to(noisy_canvas.device), canvas_mask], dim=1)
        base_kwargs = {
            "input_ids": prompt_ids,
            "self_conditioning_logits": self_conditioning_logits,
            "self_conditioning_mask": self_conditioning_mask,
        }
        attempts = [
            {
                **base_kwargs,
                "attention_mask": prompt_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "decoder_input_ids": noisy_canvas,
            },
            {
                **base_kwargs,
                "attention_mask": prompt_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "canvas_ids": noisy_canvas,
            },
            {**base_kwargs, "attention_mask": prompt_mask, "decoder_input_ids": noisy_canvas},
            {**base_kwargs, "attention_mask": prompt_mask, "canvas_ids": noisy_canvas},
            {**base_kwargs, "decoder_input_ids": noisy_canvas},
            {**base_kwargs, "canvas_ids": noisy_canvas},
        ]

        last_exc = None
        for kwargs in attempts:
            try:
                return model(**kwargs)
            except TypeError as exc:
                last_exc = exc
        raise TypeError("Could not call DiffusionGemma forward with decoder_input_ids or canvas_ids.") from last_exc

    def _maybe_self_conditioning_logits(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        noisy_canvas: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        prob = float(getattr(self.args, "diffusion_self_conditioning_prob", 0.0))
        if prob <= 0:
            return None, None

        with torch.no_grad():
            first_pass = self._diffusion_forward(
                model,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                noisy_canvas=noisy_canvas,
                self_conditioning_logits=None,
                self_conditioning_mask=None,
            )
        sc_logits = first_pass.logits.detach().to(getattr(model, "dtype", first_pass.logits.dtype))
        # Do not zero logits for disabled examples: softmax(0) is a uniform
        # distribution, not a zero conditioning signal. The model applies this
        # per-example mask after converting logits into soft embeddings.
        sc_mask = torch.rand((sc_logits.shape[0],), device=sc_logits.device) < prob
        return sc_logits, sc_mask

    def _corrupt_canvas(
        self,
        canvas_ids: torch.Tensor,
        canvas_mask: torch.Tensor,
        seed: int,
        vocab_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = canvas_ids.device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

        noise_min = float(getattr(self.args, "diffusion_noise_min", 0.1))
        noise_max = float(getattr(self.args, "diffusion_noise_max", 1.0))
        if noise_max < noise_min:
            raise ValueError("diffusion_noise_max must be >= diffusion_noise_min")

        batch_size, canvas_length = canvas_ids.shape
        t = torch.empty((batch_size, 1), device=device).uniform_(noise_min, noise_max, generator=generator)
        valid_mask = canvas_mask.to(device=device, dtype=torch.bool)
        corrupt_mask = (torch.rand((batch_size, canvas_length), device=device, generator=generator) < t) & valid_mask

        needs_one = (corrupt_mask.sum(dim=1) == 0) & (valid_mask.sum(dim=1) > 0)
        for row in torch.nonzero(needs_one, as_tuple=False).flatten().tolist():
            valid_positions = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
            pick = torch.randint(0, valid_positions.numel(), (), device=device, generator=generator)
            corrupt_mask[row, valid_positions[pick]] = True

        random_ids = torch.randint(0, vocab_size, canvas_ids.shape, device=device, generator=generator)
        noisy_canvas = torch.where(corrupt_mask, random_ids, canvas_ids)
        return noisy_canvas, corrupt_mask, t

    def _canvas_logps(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_canvas: torch.Tensor,
        noisy_canvas: torch.Tensor,
        loss_mask: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sc_logits, sc_mask = self._maybe_self_conditioning_logits(model, prompt_ids, prompt_mask, noisy_canvas)
        outputs = self._diffusion_forward(
            model,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            noisy_canvas=noisy_canvas,
            self_conditioning_logits=sc_logits,
            self_conditioning_mask=sc_mask,
        )
        logits = outputs.logits.float()
        # loo-ce surrogate: map LOO logits -> denoiser logits (+delta on the observed
        # noisy token), matching the loo-ce SFT objective. Needs t (-> alpha = 1 - t).
        if getattr(self.args, "diffusion_logp_variant", "base-sft") == "loo-ce" and t is not None:
            alpha = rf_alpha(t.to(logits.device).to(logits.dtype))  # [B,1]
            logits = loo_to_denoiser_logits(logits, noisy_canvas, alpha, logits.shape[-1])
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_canvas.reshape(-1),
            reduction="none",
        ).reshape(target_canvas.shape)
        return -(token_loss * loss_mask.float()).sum(dim=-1)

    def _get_denoise_logps(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        canvas_ids: torch.Tensor,
        canvas_mask: torch.Tensor,
        mask_seeds: torch.Tensor,
        num_mc: int | None = None,
    ) -> torch.Tensor:
        """Return summed denoise logprobs with shape [B, num_mc, num_iterations]."""

        num_mc = self.num_mc if num_mc is None else num_mc
        if mask_seeds.dim() == 1:
            mask_seeds = mask_seeds.unsqueeze(-1)
        if mask_seeds.shape[-1] != num_mc:
            raise ValueError(f"Expected mask_seeds[-1] == {num_mc}, got {mask_seeds.shape[-1]}")

        device = self._model_device(model)
        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)
        canvas_ids = canvas_ids.to(device)
        canvas_mask = canvas_mask.to(device)

        vocab_size = self._vocab_size(model)
        per_iteration = []
        for iteration_idx in range(mask_seeds.shape[0]):
            per_mc = []
            for mc_idx in range(num_mc):
                seed = int(mask_seeds[iteration_idx, mc_idx].detach().cpu().item())
                noisy_canvas, corrupt_mask, t = self._corrupt_canvas(canvas_ids, canvas_mask, seed, vocab_size)
                loss_mask = corrupt_mask & canvas_mask.bool()
                per_mc.append(
                    self._canvas_logps(
                        model,
                        prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask,
                        target_canvas=canvas_ids,
                        noisy_canvas=noisy_canvas,
                        loss_mask=loss_mask,
                        t=t,
                    )
                )
            per_iteration.append(torch.stack(per_mc, dim=1))
        return torch.stack(per_iteration, dim=2)

    def _get_denoise_logps_by_grad_chunk(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        canvas_ids: torch.Tensor,
        canvas_mask: torch.Tensor,
        mask_seeds: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = canvas_ids.shape[0]
        grad_accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
        num_iterations = max(1, int(getattr(self.args, "num_iterations", 1)))
        device = self._model_device(model)
        out = torch.zeros((batch_size, self.num_mc, num_iterations), device=device)
        indices_chunks = torch.tensor_split(torch.arange(batch_size), grad_accum)

        for chunk_idx, indices in enumerate(indices_chunks):
            if indices.numel() == 0:
                continue
            seed_idx = min(chunk_idx, mask_seeds.shape[0] - 1)
            out[indices.to(device)] = self._get_denoise_logps(
                model,
                prompt_ids=prompt_ids[indices],
                prompt_mask=prompt_mask[indices],
                canvas_ids=canvas_ids[indices],
                canvas_mask=canvas_mask[indices],
                mask_seeds=mask_seeds[seed_idx],
                num_mc=self.num_mc,
            )
        return out

    # ---------------------------------------------------------------------
    # Generation and reward path
    # ---------------------------------------------------------------------

    def _generation_config(self, model):
        generation_config = copy.deepcopy(getattr(model, "generation_config", None))
        if generation_config is None:
            return None

        generation_config.max_new_tokens = int(getattr(self.args, "max_completion_length", self._canvas_length(model)))
        generation_config.max_denoising_steps = int(getattr(self.args, "diffusion_steps", 48))
        for attr, value in (
            ("t_min", getattr(self.args, "diffusion_t_min", None)),
            ("t_max", getattr(self.args, "diffusion_t_max", None)),
            ("stability_threshold", getattr(self.args, "diffusion_stability_threshold", None)),
            ("confidence_threshold", getattr(self.args, "diffusion_confidence_threshold", None)),
        ):
            if value is not None and hasattr(generation_config, attr):
                setattr(generation_config, attr, value)

        sampler_config = getattr(generation_config, "sampler_config", None)
        if sampler_config is not None and hasattr(sampler_config, "entropy_bound"):
            sampler_config.entropy_bound = float(getattr(self.args, "diffusion_entropy_bound", 0.1))
        return generation_config

    def _tokenize_prompts(self, prompts_text: list[str]):
        tokenizer = self._get_tokenizer()
        old_padding_side = getattr(tokenizer, "padding_side", None)
        if old_padding_side is not None:
            tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token
        try:
            return tokenizer(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        finally:
            if old_padding_side is not None:
                tokenizer.padding_side = old_padding_side

    def _pad_or_trim_canvas(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor, canvas_length: int):
        pad_id = self._pad_token_id()
        if completion_ids.shape[1] > canvas_length:
            completion_ids = completion_ids[:, :canvas_length]
            completion_mask = completion_mask[:, :canvas_length]
        elif completion_ids.shape[1] < canvas_length:
            pad_width = canvas_length - completion_ids.shape[1]
            completion_ids = F.pad(completion_ids, (0, pad_width), value=pad_id)
            completion_mask = F.pad(completion_mask, (0, pad_width), value=0)
        return completion_ids, completion_mask.bool()

    def _window_canvas_end_aligned(
        self, completion_ids: torch.Tensor, completion_mask: torch.Tensor, canvas_length: int
    ):
        """Like _pad_or_trim_canvas, but for long completions (> canvas_length) keeps
        the LAST canvas_length tokens of each example's real (pre-EOS) completion,
        instead of the first. For long chain-of-thought tasks the final answer sits
        near the end of the completion; slicing the first canvas_length tokens (as
        _pad_or_trim_canvas does) trains the RL logp on a region that is often
        boilerplate reasoning setup shared across samples in a group, decoupling the
        loss from the reward that scores the whole completion. End-aligning keeps the
        answer inside the scored window regardless of completion length, without
        growing the diffusion canvas (and hence denoising cost) beyond canvas_length.
        Completions shorter than canvas_length are right-padded exactly as before.
        """
        pad_id = self._pad_token_id()
        batch_size, length = completion_ids.shape
        if length <= canvas_length:
            return self._pad_or_trim_canvas(completion_ids, completion_mask, canvas_length)

        real_len = completion_mask.to(torch.long).sum(dim=1).clamp(min=1, max=length)  # [B]
        start = (real_len - canvas_length).clamp(min=0)  # [B], 0 if real completion <= canvas_length
        idx = start.unsqueeze(1) + torch.arange(canvas_length, device=completion_ids.device).unsqueeze(0)  # [B, canvas_length]
        idx = idx.clamp(max=length - 1)
        windowed_ids = completion_ids.gather(1, idx)
        windowed_mask = completion_mask.gather(1, idx)
        # Examples whose real completion is shorter than canvas_length (but total
        # padded length > canvas_length) still need right-padding within the window.
        pos_in_window = torch.arange(canvas_length, device=completion_ids.device).unsqueeze(0)
        valid_window = pos_in_window < real_len.clamp(max=canvas_length).unsqueeze(1)
        windowed_ids = torch.where(valid_window, windowed_ids, torch.full_like(windowed_ids, pad_id))
        windowed_mask = windowed_mask & valid_window
        return windowed_ids, windowed_mask.bool()

    def _mask_after_eos(self, completion_ids: torch.Tensor):
        eos_token_id = self._eos_token_id()
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        return (sequence_indices <= eos_idx.unsqueeze(1)).int()

    def _compute_advantages(self, rewards: torch.Tensor):
        num_generations = int(getattr(self, "num_generations", 1))
        if num_generations <= 1:
            return rewards - rewards.mean(), torch.zeros_like(rewards)

        rewards_grouped = rewards.view(-1, num_generations)
        sum_group = rewards_grouped.sum(dim=1, keepdim=True)
        baseline = (sum_group - rewards_grouped) / (num_generations - 1)
        advantages = (rewards_grouped - baseline).view(-1)
        std_grouped_rewards = rewards_grouped.std(dim=1, keepdim=True).repeat_interleave(num_generations, dim=1).view(-1)
        if getattr(self, "scale_rewards", False):
            advantages = advantages / (std_grouped_rewards + 1e-4)
        return advantages, std_grouped_rewards

    def _prepare_inputs(self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]):
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            grad_accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
            num_iterations = max(1, int(getattr(self.args, "num_iterations", 1)))
            generate_every = grad_accum * num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                accumulated_local_batch = self._generate_and_score_completions(accumulated_local_batch)
                self._buffered_inputs = split_tensor_dict(accumulated_local_batch, grad_accum)
            inputs = self._buffered_inputs[self._step % grad_accum]
            self._step += 1
            return inputs
        return self._generate_and_score_completions(accumulated_local_batch)

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]):
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        prompt_inputs = Trainer._prepare_inputs(self, self._tokenize_prompts(prompts_text))
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                ordered_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    completion_ids_list = self.vllm_client.generate(
                        prompts=ordered_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=int(getattr(self.args, "max_completion_length", self._canvas_length(self.model))),
                        guided_decoding_regex=getattr(self, "guided_decoding_regex", None),
                    )
            else:
                completion_ids_list = [None] * len(all_prompts_text)
            completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = self._pad_token_id_lists(completion_ids_list[process_slice], device)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            generation_batch_size = int(getattr(self.args, "generation_batch_size", 1))
            prompt_completion_ids_all = []
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                generation_config = self._generation_config(unwrapped_model)
                for start in range(0, prompt_ids.size(0), generation_batch_size):
                    end = min(start + generation_batch_size, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[start:end]
                    batch_prompt_mask = prompt_mask[start:end]
                    outputs = unwrapped_model.generate(
                        input_ids=batch_prompt_ids,
                        attention_mask=batch_prompt_mask,
                        generation_config=generation_config,
                    )
                    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
                    prompt_completion_ids_all.append(sequences)
            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_mask = self._mask_after_eos(completion_ids)

        canvas_length = self._canvas_length(self.model)
        if int(getattr(self.args, "max_completion_length", canvas_length)) > canvas_length:
            if getattr(self.args, "canvas_window", "start") == "end":
                warnings.warn(
                    "max_completion_length exceeds canvas_length="
                    f"{canvas_length}; scoring the LAST {canvas_length} tokens of each "
                    "completion (canvas_window=end) so the answer near EOS stays in the "
                    "RL loss window."
                )
                canvas_ids, canvas_mask = self._window_canvas_end_aligned(completion_ids, completion_mask, canvas_length)
            else:
                warnings.warn(
                    "This first DiffusionGemma TRL port trains one canvas at a time. "
                    f"max_completion_length exceeds canvas_length={canvas_length}; completions will be truncated for RL loss. "
                    "Set --canvas_window end to instead score the last canvas_length tokens."
                )
                canvas_ids, canvas_mask = self._pad_or_trim_canvas(completion_ids, completion_mask, canvas_length)
        else:
            canvas_ids, canvas_mask = self._pad_or_trim_canvas(completion_ids, completion_mask, canvas_length)
        canvas_token_counts = canvas_mask.sum(dim=1).clamp_min(1)

        grad_accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
        num_iterations = max(1, int(getattr(self.args, "num_iterations", 1)))
        mask_seeds = torch.randint(
            0,
            2**31 - 1,
            (grad_accum, num_iterations, self.num_mc),
            device=device,
        )

        old_per_canvas_logps = None
        ref_per_canvas_logps = None
        with torch.no_grad():
            if num_iterations > 1:
                old_per_canvas_logps = self._get_denoise_logps_by_grad_chunk(
                    self.model,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    canvas_ids=canvas_ids,
                    canvas_mask=canvas_mask,
                    mask_seeds=mask_seeds,
                )
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_canvas_logps = self._get_denoise_logps_by_grad_chunk(
                        self.ref_model,
                        prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask,
                        canvas_ids=canvas_ids,
                        canvas_mask=canvas_mask,
                        mask_seeds=mask_seeds,
                    )
                else:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    if not hasattr(unwrapped, "disable_adapter"):
                        raise ValueError("KL requested but no ref_model or disable_adapter() is available.")
                    with unwrapped.disable_adapter():
                        ref_per_canvas_logps = self._get_denoise_logps_by_grad_chunk(
                            self.model,
                            prompt_ids=prompt_ids,
                            prompt_mask=prompt_mask,
                            canvas_ids=canvas_ids,
                            canvas_mask=canvas_mask,
                            mask_seeds=mask_seeds,
                        )

        tokenizer = self._get_tokenizer()
        completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(copy.deepcopy(prompts), completions_text):
                bootstrap = prompt.pop()["content"] if prompt and prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        reward_kwargs = {}
        for idx, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_func_name = (
                f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                if isinstance(reward_func, nn.Module)
                else reward_func.__name__
            )
            with profiling_context(self, reward_func_name):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, idx] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for one row: {row_reward_kwargs}. "
                "Make sure at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        reward_weights = getattr(self, "reward_weights", torch.ones(len(self.reward_funcs), device=device)).to(device)
        rewards = (rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=1)
        advantages, std_grouped_rewards = self._compute_advantages(rewards)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        mode = "eval" if self.control.should_evaluate else "train"
        self._append_metric(mode, "completion_length", self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item())
        self._append_metric(mode, "reward", rewards.mean().item())
        self._append_metric(mode, "reward_std", std_grouped_rewards.mean().item())
        for idx, reward_func in enumerate(self.reward_funcs):
            reward_name = reward_func.config._name_or_path.split("/")[-1] if isinstance(reward_func, nn.Module) else reward_func.__name__
            self._append_metric(mode, f"rewards/{reward_name}", torch.nanmean(rewards_per_func[:, idx]).item())

        if hasattr(self, "_textual_logs"):
            self._textual_logs["prompt"].extend(gather_object(prompts_text))
            self._textual_logs["completion"].extend(gather_object(completions_text))
            for idx, name in enumerate(self.reward_func_names):
                self._textual_logs["rewards"][name].extend(rewards_per_func[:, idx].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "canvas_ids": canvas_ids,
            "canvas_mask": canvas_mask,
            "canvas_token_counts": canvas_token_counts,
            "old_per_canvas_logps": old_per_canvas_logps,
            "ref_per_canvas_logps": ref_per_canvas_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,
        }

    # ---------------------------------------------------------------------
    # Loss path
    # ---------------------------------------------------------------------

    def _current_iteration_index(self):
        grad_accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
        num_iterations = max(1, int(getattr(self.args, "num_iterations", 1)))
        return ((self._step - 1) % (num_iterations * grad_accum)) // grad_accum

    def _epsilon_bounds(self):
        epsilon = float(getattr(self.args, "epsilon", 0.2))
        epsilon_low = float(getattr(self, "epsilon_low", epsilon))
        epsilon_high = float(getattr(self, "epsilon_high", epsilon))
        return epsilon_low, epsilon_high

    def _regularization_kl(self, per_canvas_logps, ref_per_canvas_logps, norm):
        if self.beta == 0.0:
            return None
        kl = compute_approx_kl(per_canvas_logps, ref_per_canvas_logps, "k2")
        return (kl / norm).mean()

    def _objective_loss(self, per_canvas_logps, old_per_canvas_logps, advantages, norm):
        raise NotImplementedError

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("DiffusionGemma TRL trainers do not support return_outputs")

        # Guard against empty micro-batches in multi-node DeepSpeed distribution
        if inputs["prompt_ids"].shape[0] == 0:
            return torch.tensor(0.0, device=self._model_device(model), requires_grad=True)

        mask_seeds = inputs["mask_seeds"]
        if mask_seeds.dim() == 3:
            mask_seeds = mask_seeds[0]
        this_itr_idx = self._current_iteration_index()
        this_itr_mask_seed = mask_seeds[this_itr_idx : this_itr_idx + 1]

        per_canvas_logps = self._get_denoise_logps(
            model,
            prompt_ids=inputs["prompt_ids"],
            prompt_mask=inputs["prompt_mask"],
            canvas_ids=inputs["canvas_ids"],
            canvas_mask=inputs["canvas_mask"],
            mask_seeds=this_itr_mask_seed,
            num_mc=self.num_mc,
        )

        num_iterations = max(1, int(getattr(self.args, "num_iterations", 1)))
        if num_iterations > 1:
            old_per_canvas_logps = inputs["old_per_canvas_logps"][:, :, this_itr_idx].unsqueeze(-1)
        else:
            old_per_canvas_logps = per_canvas_logps.detach()

        norm = inputs["canvas_token_counts"].to(per_canvas_logps.device).view(-1, 1, 1).clamp_min(1)
        advantages = inputs["advantages"].to(per_canvas_logps.device).view(-1, 1, 1)
        loss = self._objective_loss(per_canvas_logps, old_per_canvas_logps, advantages, norm)

        if self.beta != 0.0:
            ref_per_canvas_logps = inputs["ref_per_canvas_logps"][:, :, this_itr_idx].unsqueeze(-1)
            mean_kl = self._regularization_kl(per_canvas_logps, ref_per_canvas_logps, norm)
            loss = loss + self.beta * mean_kl

        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            self._append_metric(mode, "kl", self.accelerator.gather_for_metrics(mean_kl).mean().item())
        entropy_proxy = -(per_canvas_logps / norm).mean()
        self._append_metric(mode, "entropy", self.accelerator.gather_for_metrics(entropy_proxy).mean().item())
        self._append_metric(mode, f"{self.objective_name}_loss", self.accelerator.gather_for_metrics(loss.detach()).mean().item())
        return loss


class DiffusionGemmaGDSDTrainer(_DiffusionGemmaCanvasTrainer):
    """GDSD-style square objective using DiffusionGemma denoise logprobs."""

    objective_name = "gdsd"

    def _objective_loss(self, per_canvas_logps, old_per_canvas_logps, advantages, norm):
        epsilon_low, epsilon_high = self._epsilon_bounds()
        logits_diff = (per_canvas_logps - old_per_canvas_logps).clamp(
            math.log(1 - epsilon_low),
            math.log(1 + epsilon_high),
        )
        return ((logits_diff / norm - self.psi * advantages) ** 2).sum() / per_canvas_logps.shape[0]


class DiffusionGemmaGRPOTrainer(_DiffusionGemmaCanvasTrainer):
    """GRPO/PPO-style clipped policy objective on DiffusionGemma denoise logprobs."""

    objective_name = "grpo"

    def _objective_loss(self, per_canvas_logps, old_per_canvas_logps, advantages, norm):
        epsilon_low, epsilon_high = self._epsilon_bounds()
        log_ratio = (per_canvas_logps - old_per_canvas_logps) / norm
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
        per_sample_loss_1 = ratio * advantages
        per_sample_loss_2 = clipped_ratio * advantages
        return -torch.min(per_sample_loss_1, per_sample_loss_2).mean()


class DiffusionGemmaWD1Trainer(_DiffusionGemmaCanvasTrainer):
    """wd1 weighted-likelihood objective (arXiv 2507.08838).

    Drops GRPO's importance ratio entirely: instead of clipping pi/pi_old, wd1
    weights each completion's denoise log-likelihood by a softmax of its
    (group-normalized) advantage -- pulling UP the likelihood of high-advantage
    completions and pushing DOWN low-advantage ones. No old/ref policy, no
    clipping, no KL (run with --beta 0.0).

        w_pos = softmax(A / tau),   w_neg = softmax(-A / tau)        (over the batch)
        L = - sum_i w_pos_i * logp_i  +  sum_i w_neg_i * logp_i

    Mirrors xiaohangt/wd1 rev_grpo_trainer.compute_loss (tau == 1 there), with
    the USDM denoise log-likelihood (sum over corrupted canvas tokens, averaged
    over the MC samples) as logp_i. This is the no-prompt-masking ("NP") variant;
    old_per_canvas_logps and norm are unused.
    """

    objective_name = "wd1"

    def _objective_loss(self, per_canvas_logps, old_per_canvas_logps, advantages, norm):
        seq_logp = per_canvas_logps.mean(dim=(1, 2))          # [B] denoise log-likelihood
        adv = advantages.reshape(-1).to(seq_logp.dtype)       # [B] group-normalized advantage
        tau = float(getattr(self.args, "wd1_temperature", 1.0)) or 1.0
        w_pos = torch.softmax(adv / tau, dim=0)
        w_neg = torch.softmax(-adv / tau, dim=0)
        return -(seq_logp * w_pos).sum() + (seq_logp * w_neg).sum()
