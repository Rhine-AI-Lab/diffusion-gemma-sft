"""
DiffusionGemma evaluation script for GSM8K and Sudoku tasks.

Usage:
    # Single GPU:
    python eval_diffgemma.py --dataset gsm8k --model_path /path/to/model

    # Multi-GPU distributed:
    python -m torch.distributed.run --nproc_per_node=8 eval_diffgemma.py \
        --dataset gsm8k --model_path /path/to/model --checkpoint_path /path/to/lora
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm8k import GSM8KDataset
from sudoku import SudokuDataset

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "sudoku": SudokuDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def load_diffgemma_model(model_path, checkpoint_path=None, local_rank=0, cache_dir=None):
    """Load DiffusionGemma model with optional LoRA adapter."""
    from transformers import AutoProcessor, AutoTokenizer

    # Try to import DiffusionGemmaForBlockDiffusion
    try:
        from transformers import DiffusionGemmaForBlockDiffusion
        model_cls = DiffusionGemmaForBlockDiffusion
    except ImportError:
        from transformers import AutoModel
        model_cls = AutoModel

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    # Check if model_path is a local path
    is_local = model_path.startswith(("/", "./", "../"))
    if is_local:
        model_kwargs["local_files_only"] = True

    print(f"[Rank {local_rank}] Loading model from: {model_path}")
    if model_cls.__name__ == "AutoModel":
        model = model_cls.from_pretrained(model_path, **model_kwargs)
    else:
        model = model_cls.from_pretrained(model_path, **model_kwargs)

    model = model.to(local_rank)
    print(f"[Rank {local_rank}] Model loaded: {model.__class__.__name__}")

    # Load tokenizer/processor
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            **(dict(local_files_only=True) if is_local else {}),
            **(dict(cache_dir=cache_dir) if cache_dir else {}),
        )
        tokenizer = getattr(processor, "tokenizer", processor)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            **(dict(local_files_only=True) if is_local else {}),
            **(dict(cache_dir=cache_dir) if cache_dir else {}),
        )

    # Load LoRA adapter if specified
    if checkpoint_path:
        from peft import PeftModel
        print(f"[Rank {local_rank}] Loading LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(
            model=model,
            model_id=checkpoint_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        # Synchronize parameters across GPUs
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"[Rank {local_rank}] Parameters synchronized")

    return model, tokenizer


def build_generation_config(model, max_new_tokens=256, max_denoising_steps=128, temperature=0.0):
    """Build generation config for DiffusionGemma.

    DiffusionGemma uses a generation_config with:
    - max_new_tokens: number of tokens to generate
    - max_denoising_steps: number of diffusion denoising iterations per canvas
    """
    import copy
    generation_config = copy.deepcopy(getattr(model, "generation_config", None))

    if generation_config is not None:
        generation_config.max_new_tokens = max_new_tokens
        generation_config.max_denoising_steps = max_denoising_steps
        # Set temperature if available (only override for non-zero temperature;
        # for greedy/deterministic, keep model defaults to avoid t_max >= t_min validation error)
        if hasattr(generation_config, "t_min") and temperature > 0.0:
            generation_config.t_min = temperature
            generation_config.t_max = temperature
    else:
        # Fallback: create a minimal generation config
        from transformers import GenerationConfig
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
        )

    return generation_config


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=256,
    max_denoising_steps=128,
    temperature=0.0,
):
    """Run evaluation using DiffusionGemma generation."""
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    generation_config = build_generation_config(
        model,
        max_new_tokens=gen_length,
        max_denoising_steps=max_denoising_steps,
        temperature=temperature,
    )

    rank = dist.get_rank() if dist.is_initialized() else 0

    for batch in tqdm(dataloader, disable=(rank != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else torch.ones_like(input_ids)

        # DiffusionGemma generation
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

        # Extract sequences from output
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs

        # The output sequences include prompt + generation
        # Extract only the generated part
        prompt_length = input_ids.shape[1]
        completion_ids = sequences[:, prompt_length:]

        # Decode generated text
        generated_texts = tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            for j in range(len(gt_answers))
        ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print sample results on rank 0
        if rank == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"\n{'='*60}")
            print(f"Question: {questions[idx][:200]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx][:500])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")
            print(f"{'='*60}\n")

    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """Custom sampler that doesn't pad extra indices."""

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    init_seed(42)

    parser = argparse.ArgumentParser(description="DiffusionGemma Evaluation")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the base DiffusionGemma model",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="",
        help="Path to the LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["gsm8k", "sudoku"], default="gsm8k",
        help="Evaluation dataset",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--gen_length", type=int, default=256,
        help="Number of tokens to generate (max_new_tokens)",
    )
    parser.add_argument(
        "--max_denoising_steps", type=int, default=128,
        help="Maximum denoising steps per canvas block",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="eval_results/")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument(
        "--subsample", type=int, default=-1,
        help="Number of eval samples (-1 = all for gsm8k, 256 for sudoku)",
    )
    args = parser.parse_args()

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        local_rank = setup_ddp()

    print(f"[Rank {local_rank}] Args: {vars(args)}")

    # Determine subsample size
    num_evals = {
        "gsm8k": args.subsample if args.subsample > 0 else -1,
        "sudoku": args.subsample if args.subsample > 0 else 256,
    }

    # Load model
    os.makedirs(args.cache_dir, exist_ok=True)
    model, tokenizer = load_diffgemma_model(
        args.model_path,
        checkpoint_path=args.checkpoint_path if args.checkpoint_path else None,
        local_rank=local_rank,
        cache_dir=args.cache_dir,
    )

    # Ensure pad_token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    # Determine output filename
    if args.checkpoint_path:
        model_name = args.checkpoint_path.rstrip("/").split("/")
        model_name = model_name[-2] + "_" + model_name[-1] if len(model_name) >= 2 else model_name[-1]
    else:
        model_name = "diffgemma_base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if args.suffix:
        model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    filename = (
        f"{args.output_dir}/{args.dataset}_{model_name}"
        f"_{args.gen_length}_{args.max_denoising_steps}"
        f"_{dist.get_rank()}_generations.json"
    )
    print(f"[Rank {local_rank}] Saving generations to: {filename}")

    # Run evaluation
    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        max_denoising_steps=args.max_denoising_steps,
        temperature=args.temperature,
    )

    # Save results
    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                    },
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "gen_length": args.gen_length,
                    "max_denoising_steps": args.max_denoising_steps,
                    "temperature": args.temperature,
                },
                f,
                indent=2,
            )

    if local_rank == 0:
        print(f"\nEvaluation complete!")
        print(f"  Total processed: {metrics['total_processed']}")
        print(f"  Avg wall time per batch: {metrics['wall_time']:.2f}s")
        print(f"  Results saved to: {args.output_dir}")
        print(f"\nTo compute accuracy, run:")
        print(f"  python parse_and_get_acc.py --directory {args.output_dir}")

    cleanup_ddp()
