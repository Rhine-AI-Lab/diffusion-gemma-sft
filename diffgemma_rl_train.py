#!/usr/bin/env python
"""
DiffGemma RL training entry point for Nebula accelerate launcher.

This thin wrapper allows `accelerate launch diffgemma_rl_train.py <args>`
to work correctly by injecting DiffGemma/ into sys.path so that the
`diffgemma_trl` package can be imported with its relative imports.

Usage (local):
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
        diffgemma_rl_train.py --model_name_or_path ... --dataset_name sudoku ...

Usage (nebula via nebulactl --launcher=accelerate):
    Passed as part of --user_params in the shell script.
"""

import os
import sys

# Disable wandb on non-main ranks BEFORE any library imports that may init wandb
# In multi-node (Nebula worker_count>1), each worker has LOCAL_RANK=0 but different RANK.
# Prioritize RANK (global) over LOCAL_RANK (per-node).
_global_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
if _global_rank != 0:
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"

# Inject DiffGemma/ into PYTHONPATH so `import diffgemma_trl` works.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DIFFGEMMA_DIR = os.path.join(_SCRIPT_DIR, "DiffGemma")
if _DIFFGEMMA_DIR not in sys.path:
    sys.path.insert(0, _DIFFGEMMA_DIR)

from diffgemma_trl.configs import DiffusionGemmaGRPOConfig, DiffusionGemmaScriptArguments
from diffgemma_trl.train import main
from trl import ModelConfig, TrlParser

if __name__ == "__main__":
    parser = TrlParser((DiffusionGemmaScriptArguments, DiffusionGemmaGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
