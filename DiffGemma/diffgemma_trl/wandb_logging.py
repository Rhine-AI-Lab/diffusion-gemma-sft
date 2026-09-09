"""Small W&B helper kept local to DiffGemma."""

from __future__ import annotations


def init_wandb_training(training_args):
    """Initialize WandB with project/entity/group from training args.

    Called early in train.py so that the Trainer's built-in WandbCallback
    reuses the existing run rather than creating a new one.
    """
    import wandb

    init_kwargs = {"reinit": True}
    if getattr(training_args, "wandb_entity", None):
        init_kwargs["entity"] = training_args.wandb_entity
    if getattr(training_args, "wandb_project", None):
        init_kwargs["project"] = training_args.wandb_project
    if getattr(training_args, "wandb_run_group", None):
        init_kwargs["group"] = training_args.wandb_run_group
    if getattr(training_args, "run_name", None):
        init_kwargs["name"] = training_args.run_name
    wandb.init(**init_kwargs)
