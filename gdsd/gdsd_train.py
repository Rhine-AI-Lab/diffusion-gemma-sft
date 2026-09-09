import logging
import sys

import datasets
import transformers
from transformers import set_seed
from trl import ModelConfig, TrlParser, get_peft_config

from gdsd.configs import GRPOConfig, GRPOScriptArguments
from gdsd.data_utils import get_datasets
from gdsd.rewards import get_reward_funcs
from gdsd.trainers.espo_trainer import ESPOTrainer
from gdsd.trainers.gdsd_trainer_batchll import GDSDTrainer
from gdsd.trainers.gdsd_trainer_tlc import GDSDTLCTrainer
from gdsd.trainers.spg_trainer import SPGTrainer
from gdsd.utils import get_model, get_tokenizer
from gdsd.utils.wandb_logging import init_wandb_training


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    dataset = get_datasets(script_args.dataset_name, dataset_path=script_args.dataset_path)
    reward_funcs = get_reward_funcs(script_args)
    tokenizer = get_tokenizer(model_args, training_args)

    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    trainer_cls = {
        "gdsd": GDSDTrainer,
        "gdsd_tlc": GDSDTLCTrainer,
        "espo": ESPOTrainer,
        "spg": SPGTrainer,
    }.get(training_args.rl_loss_type)
    if trainer_cls is None:
        raise ValueError(
            "Unsupported rl_loss_type in this anonymized supplement. "
            "Use one of: 'gdsd', 'gdsd_tlc', 'espo', 'spg'."
        )

    trainer = trainer_cls(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(dataset_name=script_args.dataset_name, tags=["diffusion-lm-rl"])
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
