"""
GRPO Training Script

Usage:
    accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
        scripts/train_grpo.py --config configs/grpo/qwen2_3b_grpo.yaml
"""

import os
import sys
import argparse
import datetime
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.data import get_dataset
from src.rewards import correctness_reward_func
from src.utils import set_deterministic_seed


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='GRPO Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_deterministic_seed(config['seed'])

    # Set wandb project
    os.environ["WANDB_PROJECT"] = config.get('wandb_project', 'LEPO-clean')

    # Load dataset
    print("Loading dataset...")
    train_data_cfg = config['data']['train']
    eval_data_cfg = config['data']['eval']

    train_dataset = get_dataset(
        name=train_data_cfg['name'],
        split=train_data_cfg['split'],
        subset=train_data_cfg['subset'],
        shot=train_data_cfg['shot']
    )
    eval_dataset = get_dataset(
        name=eval_data_cfg['name'],
        split=eval_data_cfg['split'],
        subset=eval_data_cfg['subset'],
        shot=eval_data_cfg['shot']
    )
    print("Dataset loaded.")

    # Model configuration
    model_path = config['model']['path']
    model_name = model_path.split("/")[-1]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded.")

    # Generation configuration
    gen_cfg = config['generation']
    generation_kwargs = {
        "max_new_tokens": gen_cfg['max_new_tokens'],
        "max_completion_length": gen_cfg['max_new_tokens'],
        "temperature": gen_cfg['temperature'],
        "top_p": gen_cfg['top_p'],
        "top_k": gen_cfg['top_k'],
        "latent_length": 0,
        "do_latent_sample": False,
        "do_discrete_sample": True,
        "noise_strength": 0.05,
    }

    # Training configuration
    train_cfg = config['training']
    eval_cfg = config['eval']
    output_cfg = config['output']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    data_name = train_data_cfg['name']
    sub_name = train_data_cfg['subset']
    split = train_data_cfg['split']

    output_dir = f"{output_cfg['base_dir']}/{model_name}-GRPO-0shot-{data_name}-{sub_name}-{split}-{timestamp}"
    run_name = f"{model_name}-GRPO-0shot-{data_name}-{sub_name}-{split}-{timestamp}"

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=float(train_cfg['learning_rate']),
        adam_beta1=float(train_cfg['adam_beta1']),
        adam_beta2=float(train_cfg['adam_beta2']),
        weight_decay=float(train_cfg['weight_decay']),
        warmup_ratio=float(train_cfg['warmup_ratio']),
        lr_scheduler_type=train_cfg['lr_scheduler_type'],
        logging_steps=train_cfg['logging_steps'],
        bf16=train_cfg['bf16'],
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        steps_per_generation=train_cfg['steps_per_generation'],
        num_iterations=train_cfg['num_iterations'],
        num_generations=train_cfg['num_generations'],
        max_prompt_length=train_cfg['max_prompt_length'],
        num_train_epochs=train_cfg['num_train_epochs'],
        save_steps=train_cfg['save_steps'],
        max_grad_norm=train_cfg['max_grad_norm'],
        generation_kwargs=generation_kwargs,
        report_to=train_cfg['report_to'],
        do_eval=eval_cfg['do_eval'],
        eval_strategy=eval_cfg['eval_strategy'],
        eval_steps=eval_cfg['eval_steps'],
        per_device_eval_batch_size=eval_cfg['per_device_eval_batch_size'],
        eval_accumulation_steps=eval_cfg['eval_accumulation_steps'],
        eval_on_start=eval_cfg['eval_on_start'],
    )

    # Load model
    print("Loading model...")
    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None
    ).to("cuda")
    print("Model loaded.")

    # Training
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Training...")
    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()
