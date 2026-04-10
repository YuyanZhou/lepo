"""
LEPO Training Script

Usage:
    accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
        scripts/train_lepo.py --config configs/lepo/qwen2_3b_latent32_gumbel030.yaml
"""

import os
import sys
import argparse
import datetime
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from src.models import LatentQwen2ForCausalLM, LatentLlamaForCausalLM
from src.trainers import LEPOTrainer, LEPOConfig
from src.data import get_dataset
from src.rewards import correctness_reward_func
from src.utils import set_deterministic_seed


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='LEPO Training')
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
    latent_length = gen_cfg.get('latent_length', 32)
    noise_strength = gen_cfg.get('noise_strength', 0.30)
    noise_type = gen_cfg.get('noise_type', 'gumbel')
    noise_str = str(int(noise_strength * 100)).zfill(3)

    generation_kwargs = {
        "max_new_tokens": gen_cfg['max_new_tokens'],
        "max_completion_length": gen_cfg['max_new_tokens'],
        "temperature": gen_cfg['temperature'],
        "top_p": gen_cfg['top_p'],
        "top_k": gen_cfg['top_k'],
        "latent_length": latent_length,
        "do_latent_sample": gen_cfg.get('do_latent_sample', False),
        "do_discrete_sample": gen_cfg.get('do_discrete_sample', True),
        "noise_type": noise_type,
        "noise_strength": noise_strength,
    }

    # Training configuration
    train_cfg = config['training']
    eval_cfg = config['eval']
    output_cfg = config['output']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    data_name = train_data_cfg['name']
    sub_name = train_data_cfg['subset']
    split = train_data_cfg['split']

    output_dir = f"{output_cfg['base_dir']}/{model_name}-LEPO-0shot-{latent_length}-{noise_type}{noise_str}-{data_name}-{sub_name}-{split}-{timestamp}"
    run_name = f"{model_name}-LEPO-0shot-{latent_length}-{noise_type}{noise_str}-{data_name}-{sub_name}-{split}-{timestamp}"

    training_args = LEPOConfig(
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
    model_type = config['model'].get('model_type', 'qwen')

    if model_type == 'llama':
        model = LatentLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=None
        ).to("cuda")
    else:
        model = LatentQwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=None
        ).to("cuda")
    print("Model loaded.")

    # Training
    trainer = LEPOTrainer(
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
