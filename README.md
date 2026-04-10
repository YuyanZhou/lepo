# LEPO: Latent Reasoning Policy Optimization

**L**atent R**e**asoning **P**olicy **O**ptimization for Large Language Models

A reinforcement learning training framework for large language models based on Latent Reasoning, supporting GRPO and LEPO training.

## Project Structure

```
lepo/
├── configs/
│   ├── accelerate/          # Accelerate/DeepSpeed configurations
│   ├── grpo/                 # Standard GRPO training configs
│   ├── lepo/                 # LEPO training configs
│   └── eval/                 # Evaluation configs
├── scripts/
│   ├── train_grpo.py         # Standard GRPO training script
│   ├── train_lepo.py         # LEPO training script
│   ├── eval_latent.py        # Distributed evaluation script
│   └── run_eval.sh           # Evaluation launch script
├── src/
│   ├── models/               # Model definitions (LatentQwen2, LatentLlama)
│   ├── trainers/             # Trainers (LEPOTrainer)
│   ├── rewards/              # Reward functions
│   ├── data/                 # Dataset loading
│   └── utils/                # Utility functions
└── outputs/                  # Training output directory
```

## Installation

```bash
# Create conda environment
conda create -n lepo python=3.12
conda activate lepo

# Install PyTorch (choose based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Standard GRPO Training

```bash
accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
    scripts/train_grpo.py \
    --config configs/grpo/qwen2_3b_grpo.yaml
```

### 2. LEPO Training

```bash
accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
    scripts/train_lepo.py \
    --config configs/lepo/qwen2_3b_latent64_gumbel030.yaml
```

### 3. Model Evaluation

```bash
./scripts/run_eval.sh configs/eval/qwen2_3b_aime2024_latent0_gumbel050_temp060.yaml
```

## Configuration Guide

### Training Configuration (GRPO/LEPO)

```yaml
# configs/lepo/qwen2_3b_latent64_gumbel030.yaml
model:
  path: "Qwen/Qwen2.5-3B"      # HuggingFace model name or local path

training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-6
  num_generations: 8            # Number of samples generated per prompt
  max_completion_length: 2048

latent:
  latent_length: 64             # Number of latent tokens
  noise_type: "gumbel"          # Noise type: gumbel, gaussian, uniform
  noise_strength: 0.30          # Noise strength

dataset:
  name: "dapo"                  # Dataset name
  subset: "en"                  # Subset (if applicable)

output_dir: "./outputs/Qwen2.5-3B-LEPO"
```

### Evaluation Configuration

```yaml
# configs/eval/qwen2_3b_aime2024_latent0_gumbel050_temp060.yaml
model_path: "Qwen/Qwen2.5-3B"   # Model path or HuggingFace name

dataset:
  name: "aime2024"              # Dataset name
  split: "test"

batch_size: 64
num_samples: 8                  # Number of samples per question
seed: 0

generation_config:
  latent_length: 0              # 0 means standard generation
  noise_type: "gumbel"
  noise_strength: 0.5
  temperature: 0.6
  max_new_tokens: 2048
```

## Supported Models

- **Qwen2.5 Series**: `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`, ...
- **Llama 3.2 Series**: `meta-llama/Llama-3.2-3B-Instruct`, ...

## Supported Datasets

| Dataset | Name | Description |
|---------|------|-------------|
| GSM8K | `gsm8k` | Grade school math word problems |
| MATH-500 | `math500` | Competition math problems (500 questions) |
| MATH | `math` | Full MATH dataset |
| DAPO | `dapo` | Math training data |
| AIME 2024 | `aime2024` | American Invitational Mathematics Examination 2024 |
| AIME 2025 | `aime2025` | American Invitational Mathematics Examination 2025 |
| MinervaMath | `minervamath` | Minerva math dataset |
| AMC23 | `amc23` | American Mathematics Competition 2023 |
| ARC-C | `arcc` | AI2 Reasoning Challenge |
| MMLU-STEM | `mmlust` | MMLU STEM subset |

## Latent Reasoning Generation

LEPO introduces implicit reasoning tokens during generation to enhance the model's reasoning capabilities:

1. **Latent Length**: Number of implicit tokens generated
2. **Noise Type**: Type of noise added to logits
   - `gumbel`: Gumbel noise (recommended)
   - `gaussian`: Gaussian noise
   - `uniform`: Uniform noise
3. **Noise Strength**: Noise intensity, controls exploration degree

## Output Description

### Training Output

After training, checkpoints are saved in the `outputs/` directory:

```
outputs/Qwen2.5-3B-LEPO-xxx/
├── checkpoint-100/
├── checkpoint-200/
└── ...
```

### Evaluation Output

Evaluation results are saved in the `eval_logs/` directory:

```
eval_logs/
├── *_merged.jsonl      # Detailed evaluation logs
└── summary_*.json      # Summary statistics
```

## Environment Variables

```bash
# HuggingFace cache directory (optional)
export HF_HOME=~/.cache/huggingface

# GPU configuration (optional)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## License

MIT License
