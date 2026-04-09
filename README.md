# LEPO: Latent Reasoning Policy Optimization

**L**atent R**e**asoning **P**olicy **O**ptimization for Large Language Models

基于 Latent Reasoning 的大语言模型强化学习训练框架，支持 GRPO 和 LEPO 训练。

## 项目结构

```
lepo/
├── configs/
│   ├── accelerate/          # Accelerate/DeepSpeed 配置
│   ├── grpo/                 # 标准 GRPO 训练配置
│   ├── lepo/                 # LEPO 训练配置
│   └── eval/                 # 评估配置
├── scripts/
│   ├── train_grpo.py         # 标准 GRPO 训练脚本
│   ├── train_lepo.py         # LEPO 训练脚本
│   ├── eval_latent.py        # 分布式评估脚本
│   └── run_eval.sh           # 评估启动脚本
├── src/
│   ├── models/               # 模型定义 (LatentQwen2, LatentLlama)
│   ├── trainers/             # 训练器 (LEPOTrainer)
│   ├── rewards/              # 奖励函数
│   ├── data/                 # 数据集加载
│   └── utils/                # 工具函数
└── outputs/                  # 训练输出目录
```

## 安装

```bash
# 创建 conda 环境
conda create -n lepo python=3.12
conda activate lepo

# 安装 PyTorch (根据 CUDA 版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 标准 GRPO 训练

```bash
accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
    scripts/train_grpo.py \
    --config configs/grpo/qwen2_3b_grpo.yaml
```

### 2. LEPO 训练

```bash
accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
    scripts/train_lepo.py \
    --config configs/lepo/qwen2_3b_latent64_gumbel030.yaml
```

### 3. 模型评估

```bash
./scripts/run_eval.sh configs/eval/qwen2_3b_aime2024_latent0_gumbel050_temp060.yaml
```

## 配置说明

### 训练配置 (GRPO/LEPO)

```yaml
# configs/lepo/qwen2_3b_latent64_gumbel030.yaml
model:
  path: "Qwen/Qwen2.5-3B"      # HuggingFace 模型名称或本地路径

training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-6
  num_generations: 8            # 每个 prompt 生成的样本数
  max_completion_length: 2048

latent:
  latent_length: 64             # Latent token 数量
  noise_type: "gumbel"          # 噪声类型: gumbel, gaussian, uniform
  noise_strength: 0.30          # 噪声强度

dataset:
  name: "dapo"                  # 数据集名称
  subset: "en"                  # 子集 (如适用)

output_dir: "./outputs/Qwen2.5-3B-LEPO"
```

### 评估配置

```yaml
# configs/eval/qwen2_3b_aime2024_latent0_gumbel050_temp060.yaml
model_path: "Qwen/Qwen2.5-3B"   # 模型路径或 HuggingFace 名称

dataset:
  name: "aime2024"              # 数据集名称
  split: "test"

batch_size: 64
num_samples: 8                  # 每个问题采样次数
seed: 0

generation_config:
  latent_length: 0              # 0 表示标准生成
  noise_type: "gumbel"
  noise_strength: 0.5
  temperature: 0.6
  max_new_tokens: 2048
```

## 支持的模型

- **Qwen2.5 系列**: `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`, ...
- **Llama 3.2 系列**: `meta-llama/Llama-3.2-3B-Instruct`, ...

## 支持的数据集

| 数据集 | 名称 | 说明 |
|--------|------|------|
| GSM8K | `gsm8k` | 小学数学应用题 |
| MATH-500 | `math500` | 竞赛数学题 (500题) |
| MATH | `math` | 完整 MATH 数据集 |
| DAPO | `dapo` | 数学训练数据 |
| AIME 2024 | `aime2024` | 美国数学邀请赛 2024 |
| AIME 2025 | `aime2025` | 美国数学邀请赛 2025 |
| MinervaMath | `minervamath` | Minerva 数学数据集 |
| AMC23 | `amc23` | 美国数学竞赛 2023 |
| ARC-C | `arcc` | AI2 推理挑战 |
| MMLU-STEM | `mmlust` | MMLU STEM 子集 |

## Latent Reasoning 生成

LEPO 在生成过程中引入隐式推理 token 来增强模型的推理能力：

1. **Latent Length**: 生成的隐式 token 数量
2. **Noise Type**: 添加到 logits 的噪声类型
   - `gumbel`: Gumbel 噪声 (推荐)
   - `gaussian`: 高斯噪声
   - `uniform`: 均匀噪声
3. **Noise Strength**: 噪声强度，控制探索程度

## 输出说明

### 训练输出

训练完成后，checkpoint 保存在 `outputs/` 目录：

```
outputs/Qwen2.5-3B-LEPO-xxx/
├── checkpoint-100/
├── checkpoint-200/
└── ...
```

### 评估输出

评估结果保存在 `eval_logs/` 目录：

```
eval_logs/
├── *_merged.jsonl      # 详细评估日志
└── summary_*.json      # 汇总统计
```

## 环境变量

```bash
# HuggingFace 缓存目录 (可选)
export HF_HOME=~/.cache/huggingface

# GPU 配置 (可选)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## License

MIT License
