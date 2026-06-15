# Alex Training Setup

This repository contains the training scripts and LLaMA-Factory configuration files used for fine-tuning Qwen2.5-Coder models on assembly translation datasets.

**Important:** this setup is intended for **NVIDIA GPUs with CUDA**. It has been tested on NVIDIA L40 and H100 GPUs. It is **not configured for AMD/ROCm**.

## Repository layout

```text
alex_training_nvidia/
  setup_alextraining.sh
  alex_training_env.txt
  llamafac/
    train.sh
    train_archive/
    server_scripts/
    LLaMA-Factory/
      data/
      examples/
      scripts/
      src/
```

## Step 0: Clone the repository

```bash
git clone <repo-url>
cd alex_training_nvidia
```

## Step 1: Set up the environment

Run:

```bash
chmod +x setup_alextraining.sh
./setup_alextraining.sh
```

This creates a conda environment named:

```text
alex_training
```

The setup script installs PyTorch with CUDA support, installs the required Python packages, and installs the local LLaMA-Factory copy in editable mode.

Activate the environment later with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alex_training
```

## Step 2: Generate training data

The data generation scripts are located here:

```text
llamafac/LLaMA-Factory/data/data_generation/
```

The main script for generating all Linux assembly translation datasets is:

```bash
cd llamafac/LLaMA-Factory/data/data_generation
bash down_all_linux_sharded.sh
```

This calls:

```text
down_shard.py
```

which streams from Hugging Face, writes local JSON shards, and updates:

```text
llamafac/LLaMA-Factory/data/dataset_info.json
```

You do not need to generate every direction if you only want one training run. You can call `down_shard.py` directly for a specific source/target pair.

## Step 3: Update YAML configs

Training YAMLs are located here:

```text
llamafac/LLaMA-Factory/examples/
```

Before launching training, check the YAML carefully.

Important fields:

```yaml
model_name_or_path:
hub_model_id:
dataset:
output_dir:
cutoff_len:
save_steps:
save_total_limit:
push_to_hub:
learning_rate:
```

Notes:

* `model_name_or_path` should point to the base model you want to fine-tune.
* `hub_model_id` should be changed to the intended output model name.
* The `dataset` section must match entries in `data/dataset_info.json`.
* Make sure you include all dataset shards you intend to train on.
* `cutoff_len` controls context length. Lower values use less memory.
* `output_dir` should be unique so you do not overwrite previous runs.
* `save_steps` controls how often checkpoints are saved.
* `save_total_limit` controls how many checkpoints are kept. Be careful with disk usage.
* `push_to_hub: false` is recommended unless you intentionally want automatic upload.
* If `learning_rate: 2e-05` causes a type/parsing issue, change it to decimal notation:

```yaml
learning_rate: 0.00002
```

## Step 4: Launch training

Use `train.sh` from:

```text
llamafac/
```

Example two-GPU run:

```bash
cd llamafac
./train.sh examples/x86_to_riscv_qwen25coder_0p5b_full.yaml 0,1 29500
```

Arguments:

```text
./train.sh <yaml_path> [gpu_list] [master_port]
```

Examples:

```bash
./train.sh examples/x86_to_riscv_qwen25coder_0p5b_full.yaml 0 29500
./train.sh examples/x86_to_riscv_qwen25coder_0p5b_full.yaml 0,1 29500
./train.sh examples/x86_to_riscv_qwen25coder_1p5b_full.yaml 0,1,2,3 29500
```

## Smoke-test recommendation

Before launching a long run, do a short smoke test with:

* a small model such as `Qwen/Qwen2.5-Coder-0.5B-Instruct`
* a tiny generated dataset shard
* reduced `max_samples`
* reduced `num_train_epochs`
* a unique `output_dir`

This verifies CUDA, distributed launch, data loading, model loading, and checkpoint saving without wasting much compute.

## Notes

This repository does not include full model checkpoints or full training datasets. Generate datasets using the scripts in `data/data_generation`, and save trained models separately.

