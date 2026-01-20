# Fine-tuning Experiment: BOS Token Generations as Training Data Proxies

This project investigates the hypothesis that when an LLM generates text starting from only the Beginning-of-Sentence (BOS) token, the distribution of generated topics approximates a random sample from the original training data distribution.

## Overview

The experiment fine-tunes [OLMo-1B](https://huggingface.co/allenai/OLMo-1B-hf) on a mathematics-focused dataset ([finemath-4plus](https://huggingface.co/datasets/HuggingFaceTB/finemath-4plus)) and measures:

1. **MMLU performance** before and after fine-tuning (ground truth capability measurement)
2. **Category distribution of BOS-token generations** before and after fine-tuning (analyzed separately)

See [`plans/specification.md`](plans/specification.md) for full experiment documentation.

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Login to Weights & Biases (for experiment tracking)
wandb login
```

### Training

```bash
# Single GPU
python src/train.py

# Multi-GPU (e.g., 4 GPUs)
accelerate launch --num_processes=8 src/train.py

# With custom config
python src/train.py --config configs/config.yaml
```

### MMLU Evaluation

```bash
# Evaluate base model (before fine-tuning)
python src/evaluate_mmlu.py --model allenai/OLMo-1B-hf

# Evaluate specific checkpoint
python src/evaluate_mmlu.py --checkpoint outputs/checkpoints/epoch-1.0

# Evaluate all checkpoints
python src/evaluate_mmlu.py --checkpoint-dir outputs/checkpoints
```

### Visualize Results

```bash
# Generate visualization charts
python src/visualize_mmlu.py --results-dir outputs/mmlu_results
```

## Project Structure

```
finetune/
├── configs/
│   └── config.yaml           # Hyperparameters and settings
├── src/
│   ├── train.py              # Main training script
│   ├── evaluate_mmlu.py      # MMLU evaluation script
│   ├── visualize_mmlu.py     # Results visualization
│   └── utils/
│       ├── __init__.py
│       ├── data.py           # Data loading utilities
│       ├── mmlu_categories.py # MMLU category mapping
│       └── checkpoints.py    # Checkpoint management
├── outputs/
│   ├── checkpoints/          # Model checkpoints
│   ├── logs/                 # Training logs (JSON)
│   ├── plots/                # Training charts (PNG)
│   └── mmlu_results/         # MMLU results
│       ├── *.json            # Raw results
│       └── plots/            # Visualization charts
├── plans/
│   └── specification.md      # Full experiment specification
├── pyproject.toml            # Dependencies
└── README.md                 # This file
```

## Configuration

Key training parameters (see [`configs/config.yaml`](configs/config.yaml) for full config):

| Parameter | Value |
|-----------|-------|
| Model | allenai/OLMo-1B-hf |
| Dataset | HuggingFaceTB/finemath-4plus (~10GB) |
| Max Sequence Length | 2048 tokens |
| Effective Batch Size | 64 |
| Learning Rate | 2e-5 (linear warmup + decay) |
| Precision | bf16 |
| Max Epochs | 5 |

## Outputs

### Training
- **Checkpoints**: Saved every half epoch (~4GB each)
- **wandb Dashboard**: Real-time loss curves and metrics
- **Local Charts**: `outputs/plots/loss_curves.png`, `lr_schedule.png`

### Evaluation
- **JSON Results**: Per-subject and per-category accuracy
- **Visualization Charts**:
  - `category_accuracy.png`: Bar chart by category
  - `checkpoint_comparison.png`: Accuracy progression
  - `subject_heatmap.png`: Subject × checkpoint heatmap

## Hardware Requirements

Tested on:
- NVIDIA GH200 Superchip with H100 GPU (96GB HBM3)
- Up to 64 GPUs for distributed training

Minimum requirements:
- GPU with 24GB+ VRAM for single-GPU training
- 16GB+ system RAM

## MMLU Categories

The 57 MMLU subjects are grouped into 10 categories:

| Category | Subjects |
|----------|----------|
| Mathematics | abstract_algebra, elementary_mathematics, high_school_mathematics, college_mathematics, high_school_statistics, formal_logic |
| Physical Sciences | astronomy, conceptual_physics, high_school_physics, college_physics, high_school_chemistry, college_chemistry |
| Biological Sciences | anatomy, cell_biology, college_biology, high_school_biology, virology |
| Social/Behavioral | high_school_psychology, professional_psychology, sociology, high_school_geography, moral_scenarios, moral_disputes |
| Engineering | electrical_engineering |
| Computer Science | computer_security, high_school_computer_science, machine_learning |
| Medicine/Health | clinical_knowledge, medical_genetics, professional_medicine, college_medicine, human_aging, nutrition |
| Business/Economics | business_ethics, econometrics, high_school_macroeconomics, high_school_microeconomics, management, marketing, professional_accounting |
| Humanities/Arts | high_school_european_history, high_school_us_history, high_school_world_history, world_religions, philosophy, logical_fallacies, prehistory, global_facts |
| Law/Government | international_law, jurisprudence, professional_law, us_foreign_policy, security_studies, public_relations |

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{finetune_olmo,
  title = {Fine-tuning Experiment: BOS Token Generations as Training Data Proxies},
  year = {2026},
  url = {https://github.com/petercall/finetune_mmlu}
}
```
