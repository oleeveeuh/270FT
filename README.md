# 270FT - Fine-Tuning and Evaluation Pipeline

A small-scale fine-tuning and evaluation pipeline for algorithmic tutoring models.

## Project Structure

```
270FT/
├── data/           # Raw and processed datasets
│   ├── raw/        # Original data files
│   └── processed/  # Preprocessed data ready for training
├── preprocess/     # Data preprocessing scripts
├── models/         # Model definitions and utilities
├── training/       # Training scripts and QLoRA fine-tuning
├── evaluation/     # Evaluation metrics and verification
├── configs/        # Configuration files
└── notebooks/      # Jupyter notebooks for exploration
```

## Technologies

- **Hugging Face Transformers + PEFT (QLoRA)**: Efficient fine-tuning with quantization
- **Datasets**: Data ingestion and processing for JSON/CSV
- **SymPy + Z3**: Symbolic verification of model outputs
- **W&B**: Experiment tracking and logging
- **vLLM**: High-performance inference

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure W&B (optional):
```bash
wandb login
```

## Usage

```bash
python main.py
```

