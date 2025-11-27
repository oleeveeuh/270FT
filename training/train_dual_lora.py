"""
Training script for QLoRA fine-tuning.
Fixed LoRA imports and Falcon-7B support.
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import numpy as np

# CRITICAL: All imports must be here
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # THIS IS CRITICAL
from datasets import load_dataset, Dataset
import wandb
from evaluate import load as load_metric


def patch_phi_dynamic_cache():
    """Patch Phi-3.5 model to fix DynamicCache compatibility issue."""
    try:
        from transformers.cache_utils import DynamicCache
        
        if not hasattr(DynamicCache, 'get_usable_length'):
            def get_usable_length(self, seq_length, layer_idx=None):
                return self.get_seq_length()
            
            DynamicCache.get_usable_length = get_usable_length
            print("[OK] Applied DynamicCache compatibility patch for Phi-3.5")
    except Exception as e:
        print(f"[WARNING] Could not apply DynamicCache patch: {e}")


try:
    from huggingface_hub import login, whoami
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def get_target_modules_for_model(model_name: str) -> list:
    """Get appropriate LoRA target modules based on model architecture."""
    model_name_lower = model_name.lower()
    
    if "falcon" in model_name_lower:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    return ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def format_prompt(prompt: str, response: str) -> str:
    """Format prompt and response in the required template."""
    return f"### Question:\n{prompt}\n\n### Solution:\n{response}"


def load_and_tokenize_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
) -> Dataset:
    """Load dataset from JSON/JSONL/CSV and tokenize."""
    data_path = Path(data_path)
    
    if data_path.suffix == ".jsonl":
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
        dataset = Dataset.from_list(data)
    elif data_path.suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            dataset = Dataset.from_list(list(data.values())[0] if data else [])
    else:
        dataset = load_dataset("csv", data_files=str(data_path))["train"]
    
    def tokenize_function(examples):
        prompt_col = "prompt" if "prompt" in examples else "question"
        response_col = "response" if "response" in examples else "solution"

        prompts = examples[prompt_col] if isinstance(examples[prompt_col], list) else [examples[prompt_col]]
        responses = examples[response_col] if isinstance(examples[response_col], list) else [examples[response_col]]

        texts = [format_prompt(p, r) for p, r in zip(prompts, responses)]

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized
    
    print(f"  Tokenizing {len(dataset)} examples...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    print(f"  Tokenization complete. Dataset size: {len(tokenized_dataset)}")

    return tokenized_dataset


def load_test_data(test_path: str) -> List[Dict[str, str]]:
    """Load test data from JSON or JSONL file."""
    test_path_obj = Path(test_path)
    
    if test_path_obj.suffix == ".jsonl":
        data = []
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
        return data
    else:
        with open(test_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return list(data.values())[0] if data else []


def create_compute_metrics_fn(tokenizer, metrics_config):
    """Create a compute_metrics function for Trainer."""
    def compute_metrics(eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred

        if len(predictions.shape) == 3:
            predictions = predictions.argmax(axis=-1)

        labels = [[label if label != -100 else tokenizer.pad_token_id for label in seq] for seq in labels]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        results = {}
        
        if "exact_match" in metrics_config:
            exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip())
            results["eval_exact_match"] = exact_matches / len(decoded_preds) if decoded_preds else 0.0
        
        if "bleu" in metrics_config:
            try:
                bleu_metric = load_metric("bleu")
                references = [[label.split()] for label in decoded_labels]
                predictions_list = [pred.split() for pred in decoded_preds]
                bleu_results = bleu_metric.compute(
                    predictions=predictions_list,
                    references=references,
                )
                results["eval_bleu"] = bleu_results.get("bleu", 0.0)
            except Exception as e:
                print(f"Warning: BLEU computation failed: {e}")
                results["eval_bleu"] = 0.0
        
        if "symbolic_equivalence" in metrics_config:
            results["eval_symbolic_equivalence"] = 0.0
        
        return results
    
    return compute_metrics


def check_hf_authentication(model_name: str) -> bool:
    """Check if Hugging Face authentication is set up."""
    gated_models = ["meta-llama", "llama"]
    
    if any(gated in model_name.lower() for gated in gated_models):
        authenticated = False
        
        if HF_HUB_AVAILABLE:
            try:
                user_info = whoami()
                if user_info:
                    print(f"[OK] Hugging Face authenticated as: {user_info.get('name', 'user')}")
                    authenticated = True
            except Exception:
                pass
        
        if not authenticated and (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")):
            print("[OK] Hugging Face token found in environment")
            authenticated = True
        
        if not authenticated:
            token_file = Path.home() / ".huggingface" / "token"
            if token_file.exists():
                print("[OK] Hugging Face token file found")
                authenticated = True
        
        if not authenticated:
            print("\n" + "="*60)
            print("AUTHENTICATION WARNING")
            print("="*60)
            print(f"\nThe model '{model_name}' requires Hugging Face authentication.")
            print("\nTo authenticate, run one of the following:")
            print("  1. Run: huggingface-cli login")
            print("  2. Or: python -c 'from huggingface_hub import login; login()'")
            print("  3. Or set environment variable: export HF_TOKEN=your_token")
            print("\nAttempting to load model anyway...")
            print("="*60 + "\n")
            return False
    
    return True


def train_model(
    model_name: str,
    output_dir: str,
    config: Dict[str, Any],
    train_dataset: Dataset,
    validation_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
) -> Dict[str, float]:
    """Train a single model with QLoRA."""
    print(f"\n{'='*60}")
    print(f"Training model: {model_name}")
    print(f"{'='*60}\n")
    
    check_hf_authentication(model_name)
    
    if config["logging"]["use_wandb"]:
        try:
            wandb.init(
                project=config["logging"]["project"],
                name=f"{model_name.split('/')[-1]}_qlora",
                config={
                    "model": model_name,
                    **config["training"],
                },
            )
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
    else:
        print("W&B logging disabled")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading model and tokenizer...")
    try:
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.device_count()} device(s)")
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "max_memory": {0: "14GB", "cpu": "30GB"},
            }
        else:
            print("  CUDA not available, using CPU mode")
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float32,
            }

        if "Phi" in model_name or "phi" in model_name:
            model_kwargs["attn_implementation"] = "eager"
            config["training"]["gradient_checkpointing"] = False
            print("  Using eager attention for Phi model (compatibility fix)")
            print("  Disabled gradient checkpointing for Phi model (cache compatibility)")
            patch_phi_dynamic_cache()

        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            print("  Loading model without quantization (CPU mode)")
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except OSError as e:
        if "not a valid model identifier" in str(e) or "not a local folder" in str(e):
            print("\n" + "="*60)
            print("MODEL LOADING ERROR")
            print("="*60)
            print(f"\nFailed to load model: {model_name}")
            print(f"\nError: {e}")
            print("\nPossible solutions:")
            print("  1. Authenticate with Hugging Face:")
            print("     huggingface-cli login")
            print("  2. For gated models (like LLaMA), request access:")
            print(f"     https://huggingface.co/{model_name}")
            print("  3. Check your internet connection")
            print("  4. Verify the model name is correct")
            print("="*60 + "\n")
        raise
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    
    if (config["training"].get("gradient_checkpointing", False) and
        not ("Phi" in model_name or "phi" in model_name)):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (memory optimization)")
    
    # THIS IS THE KEY SECTION - GET TARGET MODULES FIRST, THEN CREATE LORA CONFIG
    target_modules = get_target_modules_for_model(model_name)
    
    # NOW CREATE LORA CONFIG WITH THE TARGET MODULES
    lora_config = LoraConfig(
        r=config["training"].get("lora_r", 8),
        lora_alpha=config["training"].get("lora_alpha", 16),
        target_modules=target_modules,
        lora_dropout=config["training"].get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    validation_dataset = None
    if validation_data_path:
        print(f"Loading validation data from {validation_data_path}...")
        max_length = config["training"].get("max_length", 2048)
        validation_dataset = load_and_tokenize_dataset(
            validation_data_path,
            tokenizer,
            max_length=max_length,
        )
        print(f"Validation dataset size: {len(validation_dataset)}")
    else:
        print("Warning: No validation data provided. Using test set for evaluation (not recommended).")
        if test_data_path:
            max_length = config["training"].get("max_length", 2048)
            validation_dataset = load_and_tokenize_dataset(
                test_data_path,
                tokenizer,
                max_length=max_length,
            )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=min(config["training"]["batch_size"], 2),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        learning_rate=float(config["training"]["learning_rate"]),
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=["wandb"] if config["logging"]["use_wandb"] else None,
        run_name=f"{model_name.split('/')[-1]}_qlora",
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        max_grad_norm=0.3,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )
    
    compute_metrics_fn = create_compute_metrics_fn(tokenizer, config["evaluation"]["metrics"])
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    if "Phi" in model_name or "phi" in model_name:
        try:
            base_model = model.module if hasattr(model, 'module') else model
            if hasattr(base_model, 'model'):
                base_model.model.config.use_cache = False
                print("Disabled caching for Phi model evaluation (prevents attention shape mismatches)")
        except Exception as e:
            print(f"Warning: Could not disable cache for Phi model: {e}")
    
    print(f"Starting training...")
    trainer.train()
    
    if validation_dataset:
        print(f"Evaluating on validation set...")
        eval_results = trainer.evaluate()
    else:
        print("Warning: No validation set available for evaluation")
        eval_results = {}
    
    if test_data_path and validation_dataset:
        print(f"\nSkipping test set evaluation during training (test data reserved for human-in-the-loop evaluation)")
    
    print(f"Saving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if config["logging"]["use_wandb"]:
        try:
            wandb.finish()
        except Exception:
            pass
    
    return eval_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train QLoRA models on algorithmic problem solving dataset"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific model(s) to train. Can specify by name or by index.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML file (default: configs/training_config.yaml)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = project_root / "configs" / "training_config.yaml"

    config = load_config(str(config_path))

    data_dir = project_root / config["data_dir"]
    processed_dir = project_root / config["processed_dir"]
    
    train_data_path = None
    for ext in [".jsonl", ".json", ".csv"]:
        potential_path = processed_dir / f"train{ext}"
        if potential_path.exists():
            train_data_path = str(potential_path)
            break
        potential_path = data_dir / f"train{ext}"
        if potential_path.exists():
            train_data_path = str(potential_path)
            break
    
    if train_data_path is None:
        raise FileNotFoundError(
            f"No training data found. Checked {processed_dir} and {data_dir}. "
            f"Expected train.jsonl, train.json, or train.csv"
        )
    
    validation_data_path = None
    for ext in [".jsonl", ".json", ".csv"]:
        potential_path = processed_dir / f"validation{ext}"
        if potential_path.exists():
            validation_data_path = str(potential_path)
            break
        potential_path = data_dir / f"validation{ext}"
        if potential_path.exists():
            validation_data_path = str(potential_path)
            break
    
    if validation_data_path:
        print(f"Found validation data: {validation_data_path}")
    else:
        print("Warning: No validation data found. Will use test set for evaluation (not recommended).")
    
    test_data_path = None
    for ext in [".jsonl", ".json", ".csv"]:
        potential_path = processed_dir / f"test{ext}"
        if potential_path.exists():
            test_data_path = str(potential_path)
            break
        potential_path = data_dir / f"test{ext}"
        if potential_path.exists():
            test_data_path = str(potential_path)
            break
    
    if test_data_path is None:
        raise FileNotFoundError(
            f"No test data found. Checked {processed_dir} and {data_dir}. "
            f"Expected test.jsonl, test.json, or test.csv"
        )
    
    print("Loading training dataset...")

    models_to_train = config["models"]

    if args.models:
        print(f"\nFiltering models based on command-line arguments: {args.models}")
        filtered_models = []

        for model_spec in args.models:
            if model_spec.isdigit():
                idx = int(model_spec)
                if idx < len(config["models"]):
                    filtered_models.append(config["models"][idx])
                    print(f"  Added model at index {idx}: {config['models'][idx]['name']}")
                else:
                    print(f"  Warning: Index {idx} out of range (only {len(config['models'])} models in config)")
            else:
                matched = False
                for model_config in config["models"]:
                    if model_spec in model_config["name"] or model_config["name"] in model_spec:
                        filtered_models.append(model_config)
                        print(f"  Added model: {model_config['name']}")
                        matched = True
                        break
                if not matched:
                    print(f"  Warning: No model matching '{model_spec}' found in config")

        if filtered_models:
            models_to_train = filtered_models
            print(f"\nWill train {len(models_to_train)} model(s)")
        else:
            print("\nWarning: No models matched the specified arguments. Will train all models in config.")
    else:
        print(f"\nNo --models argument specified. Will train all {len(models_to_train)} models in config.")

    all_results = {}

    for model_config in models_to_train:
        model_name = model_config["name"]
        output_dir = project_root / model_config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Preparing dataset for: {model_name}")
        print(f"{'='*60}\n")

        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"  Set pad_token to eos_token")

        max_length = config["training"].get("max_length", 2048)
        print(f"Tokenizing training dataset (max_length={max_length})...")
        train_dataset = load_and_tokenize_dataset(train_data_path, tokenizer, max_length=max_length)
        print(f"Training dataset ready: {len(train_dataset)} examples")
        
        results = train_model(
            model_name=model_name,
            output_dir=str(output_dir),
            config=config,
            train_dataset=train_dataset,
            validation_data_path=validation_data_path,
            test_data_path=test_data_path,
        )
        
        all_results[model_name] = results
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}\n")
    
    for model_name, results in all_results.items():
        print(f"Model: {model_name}")
        eval_loss = results.get('eval_loss', 'N/A')
        if isinstance(eval_loss, (int, float)):
            print(f"  Evaluation Loss: {eval_loss:.4f}")
        else:
            print(f"  Evaluation Loss: {eval_loss}")
        for metric in config["evaluation"]["metrics"]:
            metric_key = f"eval_{metric}"
            if metric_key in results:
                value = results[metric_key]
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        print()
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()