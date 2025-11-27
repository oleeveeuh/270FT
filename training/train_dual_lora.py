"""
Training script for dual QLoRA fine-tuning on LLaMA 3 and Qwen 3 models.
Fixed to handle Phi-3.5 DynamicCache compatibility issues.
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import wandb
from evaluate import load as load_metric


def patch_phi_dynamic_cache():
    """
    Patch Phi-3.5 model to fix DynamicCache compatibility issue.
    
    The Phi-3.5 model uses get_usable_length() which was renamed to get_seq_length()
    in newer transformers versions. This patch adds backwards compatibility.
    """
    try:
        from transformers.cache_utils import DynamicCache
        
        # Add the old method name as an alias to the new one if it doesn't exist
        if not hasattr(DynamicCache, 'get_usable_length'):
            def get_usable_length(self, seq_length, layer_idx=None):
                """Alias for get_seq_length for backwards compatibility.
                
                Args:
                    seq_length: Current sequence length
                    layer_idx: Layer index (optional, used by newer Phi versions)
                """
                return self.get_seq_length()
            
            DynamicCache.get_usable_length = get_usable_length
            print("[OK] Applied DynamicCache compatibility patch for Phi-3.5")
    except Exception as e:
        print(f"[WARNING] Could not apply DynamicCache patch: {e}")


# Hugging Face authentication
try:
    from huggingface_hub import login, whoami
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")


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
    """
    Load dataset from JSON/JSONL/CSV and tokenize with the specified format.
    
    Expected format: {"prompt": "...", "response": "..."} or CSV with these columns.
    Supports JSONL (one JSON object per line) format.
    """
    data_path = Path(data_path)
    
    # Try to load as JSONL first (one JSON object per line)
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
    # Try to load as JSON
    elif data_path.suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            # If it's a dict with a list key, extract it
            dataset = Dataset.from_list(list(data.values())[0] if data else [])
    else:
        # Assume CSV
        dataset = load_dataset("csv", data_files=str(data_path))["train"]
    
    def tokenize_function(examples):
        # Handle different column names
        prompt_col = "prompt" if "prompt" in examples else "question"
        response_col = "response" if "response" in examples else "solution"

        prompts = examples[prompt_col] if isinstance(examples[prompt_col], list) else [examples[prompt_col]]
        responses = examples[response_col] if isinstance(examples[response_col], list) else [examples[response_col]]

        texts = [format_prompt(p, r) for p, r in zip(prompts, responses)]

        # Tokenize (don't pad here - let the data collator handle it dynamically)
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding during training is more efficient
        )

        # For causal LM, labels are the same as input_ids
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
        # JSON format
        with open(test_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # If it's a dict, try to extract a list
        return list(data.values())[0] if data else []


def create_compute_metrics_fn(tokenizer, metrics_config):
    """Create a compute_metrics function for Trainer."""
    def compute_metrics(eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred

        # Convert logits to token IDs (take argmax)
        if len(predictions.shape) == 3:  # (batch_size, seq_len, vocab_size)
            predictions = predictions.argmax(axis=-1)

        # Replace -100 in labels (used for padding) with pad_token_id
        labels = [[label if label != -100 else tokenizer.pad_token_id for label in seq] for seq in labels]

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        results = {}
        
        # Exact match
        if "exact_match" in metrics_config:
            exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip())
            results["eval_exact_match"] = exact_matches / len(decoded_preds) if decoded_preds else 0.0
        
        # BLEU score
        if "bleu" in metrics_config:
            try:
                bleu_metric = load_metric("bleu")
                # BLEU expects list of lists for references
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
        
        # Symbolic equivalence (placeholder - would need SymPy/Z3 implementation)
        if "symbolic_equivalence" in metrics_config:
            # This would require actual symbolic verification logic
            results["eval_symbolic_equivalence"] = 0.0  # Placeholder
        
        return results
    
    return compute_metrics


def check_hf_authentication(model_name: str) -> bool:
    """
    Check if Hugging Face authentication appears to be set up.
    This is a best-effort check - the actual model loading will verify authentication.
    
    Returns:
        True if authentication appears to be set up, False otherwise
    """
    # Check if model requires authentication (gated models)
    gated_models = ["meta-llama", "llama"]
    
    if any(gated in model_name.lower() for gated in gated_models):
        authenticated = False
        
        # Try to verify authentication
        if HF_HUB_AVAILABLE:
            try:
                user_info = whoami()
                if user_info:
                    print(f"[OK] Hugging Face authenticated as: {user_info.get('name', 'user')}")
                    authenticated = True
            except Exception:
                pass
        
        # Check for token in environment
        if not authenticated and (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")):
            print("[OK] Hugging Face token found in environment")
            authenticated = True
        
        # Check for token file
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
            print("\nFor LLaMA models, you may also need to:")
            print("  - Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
            print("  - Accept the model's terms of use")
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
    """
    Train a single model with QLoRA.
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training model: {model_name}")
    print(f"{'='*60}\n")
    
    # Check authentication before attempting to load model (warn but don't fail)
    check_hf_authentication(model_name)
    
    # Initialize W&B if enabled
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
            print("Continuing without W&B logging...")
    else:
        print("W&B logging disabled")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer...")
    try:
        # Check device availability and configure appropriately
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.device_count()} device(s)")
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,  # Reduce CPU memory usage during loading
                "max_memory": {0: "14GB", "cpu": "30GB"},  # Set memory limits for Colab T4
            }
        elif torch.backends.mps.is_available():
            print("  MPS available, but forcing CPU for memory constraints")
            # MPS available but memory limited, use CPU mode
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float32,  # Use float32 on CPU
                # Note: No quantization on CPU
            }
        else:
            print("  CUDA/MPS not available, using CPU mode")
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float32,  # Use float32 on CPU
                # Note: No quantization on CPU
            }

        # Add attn_implementation for Phi-3.5 models to fix DynamicCache compatibility
        if "Phi" in model_name or "phi" in model_name:
            model_kwargs["attn_implementation"] = "eager"
            # Disable gradient checkpointing for Phi models to avoid cache issues
            config["training"]["gradient_checkpointing"] = False
            print("  Using eager attention for Phi model (compatibility fix)")
            print("  Disabled gradient checkpointing for Phi model (cache compatibility)")
            
            # Apply the DynamicCache patch before loading the model
            patch_phi_dynamic_cache()

        # Only use quantization config if CUDA is available
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            # Load without quantization on CPU
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
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare model for k-bit training (only if using quantization)
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing if specified (saves memory)
    # But not for Phi models due to cache compatibility issues
    if (config["training"].get("gradient_checkpointing", False) and
        not ("Phi" in model_name or "phi" in model_name)):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (memory optimization)")
    elif config["training"].get("gradient_checkpointing", False) and ("Phi" in model_name or "phi" in model_name):
        print("Gradient checkpointing disabled for Phi model (cache compatibility)")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config["training"]["lora_r"],
        lora_alpha=config["training"]["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config["training"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load validation data (for evaluation during training)
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
    
    # Training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=min(config["training"]["batch_size"], 2),  # Smaller eval batch
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        learning_rate=float(config["training"]["learning_rate"]),  # Ensure float type
        fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA available
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
        logging_steps=10,
        save_strategy="epoch",  # Match eval_strategy for load_best_model_at_end
        eval_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=["wandb"] if config["logging"]["use_wandb"] else None,
        run_name=f"{model_name.split('/')[-1]}_qlora",
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",  # Use appropriate optimizer
        max_grad_norm=0.3,  # Gradient clipping for stability
    )
    
    # Custom data collator that handles padding for causal LM
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )
    
    # Create compute_metrics function
    compute_metrics_fn = create_compute_metrics_fn(tokenizer, config["evaluation"]["metrics"])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    # Train
    print(f"Starting training...")
    trainer.train()
    
    # Evaluate on validation set (for final metrics)
    if validation_dataset:
        print(f"Evaluating on validation set...")
        eval_results = trainer.evaluate()
    else:
        print("Warning: No validation set available for evaluation")
        eval_results = {}
    
    # Skip test evaluation during training - test set will be evaluated later with human-in-the-loop
    # Test data may not have solutions, so we only use it for inference after training
    if test_data_path and validation_dataset:  # Only if we have separate validation
        print(f"\nSkipping test set evaluation during training (test data reserved for human-in-the-loop evaluation)")
        # test_data = load_test_data(test_data_path)
        # max_length = config["training"].get("max_length", 2048)
        # test_dataset = load_and_tokenize_dataset(
        #     test_data_path,
        #     tokenizer,
        #     max_length=max_length,
        # )
        # test_results = trainer.evaluate(eval_dataset=test_dataset)
        # # Add test metrics with 'test_' prefix
        # for key, value in test_results.items():
        #     eval_results[f"test_{key}"] = value
        # print(f"Test set results: {test_results}")
    
    # Save adapter
    print(f"Saving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Finish W&B run
    if config["logging"]["use_wandb"]:
        try:
            wandb.finish()
        except Exception:
            pass  # W&B wasn't initialized, so nothing to finish
    
    return eval_results


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train QLoRA models on algorithmic problem solving dataset"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific model(s) to train. Can specify by name (e.g., 'Qwen/Qwen2.5-7B-Instruct') "
             "or by index (e.g., '0' for first model in config, '1' for second). "
             "If not specified, trains all models in config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML file (default: configs/training_config.yaml)",
    )
    args = parser.parse_args()

    # Get project root (go up from training/train_dual_lora.py to project root)
    project_root = Path(__file__).parent.parent

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = project_root / "configs" / "training_config.yaml"

    config = load_config(str(config_path))

    # Setup paths
    data_dir = project_root / config["data_dir"]
    processed_dir = project_root / config["processed_dir"]
    
    # Find training data (look for train.jsonl, train.json, or train.csv)
    train_data_path = None
    for ext in [".jsonl", ".json", ".csv"]:
        # Check processed directory first (preferred)
        potential_path = processed_dir / f"train{ext}"
        if potential_path.exists():
            train_data_path = str(potential_path)
            break
        # Fallback to data_dir
        potential_path = data_dir / f"train{ext}"
        if potential_path.exists():
            train_data_path = str(potential_path)
            break
    
    if train_data_path is None:
        raise FileNotFoundError(
            f"No training data found. Checked {processed_dir} and {data_dir}. "
            f"Expected train.jsonl, train.json, or train.csv"
        )
    
    # Find validation data (optional, but recommended)
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
    
    # Find test data
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
    
    # Load and tokenize training dataset (will be reused for both models)
    print("Loading training dataset...")
    # We'll tokenize separately for each model since tokenizers differ

    # Filter models based on command-line arguments
    models_to_train = config["models"]

    if args.models:
        print(f"\nFiltering models based on command-line arguments: {args.models}")
        filtered_models = []

        for model_spec in args.models:
            # Try to match by index (e.g., "0", "1")
            if model_spec.isdigit():
                idx = int(model_spec)
                if idx < len(config["models"]):
                    filtered_models.append(config["models"][idx])
                    print(f"  Added model at index {idx}: {config['models'][idx]['name']}")
                else:
                    print(f"  Warning: Index {idx} out of range (only {len(config['models'])} models in config)")
            else:
                # Try to match by name (exact or partial)
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

    # Train each model
    all_results = {}

    for model_config in models_to_train:
        model_name = model_config["name"]
        output_dir = project_root / model_config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Preparing dataset for: {model_name}")
        print(f"{'='*60}\n")

        # Load tokenizer to tokenize dataset
        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"  Set pad_token to eos_token")

        # Tokenize dataset for this model
        max_length = config["training"].get("max_length", 2048)
        print(f"Tokenizing training dataset (max_length={max_length})...")
        train_dataset = load_and_tokenize_dataset(train_data_path, tokenizer, max_length=max_length)
        print(f"Training dataset ready: {len(train_dataset)} examples")
        
        # Train model
        results = train_model(
            model_name=model_name,
            output_dir=str(output_dir),
            config=config,
            train_dataset=train_dataset,
            validation_data_path=validation_data_path,
            test_data_path=test_data_path,
        )
        
        all_results[model_name] = results
    
    # Print summary
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