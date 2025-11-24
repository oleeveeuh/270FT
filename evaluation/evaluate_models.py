"""
Evaluation script for fine-tuned models with symbolic verification.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load as load_metric
from sympy import simplify, Eq, symbols, sympify, parse_expr
import re


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def format_prompt(prompt: str) -> str:
    """Format prompt in the training template format."""
    return f"### Question:\n{prompt}\n\n### Solution:\n"


def load_model_with_adapter(
    base_model_name: str,
    adapter_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and apply LoRA adapter."""
    adapter_path_obj = Path(adapter_path)
    
    # Verify adapter path exists
    if not adapter_path_obj.exists():
        raise FileNotFoundError(
            f"Adapter path does not exist: {adapter_path}\n"
            f"Please ensure the model has been trained and saved to this location."
        )
    
    # Check for required adapter files
    adapter_config = adapter_path_obj / "adapter_config.json"
    adapter_weights = adapter_path_obj / "adapter_model.safetensors"
    if not adapter_weights.exists():
        adapter_weights = adapter_path_obj / "adapter_model.bin"
    
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"Adapter config not found at {adapter_config}. "
            f"Model may not have been saved correctly during training."
        )
    
    if not adapter_weights.exists():
        raise FileNotFoundError(
            f"Adapter weights not found at {adapter_weights}. "
            f"Model may not have been saved correctly during training."
        )
    
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    print(f"  - Config: {adapter_config}")
    print(f"  - Weights: {adapter_weights}")
    model = PeftModel.from_pretrained(model, str(adapter_path_obj))
    model.eval()
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"[OK] Model and adapter loaded successfully")
    return model, tokenizer


def generate_solution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate solution for a given prompt."""
    formatted_prompt = format_prompt(prompt)
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return generated_text.strip()


def extract_expression(text: str) -> str:
    """
    Extract mathematical expression from text.
    Tries to find expressions like 'x^2 + 3*x + 2' or similar.
    """
    # Remove common prefixes/suffixes
    text = text.strip()
    
    # Try to find expressions after "=" or "Solution:" etc.
    patterns = [
        r'(?:solution|answer|result)[:\s]*([^\.\n]+)',
        r'=\s*([^\.\n]+)',
        r'([a-zA-Z0-9\s\+\-\*/\^\(\)]+)',  # General math expression
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            # Clean up the expression
            expr = re.sub(r'\s+', '', expr)  # Remove whitespace
            return expr
    
    # If no pattern matches, try to parse the whole text
    return text.strip()


def check_symbolic_equivalence(
    reference_expr: str,
    predicted_expr: str,
) -> bool:
    """
    Check if two expressions are symbolically equivalent using SymPy.
    
    Args:
        reference_expr: Reference expression string
        predicted_expr: Predicted expression string
    
    Returns:
        True if expressions are equivalent, False otherwise
    """
    try:
        # Extract expressions from text if needed
        ref_clean = extract_expression(reference_expr)
        pred_clean = extract_expression(predicted_expr)
        
        # Define common symbols
        x, y, z = symbols('x y z')
        
        # Try to parse expressions
        try:
            ref_sym = parse_expr(ref_clean, transformations='all')
        except:
            try:
                ref_sym = sympify(ref_clean)
            except:
                return False
        
        try:
            pred_sym = parse_expr(pred_clean, transformations='all')
        except:
            try:
                pred_sym = sympify(pred_clean)
            except:
                return False
        
        # Check equivalence by simplifying the difference
        diff = simplify(ref_sym - pred_sym)
        
        # If difference simplifies to 0, they're equivalent
        if diff == 0:
            return True
        
        # Also check if they're equal as expressions
        if ref_sym.equals(pred_sym):
            return True
        
        # Try checking with Eq
        eq = Eq(ref_sym, pred_sym)
        if eq.simplify() == True:
            return True
        
        return False
    
    except Exception as e:
        # If any error occurs, return False
        print(f"Warning: Symbolic equivalence check failed: {e}")
        return False


def compute_exact_match(reference: str, prediction: str) -> bool:
    """Compute exact match between reference and prediction."""
    return reference.strip().lower() == prediction.strip().lower()


def compute_bleu_score(reference: str, prediction: str) -> float:
    """Compute BLEU score between reference and prediction."""
    try:
        bleu_metric = load_metric("bleu")
        references = [[reference.split()]]
        predictions = [prediction.split()]
        result = bleu_metric.compute(
            predictions=predictions,
            references=references,
        )
        return result.get("bleu", 0.0)
    except Exception as e:
        print(f"Warning: BLEU computation failed: {e}")
        return 0.0


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_data: List[Dict[str, str]],
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluate a single model on test data.
    
    Returns:
        Dictionary with aggregated metrics and per-item results
    """
    print(f"\nEvaluating {model_name}...")
    print(f"Test items: {len(test_data)}\n")
    
    results = {
        "model_name": model_name,
        "total_items": len(test_data),
        "exact_matches": 0,
        "symbolic_equivalences": 0,
        "bleu_scores": [],
        "per_item_results": [],
    }
    
    for idx, item in enumerate(test_data):
        if idx % 10 == 0:
            print(f"Processing item {idx + 1}/{len(test_data)}...")
        
        # Get prompt and reference
        prompt = item.get("prompt", item.get("question", ""))
        reference = item.get("response", item.get("solution", item.get("answer", "")))
        
        if not prompt or not reference:
            print(f"Warning: Skipping item {idx + 1} - missing prompt or reference")
            continue
        
        # Generate prediction
        prediction = generate_solution(model, tokenizer, prompt)
        
        # Compute metrics
        exact_match = compute_exact_match(reference, prediction)
        bleu_score = compute_bleu_score(reference, prediction)
        symbolic_eq = check_symbolic_equivalence(reference, prediction)
        
        # Update aggregates
        if exact_match:
            results["exact_matches"] += 1
        if symbolic_eq:
            results["symbolic_equivalences"] += 1
        results["bleu_scores"].append(bleu_score)
        
        # Store per-item result
        results["per_item_results"].append({
            "item_id": idx,
            "prompt": prompt,
            "reference": reference,
            "prediction": prediction,
            "exact_match": exact_match,
            "bleu_score": bleu_score,
            "symbolic_equivalence": symbolic_eq,
        })
    
    # Compute aggregate metrics
    results["exact_match_rate"] = results["exact_matches"] / results["total_items"]
    results["symbolic_equivalence_rate"] = results["symbolic_equivalences"] / results["total_items"]
    results["avg_bleu_score"] = sum(results["bleu_scores"]) / len(results["bleu_scores"]) if results["bleu_scores"] else 0.0
    
    print(f"\n{model_name} Results:")
    print(f"  Exact Match Rate: {results['exact_match_rate']:.4f}")
    print(f"  Symbolic Equivalence Rate: {results['symbolic_equivalence_rate']:.4f}")
    print(f"  Average BLEU Score: {results['avg_bleu_score']:.4f}")
    
    return results


def main():
    """Main evaluation function."""
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "270FT" / "configs" / "training_config.yaml"
    
    # Load config
    config = load_config(str(config_path))
    
    # Setup paths
    processed_dir = project_root / "270FT" / config["processed_dir"]
    
    # Try to find test data (JSONL or JSON)
    test_data_path = None
    for ext in [".jsonl", ".json"]:
        potential_path = processed_dir / f"test{ext}"
        if potential_path.exists():
            test_data_path = potential_path
            break
    
    if test_data_path is None:
        raise FileNotFoundError(
            f"Test data not found. Checked {processed_dir / 'test.jsonl'} and {processed_dir / 'test.json'}"
        )
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    
    if test_data_path.suffix == ".jsonl":
        # Load JSONL format (one JSON object per line)
        test_data = []
        with open(test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        test_data.append(item)
                    except json.JSONDecodeError:
                        continue
    else:
        # Load JSON format
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        if isinstance(test_data, dict):
            # If it's a dict, try to extract a list
            test_data = list(test_data.values())[0] if test_data else []
    
    if not isinstance(test_data, list):
        raise ValueError("Test data must be a list of items")
    
    print(f"Loaded {len(test_data)} test items\n")
    
    # Create results directory
    results_dir = project_root / "270FT" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    all_results = {}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    for model_config in config["models"]:
        base_model_name = model_config["name"]
        adapter_path = project_root / "270FT" / model_config["output_dir"]
        
        if not adapter_path.exists():
            print(f"Warning: Adapter path {adapter_path} does not exist. Skipping {base_model_name}")
            continue
        
        # Load model with adapter
        model, tokenizer = load_model_with_adapter(
            base_model_name,
            str(adapter_path),
            device=device,
        )
        
        # Evaluate
        model_results = evaluate_model(
            model,
            tokenizer,
            test_data,
            base_model_name,
        )
        
        all_results[base_model_name] = model_results
        
        # Clean up model from memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if device == "cuda" else None
    
    # Create final report
    report = {
        "evaluation_summary": {
            "total_test_items": len(test_data),
            "models_evaluated": list(all_results.keys()),
        },
        "model_results": all_results,
    }
    
    # Save report
    report_path = results_dir / "metrics_report.json"
    print(f"\nSaving evaluation report to {report_path}...")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}\n")
    
    for model_name, results in all_results.items():
        print(f"Model: {model_name}")
        print(f"  Exact Match Rate: {results['exact_match_rate']:.4f}")
        print(f"  Symbolic Equivalence Rate: {results['symbolic_equivalence_rate']:.4f}")
        print(f"  Average BLEU Score: {results['avg_bleu_score']:.4f}")
        print()
    
    print(f"{'='*60}")
    print(f"Full report saved to: {report_path}")


if __name__ == "__main__":
    main()

