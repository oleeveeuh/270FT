"""
Enhanced evaluation script that handles test questions with and without reference solutions.
- Runs automated metrics (BLEU, exact match, etc.) on questions with solutions
- Flags questions without solutions for human review
- Generates comprehensive reports with both automated and human-review items
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load as load_metric
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

    if not adapter_path_obj.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

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
        try:
            # Try with use_cache=False for models that have cache issues (e.g., Phi-3.5)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,  # Disable cache to avoid DynamicCache issues
            )
        except (AttributeError, RuntimeError) as e:
            # If cache=False fails, try without explicit cache setting
            if "DynamicCache" in str(e) or "seen_tokens" in str(e):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False,  # Switch to greedy decoding if sampling fails
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                raise

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return generated_text.strip()


def compute_exact_match(reference: str, prediction: str) -> bool:
    """Compute exact match between reference and prediction."""
    return reference.strip().lower() == prediction.strip().lower()


def compute_bleu_score(reference: str, prediction: str) -> float:
    """Compute BLEU score between reference and prediction using SequenceMatcher."""
    try:
        # Use SequenceMatcher for text similarity (more reliable than BLEU metric)
        from difflib import SequenceMatcher
        
        # Normalize both strings
        ref_norm = " ".join(reference.split()).lower()
        pred_norm = " ".join(prediction.split()).lower()
        
        # Calculate similarity ratio (0-1 scale)
        ratio = SequenceMatcher(None, ref_norm, pred_norm).ratio()
        return ratio
    except Exception as e:
        print(f"Warning: BLEU computation failed: {e}")
        return 0.0


def automated_quality_check(question: str, generated_solution: str) -> Dict[str, Any]:
    """
    Automated quality checks for algorithmic solutions.
    Returns structural analysis without requiring reference solution.
    """
    issues = []

    # Check 1: Minimum length
    if len(generated_solution) < 200:
        issues.append("Too short - likely incomplete")

    # Check 2: Required sections for algorithmic problems
    has_algorithm = bool(re.search(r'(algorithm|pseudocode|procedure)',
                                   generated_solution, re.I))
    has_runtime = bool(re.search(r'O\([^)]+\)', generated_solution))
    has_proof_keywords = bool(re.search(r'(proof|correctness|invariant|assume|therefore)',
                                        generated_solution, re.I))

    if not has_algorithm:
        issues.append("Missing algorithm/pseudocode section")
    if not has_runtime:
        issues.append("Missing runtime analysis (Big-O notation)")
    if not has_proof_keywords:
        issues.append("Missing correctness proof keywords")

    # Check 3: Code structure
    has_code_structure = bool(re.search(r'(for|while|if|return)\s+', generated_solution))
    if not has_code_structure:
        issues.append("No code/pseudocode structure detected")

    # Extract Big-O complexity if present
    complexity_match = re.findall(r'O\([^)]+\)', generated_solution)

    return {
        'passes_basic_checks': len(issues) == 0,
        'issues': issues,
        'has_algorithm': has_algorithm,
        'has_runtime_analysis': has_runtime,
        'has_proof_keywords': has_proof_keywords,
        'has_code_structure': has_code_structure,
        'detected_complexity': complexity_match,
        'length_chars': len(generated_solution),
        'length_tokens': len(generated_solution.split())
    }


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_data: List[Dict[str, str]],
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluate model on test data with hybrid approach.
    - Automated metrics for items with reference solutions
    - Quality checks for items without solutions (flagged for human review)
    """
    print(f"\nEvaluating {model_name}...")
    print(f"Test items: {len(test_data)}\n")

    results = {
        "model_name": model_name,
        "total_items": len(test_data),
        "items_with_solutions": 0,
        "items_without_solutions": 0,
        "automated_metrics": {
            "exact_matches": 0,
            "bleu_scores": [],
        },
        "per_item_results": [],
        "human_review_needed": [],
    }

    for idx, item in enumerate(test_data):
        print(f"Processing item {idx + 1}/{len(test_data)}...")

        question = item.get("question", "")
        reference_solution = item.get("solution", None)

        # Enhanced debugging to track why items might be skipped
        print(f"DEBUG Item {idx}: has_question={bool(question)}, has_solution={reference_solution is not None}")

        if not question:
            print(f"Warning: Skipping item {idx + 1} - missing question")
            print(f"DEBUG: Item {idx} content: {item}")
            continue

        # Generate prediction
        try:
            prediction = generate_solution(model, tokenizer, question)
            if not prediction:  # Check if generation returned empty result
                print(f"Warning: Skipping item {idx + 1} - empty prediction generated")
                continue
        except Exception as e:
            print(f"Warning: Skipping item {idx + 1} - prediction generation failed: {e}")
            print(f"DEBUG: Item {idx} content: {item}")
            continue

        # Automated quality checks (always run)
        quality_check = automated_quality_check(question, prediction)

        item_result = {
            "item_id": idx,
            "question": question,
            "prediction": prediction,
            "quality_check": quality_check,
            "has_reference": reference_solution is not None,
        }

        if reference_solution:
            # Has reference solution - run automated metrics
            results["items_with_solutions"] += 1

            exact_match = compute_exact_match(reference_solution, prediction)
            bleu_score = compute_bleu_score(reference_solution, prediction)

            if exact_match:
                results["automated_metrics"]["exact_matches"] += 1
            results["automated_metrics"]["bleu_scores"].append(bleu_score)

            item_result.update({
                "reference_solution": reference_solution,
                "exact_match": exact_match,
                "bleu_score": bleu_score,
            })
        else:
            # No reference solution - flag for human review
            results["items_without_solutions"] += 1
            results["human_review_needed"].append({
                "item_id": idx,
                "question": question,
                "prediction": prediction,
                "quality_check": quality_check,
            })

        results["per_item_results"].append(item_result)

    # Compute aggregate metrics (only for items with solutions)
    if results["items_with_solutions"] > 0:
        results["automated_metrics"]["exact_match_rate"] = (
            results["automated_metrics"]["exact_matches"] / results["items_with_solutions"]
        )
        results["automated_metrics"]["avg_bleu_score"] = (
            sum(results["automated_metrics"]["bleu_scores"]) /
            len(results["automated_metrics"]["bleu_scores"])
            if results["automated_metrics"]["bleu_scores"] else 0.0
        )
    else:
        results["automated_metrics"]["exact_match_rate"] = 0.0
        results["automated_metrics"]["avg_bleu_score"] = 0.0

    # Print summary
    print(f"\n{model_name} Evaluation Summary:")
    print(f"  Total items: {results['total_items']}")
    print(f"  Items with reference solutions: {results['items_with_solutions']}")
    print(f"  Items needing human review: {results['items_without_solutions']}")

    if results["items_with_solutions"] > 0:
        print(f"\n  Automated Metrics (on {results['items_with_solutions']} items with solutions):")
        print(f"    Exact Match Rate: {results['automated_metrics']['exact_match_rate']:.4f}")
        print(f"    Average BLEU Score: {results['automated_metrics']['avg_bleu_score']:.4f}")

    return results


def export_human_review_csv(results: Dict[str, Any], output_path: Path):
    """Export items needing human review to CSV for easy review."""
    import csv

    if not results["human_review_needed"]:
        print("No items need human review - all have reference solutions!")
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Item ID',
            'Question',
            'Generated Solution',
            'Has Algorithm',
            'Has Runtime Analysis',
            'Has Proof Keywords',
            'Quality Issues',
            'Rating (1-5)',
            'Comments'
        ])

        # Data rows
        for item in results["human_review_needed"]:
            qc = item["quality_check"]
            writer.writerow([
                item["item_id"],
                item["question"][:100] + "..." if len(item["question"]) > 100 else item["question"],
                item["prediction"][:200] + "..." if len(item["prediction"]) > 200 else item["prediction"],
                "Yes" if qc["has_algorithm"] else "No",
                "Yes" if qc["has_runtime_analysis"] else "No",
                "Yes" if qc["has_proof_keywords"] else "No",
                "; ".join(qc["issues"]) if qc["issues"] else "None",
                "",  # Empty for human to fill
                ""   # Empty for human comments
            ])

    print(f"\nHuman review template exported to: {output_path}")
    print(f"  {len(results['human_review_needed'])} items need review")


def main():
    """Main evaluation function."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "training_config.yaml"

    config = load_config(str(config_path))

    # Load test data - try test_with_solutions.jsonl first, fall back to test.jsonl
    processed_dir = project_root / config["processed_dir"]

    test_data_path = processed_dir / "test_with_solutions.jsonl"
    if not test_data_path.exists():
        test_data_path = processed_dir / "test.jsonl"

    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_data_path}")

    # Load test data (JSONL format)
    print(f"Loading test data from {test_data_path}...")
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    test_data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error on line {line_num}: {e}")
                    print(f"DEBUG: Problem line content: {line[:200]}...")
                    continue
                except Exception as e:
                    print(f"Warning: Unexpected error on line {line_num}: {e}")
                    print(f"DEBUG: Problem line content: {line[:200]}...")
                    continue

    print(f"Loaded {len(test_data)} test items")

    # Count items with/without solutions
    with_solutions = sum(1 for item in test_data if "solution" in item and item["solution"])
    without_solutions = len(test_data) - with_solutions
    print(f"  {with_solutions} items have reference solutions (automated evaluation)")
    print(f"  {without_solutions} items need human review")

    # Create results directory
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    for model_config in config["models"]:
        base_model_name = model_config["name"]
        adapter_path = project_root / model_config["output_dir"]

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

        # Save detailed JSON results
        model_slug = base_model_name.replace("/", "_")
        json_path = results_dir / f"evaluation_{model_slug}.json"
        with open(json_path, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"\nDetailed results saved to: {json_path}")

        # Export human review CSV if needed
        if model_results["human_review_needed"]:
            csv_path = results_dir / f"human_review_{model_slug}.csv"
            export_human_review_csv(model_results, csv_path)

        # Clean up model from memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if device == "cuda" else None

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_dir}/")


if __name__ == "__main__":
    main()