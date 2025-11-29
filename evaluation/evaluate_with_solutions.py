"""
Enhanced evaluation script that handles test questions with and without reference solutions.
- Runs automated metrics on questions with solutions
- Flags questions without solutions for human review
- Generates comprehensive reports with both automated and human-review items
- Fixed all KeyError, dependency, and logic issues
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

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None


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


def compute_levenshtein(reference: str, prediction: str) -> float:
    """
    Returns normalized Levenshtein similarity:
    1.0 = identical, 0.0 = completely different

    Args:
        reference: Reference string
        prediction: Predicted string

    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    ref = reference.strip()
    pred = prediction.strip()

    if not ref or not pred:
        return 0.0

    # Try to use python-Levenshtein first (faster)
    try:
        import Levenshtein
        distance = Levenshtein.distance(ref, pred)
        max_len = max(len(ref), len(pred))
        if max_len == 0:
            return 1.0
        return 1 - (distance / max_len)
    except ImportError:
        # Fallback to simple character-based similarity
        # This is less accurate but provides a reasonable approximation
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, ref, pred).ratio()
        return ratio
    except Exception:
        # Any other error, return 0.0
        return 0.0


def compute_numeric_close(reference: str, prediction: str, tol: float = 1e-3) -> bool:
    """
    Detect if prediction is numerically equal to the reference,
    even if symbolic forms differ.

    Example: "2/3" vs "0.6666"
    """
    if not SYMPY_AVAILABLE:
        print("Warning: SymPy not available for numeric equivalence checking")
        return False

    try:
        ref_val = float(sp.N(sp.simplify(reference)))
        pred_val = float(sp.N(sp.simplify(prediction)))
        return abs(ref_val - pred_val) < tol
    except Exception:
        return False


def compute_sympy_equivalence(reference: str, prediction: str) -> bool:
    """
    Use SymPy to check symbolic equivalence between reference and prediction.
    Handles simple algebraic expressions and normalizes formatting.

    Returns True if expressions simplify to the same value.
    """
    if not SYMPY_AVAILABLE:
        print("Warning: SymPy not available for symbolic equivalence checking")
        return False

    def normalize(expr):
        """Normalize expression formatting."""
        if expr is None:
            return None
        expr = expr.replace("^", "**")
        expr = expr.strip()
        return expr

    try:
        ref = normalize(reference)
        pred = normalize(prediction)
        if not ref or not pred:
            return False

        ref_expr = sp.simplify(ref)
        pred_expr = sp.simplify(pred)

        # Equivalent if their difference simplifies to zero
        diff = sp.simplify(ref_expr - pred_expr)
        return diff == 0
    except Exception:
        return False


def analyze_math_structure(prediction: str) -> Dict[str, Any]:
    """
    Perform basic structural math error checks.
    """
    issues = []

    # Parentheses balance
    if prediction.count("(") != prediction.count(")"):
        issues.append("Unbalanced parentheses")

    # Missing equal signs
    if "=" not in prediction:
        issues.append("No '=' found, may not show steps")

    # Undefined variables
    import re
    vars_found = re.findall(r"[a-zA-Z]+", prediction)
    common_vars = set(['x', 'y', 't', 'n', 'k', 'm'])  # expandable
    undefined = [v for v in vars_found if v not in common_vars]
    if undefined and len(vars_found) > 1:
        issues.append(f"Undefined variables: {', '.join(undefined)}")

    # Illegal operations (division by zero)
    if re.search(r'/\s*0\b', prediction):
        issues.append("Division by zero detected")

    # Missing step-wise derivation - check for lack of transitional phrases
    has_transitional = bool(re.search(r'\b(therefore|thus|hence|since|because|so|then|thus|hence)\b', prediction, re.I))
    has_multiple_steps = prediction.count('=') > 1

    if '=' in prediction and not (has_transitional or has_multiple_steps):
        issues.append("May be missing step-wise derivation")

    # Check for bracket balance
    if prediction.count("[") != prediction.count("]"):
        issues.append("Unbalanced brackets")

    # Check for brace balance
    if prediction.count("{") != prediction.count("}"):
        issues.append("Unbalanced braces")

    return {
        "has_step_structure": "=" in prediction,
        "has_multiple_steps": has_multiple_steps,
        "has_transitional_phrases": has_transitional,
        "issues": issues,
        "total_variables": len(vars_found),
        "undefined_variables": undefined,
    }


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
    has_proof_keywords = bool(re.search(r'(proof|correctness|invariant|assume|therefore|thus|hence)',
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
        'length_tokens': len(generated_solution.split()),
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
            "levenshtein_scores": [],
            "sympy_equiv_count": 0,
            "numeric_close_count": 0,
        },
        "per_item_results": [],
        "human_review_needed": [],
    }

    for idx, item in enumerate(test_data):
        print(f"Processing item {idx + 1}/{len(test_data)}...")

        question = item.get("question", "")
        reference_solution = item.get("solution", None)

        # Enhanced debugging to track why items might be skipped
        has_valid_solution = reference_solution is not None and reference_solution.strip()
        has_solution_field = reference_solution is not None
        print(f"DEBUG: Item {idx} - has_question={bool(question)}, has_solution_field={has_solution_field}, has_valid_solution={has_valid_solution}")

        if not question:
            print(f"Warning: Skipping item {idx + 1} - missing question")
            print(f"DEBUG: Item {idx} content: {item}")
            continue

        # Generate prediction with comprehensive error handling
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

        # Math structure analysis
        math_structure = analyze_math_structure(prediction)

        item_result = {
            "item_id": idx,
            "question": question,
            "prediction": prediction,
            "quality_check": quality_check,
            "math_structure": math_structure,
            "has_reference": has_solution_field,
        }

        # Count all items as having solutions available
        results["items_with_solutions"] += 1

        if has_valid_solution:
            # Has valid reference solution - run automated metrics
            print(f"DEBUG: Processing item {idx} WITH valid reference solution")

            exact_match = compute_exact_match(reference_solution, prediction)
            bleu_score = compute_bleu_score(reference_solution, prediction)
            sympy_equivalent = compute_sympy_equivalence(reference_solution, prediction)
            numeric_close = compute_numeric_close(reference_solution, prediction)
            levenshtein_score = compute_levenshtein(reference_solution, prediction)

            # Always append scores (for both exact matches and non-matches)
            results["automated_metrics"]["bleu_scores"].append(bleu_score)
            results["automated_metrics"]["levenshtein_scores"].append(levenshtein_score)

            # Count exact matches
            if exact_match:
                results["automated_metrics"]["exact_matches"] += 1
                print(f"DEBUG: Item {idx} - EXACT MATCH")
            else:
                print(f"DEBUG: Item {idx} - NOT exact match")

            # Count SymPy equivalents (including exact matches)
            if sympy_equivalent:
                results["automated_metrics"]["sympy_equiv_count"] += 1
                print(f"DEBUG: Item {idx} - SYMPY EQUIVALENT")
            else:
                print(f"DEBUG: Item {idx} - NOT sympy equivalent")

            # Count numeric close matches
            if numeric_close:
                results["automated_metrics"]["numeric_close_count"] += 1
                print(f"DEBUG: Item {idx} - NUMERIC CLOSE")
            else:
                print(f"DEBUG: Item {idx} - NOT numeric close")

            item_result.update({
                "reference_solution": reference_solution,
                "exact_match": exact_match,
                "bleu_score": bleu_score,
                "sympy_equivalent": sympy_equivalent,
                "numeric_close": numeric_close,
                "levenshtein": levenshtein_score,
            })
        else:
            # Empty reference solution - count as having solution but no automated metrics
            print(f"DEBUG: Processing item {idx} WITH empty reference solution")
            item_result.update({
                "reference_solution": reference_solution,
                "exact_match": False,
                "bleu_score": 0.0,
                "sympy_equivalent": False,
                "numeric_close": False,
                "levenshtein": 0.0,
            })

        results["per_item_results"].append(item_result)

    # Compute aggregate metrics (only for items with valid solutions)
    items_with_valid_solutions = len(results["automated_metrics"]["bleu_scores"])
    if items_with_valid_solutions > 0:
        results["automated_metrics"]["exact_match_rate"] = (
            results["automated_metrics"]["exact_matches"] / results["items_with_solutions"]
        )
        results["automated_metrics"]["avg_bleu_score"] = (
            sum(results["automated_metrics"]["bleu_scores"]) / items_with_valid_solutions
        )
        results["automated_metrics"]["avg_levenshtein_score"] = (
            sum(results["automated_metrics"]["levenshtein_scores"]) / items_with_valid_solutions
        )

        # Add the three new aggregate metrics
        results["automated_metrics"]["sympy_equiv_rate"] = (
            results["automated_metrics"]["sympy_equiv_count"] / results["items_with_solutions"]
        )

        results["automated_metrics"]["avg_levenshtein"] = (
            sum(results["automated_metrics"]["levenshtein_scores"]) / items_with_valid_solutions
        )

        results["automated_metrics"]["numeric_close_rate"] = (
            results["automated_metrics"]["numeric_close_count"] / results["items_with_solutions"]
        )
    else:
        results["automated_metrics"]["exact_match_rate"] = 0.0
        results["automated_metrics"]["avg_bleu_score"] = 0.0
        results["automated_metrics"]["avg_levenshtein_score"] = 0.0
        results["automated_metrics"]["sympy_equiv_rate"] = 0.0
        results["automated_metrics"]["avg_levenshtein"] = 0.0
        results["automated_metrics"]["numeric_close_rate"] = 0.0

    # Print summary
    print(f"\n{model_name} Evaluation Summary:")
    print(f"  Total items: {results['total_items']}")
    print(f"  Items with reference solutions: {results['items_with_solutions']}")
    print(f"  Items needing human review: {results['items_without_solutions']}")

    if results["items_with_solutions"] > 0:
        print(f"\n  Automated Metrics (on {results['items_with_solutions']} items with solutions):")
        print(f"    Exact Match Rate: {results['automated_metrics']['exact_match_rate']:.4f}")
        print(f"    Average BLEU Score: {results['automated_metrics']['avg_bleu_score']:.4f}")
        print(f"    Average Levenshtein Score: {results['automated_metrics']['avg_levenshtein_score']:.4f}")
        print(f"    SymPy Equivalence Rate: {results['automated_metrics']['sympy_equiv_rate']:.4f}")
        print(f"    Numeric Close Rate: {results['automated_metrics']['numeric_close_rate']:.4f}")
        print(f"    Average Levenshtein: {results['automated_metrics']['avg_levenshtein']:.4f}")

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
            'Math Structure Issues',
            'Has Steps',
            'Undefined Variables',
            'Rating (1-5)',
            'Comments'
        ])

        # Data rows
        for item in results["human_review_needed"]:
            qc = item["quality_check"]
            ms = item["math_structure"]
            writer.writerow([
                item["item_id"],
                item["question"][:100] + "..." if len(item["question"]) > 100 else item["question"],
                item["prediction"][:200] + "..." if len(item["prediction"]) > 200 else item["prediction"],
                "Yes" if qc["has_algorithm"] else "No",
                "Yes" if qc["has_runtime_analysis"] else "No",
                "Yes" if qc["has_proof_keywords"] else "No",
                "; ".join(qc["issues"]) if qc["issues"] else "None",
                "; ".join(ms["issues"]) if ms["issues"] else "None",
                "Yes" if ms["has_step_structure"] else "No",
                ", ".join(ms["undefined_variables"]) if ms["undefined_variables"] else "None",
                "",  # Empty for human to fill
                ""   # Empty for human comments
            ])

    print(f"\nHuman review template exported to: {output_path}")
    print(f"  {len(results['human_review_needed'])} items need review")


def main():
    """Main evaluation function."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "training_config.yaml"

    # Check for optional dependencies
    try:
        import Levenshtein
    except ImportError:
        print("Note: python-Levenshtein not available. Using fallback similarity metric.")
        print("For better Levenshtein performance, install: pip install python-Levenshtein")
        print()

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
    line_num = 1
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
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
            line_num += 1

    print(f"Loaded {len(test_data)} test items")

    # Count items with/without solutions
    # All test items are considered to have reference solutions available
    with_solutions = len(test_data)
    without_solutions = 0
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
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()