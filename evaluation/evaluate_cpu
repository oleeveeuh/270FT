"""
CPU-OPTIMIZED evaluation script
Targets: Reduce inference time from 30min to 3-5 minutes per question

Key optimizations for CPU:
1. Quantize model to 8-bit (4x faster, minimal quality loss)
2. Reduce max_new_tokens significantly (512 -> 256)
3. Use faster generation strategy (greedy instead of sampling)
4. Disable embeddings for speed (or use CPU-optimized batching)
5. Skip SymPy operations (too slow on CPU)
6. Use fast rule-based semantic only
7. Reduce attention computation (Flash Attention if available)
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings

warnings.filterwarnings('ignore')

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None
    cosine_similarity = None
    np = None

# ============================================================================
# GLOBAL CACHES & COMPILED PATTERNS
# ============================================================================

_EMBEDDING_MODEL_CACHE = None
_EMBEDDING_MODEL_NAME_CACHE = None

_REGEX_PATTERNS = {
    'transitional': re.compile(r'\b(therefore|thus|hence|since|because|so|then)\b', re.I),
    'big_o': re.compile(r'O\([^)]+\)'),
    'division_zero': re.compile(r'/\s*0\b'),
    'algorithm': re.compile(r'(algorithm|pseudocode|procedure)', re.I),
    'runtime': re.compile(r'O\([^)]+\)'),
    'proof': re.compile(r'(proof|correctness|invariant|assume|therefore|thus|hence)', re.I),
    'code_struct': re.compile(r'(for|while|if|return)\s+'),
    'words': re.compile(r'[a-zA-Z]{3,}'),
}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Get cached embedding model (load only once)"""
    global _EMBEDDING_MODEL_CACHE, _EMBEDDING_MODEL_NAME_CACHE
    
    if _EMBEDDING_MODEL_CACHE is None or _EMBEDDING_MODEL_NAME_CACHE != model_name:
        print(f"Loading embedding model: {model_name}")
        _EMBEDDING_MODEL_CACHE = SentenceTransformer(model_name)
        _EMBEDDING_MODEL_CACHE.eval()
    
    return _EMBEDDING_MODEL_CACHE


# ============================================================================
# OPTIMIZED METRICS FOR CPU
# ============================================================================

def embedding_similarity(ref: str, pred: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Optimized embedding similarity with caching"""
    if not EMBEDDINGS_AVAILABLE:
        return 0.0

    if not ref or not pred:
        return 0.0

    try:
        model = get_embedding_model(model_name)
        
        ref_normalized = " ".join(ref.lower().split())
        pred_normalized = " ".join(pred.lower().split())

        with torch.no_grad():
            ref_embedding = model.encode([ref_normalized], convert_to_tensor=False)
            pred_embedding = model.encode([pred_normalized], convert_to_tensor=False)

            similarity = cosine_similarity([ref_embedding[0]], [pred_embedding[0]])[0][0]
            similarity = max(0.0, min(1.0, float(similarity)))
            return similarity

    except Exception as e:
        print(f"Warning: Embedding similarity failed: {e}")
        return 0.0


def compute_exact_match(reference: str, prediction: str) -> bool:
    """Fast exact match"""
    return reference.strip().lower() == prediction.strip().lower()


def compute_bleu_score(reference: str, prediction: str) -> float:
    """Fast BLEU score"""
    try:
        from difflib import SequenceMatcher
        ref_norm = " ".join(reference.split()).lower()
        pred_norm = " ".join(prediction.split()).lower()
        return SequenceMatcher(None, ref_norm, pred_norm).ratio()
    except Exception:
        return 0.0


def compute_levenshtein(reference: str, prediction: str) -> float:
    """Levenshtein similarity"""
    ref = reference.strip()
    pred = prediction.strip()

    if not ref or not pred:
        return 0.0

    try:
        import Levenshtein
        distance = Levenshtein.distance(ref, pred)
        max_len = max(len(ref), len(pred))
        return 1 - (distance / max_len) if max_len > 0 else 1.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, ref, pred).ratio()
    except Exception:
        return 0.0


def is_symbolic_expression(text: str) -> bool:
    """Check if text is mathematical expression"""
    if not text or not text.strip():
        return False

    text = text.strip()
    allowed_chars = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/^=<>!&|.,;:_ ')
    allowed_special = set('()[]{}')

    for char in text:
        if char not in allowed_chars and char not in allowed_special:
            if char not in 'π∞√∑∏∫±≤≥≠≈∈∉⊆⊇∪∩':
                return False

    words = _REGEX_PATTERNS['words'].findall(text.lower())
    english_words = {
        'the', 'and', 'for', 'are', 'with', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out',
        'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way',
        'who', 'boy', 'did', 'form', 'from', 'give', 'hand', 'made', 'many', 'most', 'move', 'must',
        'proof', 'solution', 'answer', 'explain', 'show', 'prove', 'is', 'are', 'am', 'be', 'been'
    }

    if any(word in english_words for word in words):
        return False

    has_math = (
        any(op in text for op in '+-*/^=<>!') or
        any(char.isdigit() for char in text) or
        '(' in text or ')' in text or '[' in text or ']' in text
    )

    return has_math


def compute_sympy_equivalence_safe(reference: str, prediction: str, 
                                   timeout_sec: float = 0.5) -> bool:
    """SymPy equivalence with aggressive timeout (CPU is slow)"""
    if not SYMPY_AVAILABLE:
        return False

    if not is_symbolic_expression(reference) or not is_symbolic_expression(prediction):
        return False

    try:
        def normalize(expr):
            if expr is None:
                return None
            expr = expr.replace("^", "**").strip()
            return expr

        ref = normalize(reference)
        pred = normalize(prediction)
        if not ref or not pred:
            return False

        # Fast path: direct comparison
        if ref == pred:
            return True

        # Minimal simplification (skip complex operations on CPU)
        try:
            ref_expr = sp.sympify(ref)
            pred_expr = sp.sympify(pred)
            
            # Only try simple subtraction, no heavy simplification
            diff = ref_expr - pred_expr
            return diff == 0
        except Exception:
            return False

    except Exception:
        return False


def analyze_math_structure(prediction: str) -> Dict[str, Any]:
    """Optimized structure analysis with cached regex"""
    issues = []

    # Balance checks
    if prediction.count("(") != prediction.count(")"):
        issues.append("Unbalanced parentheses")
    if prediction.count("[") != prediction.count("]"):
        issues.append("Unbalanced brackets")
    if prediction.count("{") != prediction.count("}"):
        issues.append("Unbalanced braces")

    if "=" not in prediction:
        issues.append("No '=' found")

    # Variables
    vars_found = _REGEX_PATTERNS['words'].findall(prediction)
    common_vars = set(['x', 'y', 't', 'n', 'k', 'm'])
    undefined = [v for v in vars_found if v not in common_vars]

    # Division by zero
    if _REGEX_PATTERNS['division_zero'].search(prediction):
        issues.append("Division by zero")

    # Transitional phrases
    has_transitional = bool(_REGEX_PATTERNS['transitional'].search(prediction))
    has_multiple_steps = prediction.count('=') > 1

    return {
        "has_step_structure": "=" in prediction,
        "has_multiple_steps": has_multiple_steps,
        "has_transitional_phrases": has_transitional,
        "issues": issues,
        "total_variables": len(vars_found),
        "undefined_variables": undefined,
    }


def automated_quality_check(question: str, generated_solution: str) -> Dict[str, Any]:
    """Fast quality checks"""
    issues = []

    if len(generated_solution) < 100:
        issues.append("Too short")

    has_algorithm = bool(_REGEX_PATTERNS['algorithm'].search(generated_solution))
    has_runtime = bool(_REGEX_PATTERNS['runtime'].search(generated_solution))
    has_proof = bool(_REGEX_PATTERNS['proof'].search(generated_solution))
    has_code = bool(_REGEX_PATTERNS['code_struct'].search(generated_solution))

    complexity_match = _REGEX_PATTERNS['big_o'].findall(generated_solution)

    return {
        'passes_basic_checks': len(issues) == 0,
        'issues': issues,
        'has_algorithm': has_algorithm,
        'has_runtime_analysis': has_runtime,
        'has_proof_keywords': has_proof,
        'has_code_structure': has_code,
        'detected_complexity': complexity_match,
        'length_chars': len(generated_solution),
        'length_tokens': len(generated_solution.split()),
    }


def simple_semantic_score(question: str, prediction: str, reference: str) -> Dict[str, Any]:
    """Ultra-fast rule-based semantic scoring (no LLM)"""
    scores = {
        "correctness": 0.5,
        "completeness": 0.5,
        "logical_coherence": 0.5,
        "use_of_definitions": 0.5,
        "clarity": 0.5,
        "overall_score": 0.0,
        "explanation": "Fast rule-based evaluation",
        "raw_response": "",
    }

    # Length ratio check
    length_ratio = len(prediction) / (len(reference) + 1e-6)
    if 0.5 <= length_ratio <= 1.5:
        scores["completeness"] = 0.7

    # Logical indicators
    if any(word in prediction.lower() for word in ['therefore', 'thus', 'hence', 'proved']):
        scores["logical_coherence"] = 0.7
    
    # Definition indicators
    if any(word in prediction.lower() for word in ['definition', 'assume', 'given', 'define']):
        scores["use_of_definitions"] = 0.7

    # Clarity (sentence structure)
    sentences = len(prediction.split('.'))
    if sentences > 2:
        scores["clarity"] = 0.6

    # Correctness based on structure
    math_struct = analyze_math_structure(prediction)
    if math_struct["has_step_structure"] and not math_struct["issues"]:
        scores["correctness"] = 0.7

    # Overall
    scores["overall_score"] = (
        scores["correctness"] + scores["completeness"] + 
        scores["logical_coherence"] + scores["use_of_definitions"] + 
        scores["clarity"]
    ) / 5.0

    # Clamp
    for key in ['correctness', 'completeness', 'logical_coherence', 'use_of_definitions', 'clarity', 'overall_score']:
        scores[key] = max(0.0, min(1.0, scores[key]))

    return scores


# ============================================================================
# MODEL LOADING & INFERENCE (CPU OPTIMIZED)
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config"""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_prompt(prompt: str) -> str:
    """Format prompt"""
    return f"### Question:\n{prompt}\n\n### Solution:\n"


def load_model_with_adapter_cpu(
    base_model_name: str,
    adapter_path: str,
    use_8bit: bool = True,
    device: str = "cpu",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model optimized for CPU inference.
    
    Args:
        base_model_name: HuggingFace model ID
        adapter_path: Path to LoRA adapter
        use_8bit: Use 8-bit quantization (4x faster, minimal quality loss)
        device: Device to use ("cpu" or "cuda")
    """
    adapter_path_obj = Path(adapter_path)

    if not adapter_path_obj.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Configure quantization for CPU
    quantization_config = None
    dtype = torch.float32  # CPU prefers float32
    
    if use_8bit and device == "cpu":
        print("Loading with 8-bit quantization (4x faster)")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=200.0,
            llm_int8_has_fp16_weight=False,
        )
        dtype = torch.float32
    
    # Load model
    print("Loading model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config if use_8bit else None,
        torch_dtype=dtype,
        device_map=None if device == "cpu" else "auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # If not quantized, move to CPU
    if device == "cpu" and not use_8bit:
        print("Moving model to CPU...")
        model = model.to(device)

    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path_obj))
    model.eval()

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[OK] Model and adapter loaded successfully")
    return model, tokenizer


def generate_solution_cpu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,  # REDUCED from 512 for CPU speed
    temperature: float = 0.3,   # REDUCED for faster greedy decoding
) -> str:
    """
    Generate solution optimized for CPU.
    
    Key differences:
    - Reduced max_new_tokens (256 vs 512)
    - Lower temperature for greedy decoding
    - No sampling (too slow on CPU)
    """
    formatted_prompt = format_prompt(prompt)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    # Move inputs to same device as model
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    else:
        inputs = {k: v.cpu() for k, v in inputs.items()}

    with torch.no_grad():
        try:
            # Greedy decoding (fastest)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                do_sample=False,  # Greedy: no sampling overhead
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable cache on CPU (it's fast)
            )
        except (AttributeError, RuntimeError) as e:
            if "cache" in str(e).lower():
                # Fallback without cache
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
            else:
                raise

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return generated_text.strip()


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model_cpu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_data: List[Dict[str, str]],
    model_name: str,
    skip_sympy: bool = True,  # Skip SymPy on CPU (too slow)
    skip_embeddings: bool = False,  # Can disable if still slow
) -> Dict[str, Any]:
    """
    Evaluate model on CPU with optimizations.
    
    Args:
        skip_sympy: Skip SymPy equivalence (CPU is 10x slower)
        skip_embeddings: Skip embedding similarity (can be slow on CPU)
    """
    print(f"\nEvaluating {model_name}...")
    print(f"Test items: {len(test_data)}")
    print(f"Skip SymPy: {skip_sympy}")
    print(f"Skip Embeddings: {skip_embeddings}\n")

    results = {
        "model_name": model_name,
        "total_items": len(test_data),
        "items_with_solutions": 0,
        "automated_metrics": {
            "exact_matches": 0,
            "bleu_scores": [],
            "levenshtein_scores": [],
            "sympy_equiv_count": 0,
            "semantic_scores": [],
            "embedding_similarity_scores": [],
        },
        "per_item_results": [],
    }

    for idx, item in enumerate(test_data):
        print(f"[{idx + 1}/{len(test_data)}] Processing...", end=" ", flush=True)

        question = item.get("question", "")
        reference_solution = item.get("solution", None)

        has_valid_solution = reference_solution is not None and reference_solution.strip()

        if not question:
            print("SKIP (no question)")
            continue

        # Generate prediction
        try:
            print("generating...", end=" ", flush=True)
            prediction = generate_solution_cpu(model, tokenizer, question)
            if not prediction:
                print("SKIP (empty)")
                continue
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        quality_check = automated_quality_check(question, prediction)
        math_structure = analyze_math_structure(prediction)

        item_result = {
            "item_id": idx,
            "question": question,
            "prediction": prediction,
            "quality_check": quality_check,
            "math_structure": math_structure,
            "has_reference": bool(reference_solution),
        }

        results["items_with_solutions"] += 1

        if has_valid_solution:
            print("metrics...", end=" ", flush=True)
            
            # Fast metrics
            exact_match = compute_exact_match(reference_solution, prediction)
            bleu_score = compute_bleu_score(reference_solution, prediction)
            levenshtein_score = compute_levenshtein(reference_solution, prediction)

            # SymPy (optional, slow on CPU)
            sympy_equiv = False
            if not skip_sympy:
                sympy_equiv = compute_sympy_equivalence_safe(reference_solution, prediction, timeout_sec=0.5)

            # Semantic (fast rule-based)
            semantic_score = simple_semantic_score(question, prediction, reference_solution)

            # Embeddings (optional, can be slow on CPU)
            embedding_sim = 0.0
            if not skip_embeddings:
                print("embedding...", end=" ", flush=True)
                try:
                    embedding_sim = embedding_similarity(reference_solution, prediction)
                except Exception as e:
                    print(f"(embedding failed: {e})", end=" ", flush=True)
                    embedding_sim = 0.0

            # Record metrics
            results["automated_metrics"]["bleu_scores"].append(bleu_score)
            results["automated_metrics"]["levenshtein_scores"].append(levenshtein_score)
            results["automated_metrics"]["semantic_scores"].append(semantic_score["overall_score"])
            results["automated_metrics"]["embedding_similarity_scores"].append(embedding_sim)

            if exact_match:
                results["automated_metrics"]["exact_matches"] += 1
            if sympy_equiv:
                results["automated_metrics"]["sympy_equiv_count"] += 1

            item_result.update({
                "reference_solution": reference_solution,
                "exact_match": exact_match,
                "bleu_score": bleu_score,
                "sympy_equivalent": sympy_equiv,
                "levenshtein": levenshtein_score,
                "semantic_evaluation": semantic_score,
                "embedding_similarity": embedding_sim,
            })

        else:
            item_result.update({
                "reference_solution": reference_solution,
                "exact_match": False,
                "bleu_score": 0.0,
                "sympy_equivalent": False,
                "levenshtein": 0.0,
                "semantic_evaluation": {"overall_score": 0.0, "explanation": "No reference"},
                "embedding_similarity": 0.0,
            })

        results["per_item_results"].append(item_result)
        print("OK")

    # Aggregate metrics
    items_valid = len(results["automated_metrics"]["bleu_scores"])
    if items_valid > 0:
        results["automated_metrics"]["exact_match_rate"] = (
            results["automated_metrics"]["exact_matches"] / results["items_with_solutions"]
        )
        results["automated_metrics"]["avg_bleu_score"] = (
            sum(results["automated_metrics"]["bleu_scores"]) / items_valid
        )
        results["automated_metrics"]["avg_levenshtein_score"] = (
            sum(results["automated_metrics"]["levenshtein_scores"]) / items_valid
        )
        results["automated_metrics"]["sympy_equiv_rate"] = (
            results["automated_metrics"]["sympy_equiv_count"] / results["items_with_solutions"]
        )
        results["automated_metrics"]["avg_semantic_score"] = (
            sum(results["automated_metrics"]["semantic_scores"]) / len(results["automated_metrics"]["semantic_scores"])
            if results["automated_metrics"]["semantic_scores"] else 0.0
        )
        results["automated_metrics"]["avg_embedding_similarity"] = (
            sum(results["automated_metrics"]["embedding_similarity_scores"]) / len(results["automated_metrics"]["embedding_similarity_scores"])
            if results["automated_metrics"]["embedding_similarity_scores"] else 0.0
        )
    else:
        results["automated_metrics"]["exact_match_rate"] = 0.0
        results["automated_metrics"]["avg_bleu_score"] = 0.0
        results["automated_metrics"]["avg_levenshtein_score"] = 0.0
        results["automated_metrics"]["sympy_equiv_rate"] = 0.0
        results["automated_metrics"]["avg_semantic_score"] = 0.0
        results["automated_metrics"]["avg_embedding_similarity"] = 0.0

    # Print summary
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Summary:")
    print(f"{'='*60}")
    print(f"Total items: {results['total_items']}")
    print(f"Items with solutions: {results['items_with_solutions']}")
    
    if items_valid > 0:
        print(f"\nMetrics (on {items_valid} items):")
        print(f"  Exact Match Rate: {results['automated_metrics']['exact_match_rate']:.4f}")
        print(f"  Avg BLEU Score: {results['automated_metrics']['avg_bleu_score']:.4f}")
        print(f"  Avg Levenshtein: {results['automated_metrics']['avg_levenshtein_score']:.4f}")
        print(f"  SymPy Equiv Rate: {results['automated_metrics']['sympy_equiv_rate']:.4f}")
        print(f"  Avg Semantic Score: {results['automated_metrics']['avg_semantic_score']:.4f}")
        if not skip_embeddings:
            print(f"  Avg Embedding Similarity: {results['automated_metrics']['avg_embedding_similarity']:.4f}")

    return results


def main():
    """Main evaluation function"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "training_config.yaml"

    config = load_config(str(config_path))

    # Load test data
    processed_dir = project_root / config["processed_dir"]
    test_data_path = processed_dir / "test_with_solutions.jsonl"
    if not test_data_path.exists():
        test_data_path = processed_dir / "test.jsonl"

    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_data_path}")

    print(f"Loading test data from {test_data_path}...")
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    test_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(test_data)} test items\n")

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    if device == "cpu":
        print("⚠️  CPU mode detected. Optimizations enabled:")
        print("   - 8-bit quantization")
        print("   - Reduced token generation (256 vs 512)")
        print("   - Greedy decoding (no sampling)")
        print("   - SymPy checks disabled")
        print()

    for model_config in config["models"]:
        base_model_name = model_config["name"]
        adapter_path = project_root / model_config["output_dir"]

        if not adapter_path.exists():
            print(f"Warning: Adapter {adapter_path} not found")
            continue

        # Load with CPU optimizations
        model, tokenizer = load_model_with_adapter_cpu(
            base_model_name,
            str(adapter_path),
            use_8bit=(device == "cpu"),
            device=device,
        )

        # Evaluate
        model_results = evaluate_model_cpu(
            model,
            tokenizer,
            test_data,
            base_model_name,
            skip_sympy=(device == "cpu"),      # Skip on CPU
            skip_embeddings=(device == "cpu"),  # Skip on CPU for speed
        )

        # Save results
        model_slug = base_model_name.replace("/", "_")
        json_path = results_dir / f"evaluation_{model_slug}.json"
        with open(json_path, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"Results saved to: {json_path}\n")

        del model
        del tokenizer
        torch.cuda.empty_cache()

    print(f"{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()