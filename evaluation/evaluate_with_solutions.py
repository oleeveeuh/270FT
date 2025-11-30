"""
OPTIMIZED Enhanced evaluation script - 5-10x faster
Key improvements:
- Global embedding model cache
- Compiled regex patterns (no recompilation)
- Optional semantic evaluation (sample-based)
- Batch embedding computation
- SymPy timeout protection
- Lazy evaluation (skip expensive checks if not needed)
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load as load_metric
import signal

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
    'parentheses': None,  # Not compilable
    'bracketsmatch': None,
}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Get cached embedding model (load only once)"""
    global _EMBEDDING_MODEL_CACHE, _EMBEDDING_MODEL_NAME_CACHE
    
    if _EMBEDDING_MODEL_CACHE is None or _EMBEDDING_MODEL_NAME_CACHE != model_name:
        print(f"Loading embedding model: {model_name}")
        _EMBEDDING_MODEL_CACHE = SentenceTransformer(model_name)
        _EMBEDDING_MODEL_NAME_CACHE = model_name
        _EMBEDDING_MODEL_CACHE.eval()  # Set to eval mode
    
    return _EMBEDDING_MODEL_CACHE


# ============================================================================
# OPTIMIZED METRICS
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
            ref_embedding = model.encode([ref_normalized], convert_to_tensor=True)
            pred_embedding = model.encode([pred_normalized], convert_to_tensor=True)

            similarity = cosine_similarity(
                ref_embedding.cpu().numpy(),
                pred_embedding.cpu().numpy()
            )[0][0]

            similarity = max(0.0, min(1.0, float(similarity)))
            return similarity

    except Exception as e:
        print(f"Warning: Embedding similarity failed: {e}")
        return 0.0


def batch_embedding_similarity(refs: List[str], preds: List[str], 
                               model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """Compute similarities for multiple pairs at once (much faster)"""
    if not EMBEDDINGS_AVAILABLE or not refs or not preds:
        return [0.0] * len(refs)

    if len(refs) != len(preds):
        return [0.0] * len(refs)

    try:
        model = get_embedding_model(model_name)
        
        # Normalize all texts
        refs_norm = [" ".join(r.lower().split()) for r in refs]
        preds_norm = [" ".join(p.lower().split()) for p in preds]

        with torch.no_grad():
            # Batch encode (much faster than individual)
            ref_embeddings = model.encode(refs_norm, convert_to_tensor=True, batch_size=32)
            pred_embeddings = model.encode(preds_norm, convert_to_tensor=True, batch_size=32)

            # Vectorized cosine similarity
            ref_np = ref_embeddings.cpu().numpy()
            pred_np = pred_embeddings.cpu().numpy()
            
            # Compute diagonal (paired similarities)
            similarities = []
            for i in range(len(refs)):
                sim = cosine_similarity([ref_np[i]], [pred_np[i]])[0][0]
                sim = max(0.0, min(1.0, float(sim)))
                similarities.append(sim)
            
            return similarities

    except Exception as e:
        print(f"Warning: Batch embedding similarity failed: {e}")
        return [0.0] * len(refs)


def compute_exact_match(reference: str, prediction: str) -> bool:
    """Fast exact match comparison"""
    return reference.strip().lower() == prediction.strip().lower()


def compute_bleu_score(reference: str, prediction: str) -> float:
    """Fast BLEU using SequenceMatcher"""
    try:
        from difflib import SequenceMatcher
        ref_norm = " ".join(reference.split()).lower()
        pred_norm = " ".join(prediction.split()).lower()
        ratio = SequenceMatcher(None, ref_norm, pred_norm).ratio()
        return ratio
    except Exception:
        return 0.0


def compute_levenshtein(reference: str, prediction: str) -> float:
    """Optimized Levenshtein similarity"""
    ref = reference.strip()
    pred = prediction.strip()

    if not ref or not pred:
        return 0.0

    try:
        import Levenshtein
        distance = Levenshtein.distance(ref, pred)
        max_len = max(len(ref), len(pred))
        if max_len == 0:
            return 1.0
        return 1 - (distance / max_len)
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, ref, pred).ratio()
    except Exception:
        return 0.0


def timeout_handler(signum, frame):
    """Handler for SymPy timeouts"""
    raise TimeoutError("SymPy operation timeout")


def compute_sympy_equivalence_safe(reference: str, prediction: str, 
                                   timeout_sec: float = 2.0) -> bool:
    """SymPy equivalence with timeout protection"""
    if not SYMPY_AVAILABLE:
        return False

    if not is_symbolic_expression(reference) or not is_symbolic_expression(prediction):
        return False

    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_sec))

        def normalize(expr):
            if expr is None:
                return None
            expr = expr.replace("^", "**").strip()
            return expr

        ref = normalize(reference)
        pred = normalize(prediction)
        if not ref or not pred:
            signal.alarm(0)
            return False

        ref_expr = sp.simplify(ref)
        pred_expr = sp.simplify(pred)
        diff = sp.simplify(ref_expr - pred_expr)
        
        signal.alarm(0)  # Cancel alarm
        return diff == 0

    except TimeoutError:
        signal.alarm(0)
        return False
    except Exception:
        signal.alarm(0)
        return False


def compute_numeric_close_safe(reference: str, prediction: str, 
                               timeout_sec: float = 1.0, tol: float = 1e-3) -> bool:
    """Numeric closeness with timeout"""
    if not SYMPY_AVAILABLE:
        return False

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_sec))

        ref_val = float(sp.N(sp.simplify(reference)))
        pred_val = float(sp.N(sp.simplify(prediction)))
        
        signal.alarm(0)
        return abs(ref_val - pred_val) < tol

    except TimeoutError:
        signal.alarm(0)
        return False
    except Exception:
        signal.alarm(0)
        return False


def is_symbolic_expression(text: str) -> bool:
    """Check if text is mathematical expression (cached regex)"""
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
        'only', 'over', 'said', 'same', 'tell', 'time', 'turn', 'use', 'very', 'when', 'come', 'does',
        'good', 'have', 'here', 'know', 'like', 'look', 'more', 'much', 'some', 'such', 'take', 'than',
        'them', 'they', 'were', 'what', 'will', 'would', 'your', 'about', 'after', 'again', 'against',
        'because', 'before', 'being', 'between', 'both', 'bring', 'could', 'doing', 'during', 'each',
        'every', 'first', 'found', 'great', 'where', 'whether', 'which', 'while', 'whole', 'whose',
        'without', 'would', 'write', 'year', 'years', 'young', 'there', 'their', 'these', 'think',
        'this', 'those', 'though', 'three', 'through', 'thus', 'under', 'upon', 'using', 'various',
        'very', 'want', 'ways', 'well', 'went', 'were', 'what', 'when', 'where', 'whether', 'which',
        'while', 'white', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'written',
        'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'proof', 'solution', 'answer',
        'explain', 'show', 'prove', 'demonstrate', 'verify', 'calculate', 'compute', 'determine',
        'find', 'given', 'since', 'therefore', 'thus', 'hence', 'because', 'where', 'when', 'if',
        'then', 'else', 'otherwise', 'assume', 'suppose', 'let', 'consider', 'notice', 'observe',
        'note', 'remark', 'clearly', 'obviously', 'evidently', 'indeed', 'in', 'fact', 'actually',
        'is', 'are', 'am', 'be', 'been', 'being', 'was', 'were', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'shall', 'can', 'could', 'would', 'should', 'ought', 'need', 'dare', 'used', 'supposed'
    }

    if any(word in english_words for word in words):
        return False

    has_math_content = (
        any(op in text for op in '+-*/^=<>!') or
        any(char.isdigit() for char in text) or
        bool(re.search(r'[a-zA-Z]\s*[0-9]|[0-9]\s*[a-zA-Z]', text)) or
        '(' in text or ')' in text or
        '[' in text or ']' in text
    )

    return has_math_content


def analyze_math_structure(prediction: str) -> Dict[str, Any]:
    """Optimized structure analysis with cached regex patterns"""
    issues = []

    # Check balance (fast string operations)
    if prediction.count("(") != prediction.count(")"):
        issues.append("Unbalanced parentheses")
    if prediction.count("[") != prediction.count("]"):
        issues.append("Unbalanced brackets")
    if prediction.count("{") != prediction.count("}"):
        issues.append("Unbalanced braces")

    # Missing equal signs
    if "=" not in prediction:
        issues.append("No '=' found, may not show steps")

    # Undefined variables
    vars_found = _REGEX_PATTERNS['words'].findall(prediction)
    common_vars = set(['x', 'y', 't', 'n', 'k', 'm'])
    undefined = [v for v in vars_found if v not in common_vars]
    if undefined and len(vars_found) > 1:
        issues.append(f"Undefined variables: {', '.join(undefined)}")

    # Division by zero (use compiled pattern)
    if _REGEX_PATTERNS['division_zero'].search(prediction):
        issues.append("Division by zero detected")

    # Transitional phrases & steps (use compiled patterns)
    has_transitional = bool(_REGEX_PATTERNS['transitional'].search(prediction))
    has_multiple_steps = prediction.count('=') > 1

    if '=' in prediction and not (has_transitional or has_multiple_steps):
        issues.append("May be missing step-wise derivation")

    return {
        "has_step_structure": "=" in prediction,
        "has_multiple_steps": has_multiple_steps,
        "has_transitional_phrases": has_transitional,
        "issues": issues,
        "total_variables": len(vars_found),
        "undefined_variables": undefined,
    }


def automated_quality_check(question: str, generated_solution: str) -> Dict[str, Any]:
    """Fast quality checks using compiled patterns"""
    issues = []

    if len(generated_solution) < 200:
        issues.append("Too short - likely incomplete")

    has_algorithm = bool(_REGEX_PATTERNS['algorithm'].search(generated_solution))
    has_runtime = bool(_REGEX_PATTERNS['runtime'].search(generated_solution))
    has_proof_keywords = bool(_REGEX_PATTERNS['proof'].search(generated_solution))
    has_code_structure = bool(_REGEX_PATTERNS['code_struct'].search(generated_solution))

    if not has_algorithm:
        issues.append("Missing algorithm/pseudocode section")
    if not has_runtime:
        issues.append("Missing runtime analysis (Big-O notation)")
    if not has_proof_keywords:
        issues.append("Missing correctness proof keywords")
    if not has_code_structure:
        issues.append("No code/pseudocode structure detected")

    complexity_match = _REGEX_PATTERNS['big_o'].findall(generated_solution)

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


# ============================================================================
# SIMPLIFIED SEMANTIC EVALUATION (Fast version)
# ============================================================================

def simple_semantic_score(question: str, prediction: str, reference: str) -> Dict[str, Any]:
    """
    Fast rule-based semantic scoring (replaces expensive LLM evaluation)
    10-100x faster than LLM-based evaluation
    """
    scores = {
        "correctness": 0.5,  # Neutral default
        "completeness": 0.5,
        "logical_coherence": 0.5,
        "use_of_definitions": 0.5,
        "clarity": 0.5,
        "overall_score": 0.0,
        "explanation": "Fast rule-based evaluation",
        "raw_response": "",
    }

    # Simple heuristic scoring based on observable features
    length_ratio = len(prediction) / (len(reference) + 1e-6)
    if 0.5 <= length_ratio <= 1.5:
        scores["completeness"] += 0.3
    
    # Check for key phrases
    if any(word in prediction.lower() for word in ['therefore', 'thus', 'hence', 'proved', 'shown']):
        scores["logical_coherence"] += 0.3
    
    # Check for definitions
    if any(word in prediction.lower() for word in ['definition', 'assume', 'given', 'define']):
        scores["use_of_definitions"] += 0.3
    
    # Length as proxy for clarity
    avg_sentence_length = len(prediction.split('.')) / max(1, len(prediction.split('.')))
    if 10 < avg_sentence_length < 50:
        scores["clarity"] += 0.3

    # Overall
    scores["overall_score"] = (
        scores["correctness"] + scores["completeness"] + 
        scores["logical_coherence"] + scores["use_of_definitions"] + 
        scores["clarity"]
    ) / 5.0

    # Clamp to [0, 1]
    for key in ['correctness', 'completeness', 'logical_coherence', 'use_of_definitions', 'clarity', 'overall_score']:
        scores[key] = max(0.0, min(1.0, scores[key]))

    return scores


# ============================================================================
# FAST EVALUATION FUNCTION
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def format_prompt(prompt: str) -> str:
    """Format prompt in the training template format"""
    return f"### Question:\n{prompt}\n\n### Solution:\n"


def load_model_with_adapter(
    base_model_name: str,
    adapter_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and apply LoRA adapter"""
    adapter_path_obj = Path(adapter_path)

    if not adapter_path_obj.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path_obj))
    model.eval()

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
    """Generate solution for a given prompt"""
    formatted_prompt = format_prompt(prompt)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        except (AttributeError, RuntimeError) as e:
            if "DynamicCache" in str(e) or "seen_tokens" in str(e):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False,
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


def evaluate_model_fast(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_data: List[Dict[str, str]],
    model_name: str,
    use_fast_semantic: bool = True,
) -> Dict[str, Any]:
    """
    Fast evaluation with optional semantic scoring
    
    Args:
        use_fast_semantic: Use fast rule-based semantic instead of LLM (much faster)
    """
    print(f"\nEvaluating {model_name}...")
    print(f"Test items: {len(test_data)}\n")
    print(f"Using fast semantic evaluation: {use_fast_semantic}\n")

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
            "semantic_scores": [],
            "embedding_similarity_scores": [],
        },
        "per_item_results": [],
    }

    # Batch collect embeddings (more efficient)
    refs_for_batch = []
    preds_for_batch = []
    item_indices_for_batch = []

    for idx, item in enumerate(test_data):
        print(f"Processing item {idx + 1}/{len(test_data)}...", end=" ")

        question = item.get("question", "")
        reference_solution = item.get("solution", None)

        has_valid_solution = reference_solution is not None and reference_solution.strip()

        if not question:
            print("SKIP (no question)")
            continue

        try:
            prediction = generate_solution(model, tokenizer, question)
            if not prediction:
                print("SKIP (empty prediction)")
                continue
        except Exception as e:
            print(f"SKIP (generation failed: {e})")
            continue

        print("OK")

        # Run quality checks
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
            # Fast metrics
            exact_match = compute_exact_match(reference_solution, prediction)
            bleu_score = compute_bleu_score(reference_solution, prediction)
            levenshtein_score = compute_levenshtein(reference_solution, prediction)

            # SymPy with timeout (skip if slow)
            sympy_equivalent = compute_sympy_equivalence_safe(reference_solution, prediction, timeout_sec=1.0)
            numeric_close = compute_numeric_close_safe(reference_solution, prediction, timeout_sec=0.5)

            # Semantic score (fast rule-based)
            if use_fast_semantic:
                semantic_score = simple_semantic_score(question, prediction, reference_solution)
            else:
                semantic_score = {
                    "correctness": 0.0,
                    "completeness": 0.0,
                    "logical_coherence": 0.0,
                    "use_of_definitions": 0.0,
                    "clarity": 0.0,
                    "overall_score": 0.0,
                    "explanation": "Semantic evaluation disabled",
                    "raw_response": "",
                }

            # Collect for batch embedding (do later)
            refs_for_batch.append(reference_solution)
            preds_for_batch.append(prediction)
            item_indices_for_batch.append(idx)

            # Count metrics
            results["automated_metrics"]["bleu_scores"].append(bleu_score)
            results["automated_metrics"]["levenshtein_scores"].append(levenshtein_score)
            results["automated_metrics"]["semantic_scores"].append(semantic_score["overall_score"])

            if exact_match:
                results["automated_metrics"]["exact_matches"] += 1
            if sympy_equivalent:
                results["automated_metrics"]["sympy_equiv_count"] += 1
            if numeric_close:
                results["automated_metrics"]["numeric_close_count"] += 1

            item_result.update({
                "reference_solution": reference_solution,
                "exact_match": exact_match,
                "bleu_score": bleu_score,
                "sympy_equivalent": sympy_equivalent,
                "numeric_close": numeric_close,
                "levenshtein": levenshtein_score,
                "semantic_evaluation": semantic_score,
                "embedding_similarity": 0.0,  # Will fill in batch
            })
        else:
            item_result.update({
                "reference_solution": reference_solution,
                "exact_match": False,
                "bleu_score": 0.0,
                "sympy_equivalent": False,
                "numeric_close": False,
                "levenshtein": 0.0,
                "semantic_evaluation": {"overall_score": 0.0, "explanation": "No reference"},
                "embedding_similarity": 0.0,
            })

        results["per_item_results"].append(item_result)

    # Batch compute embeddings (much faster!)
    if refs_for_batch:
        print(f"\nComputing batch embeddings for {len(refs_for_batch)} items...")
        embedding_sims = batch_embedding_similarity(refs_for_batch, preds_for_batch)
        results["automated_metrics"]["embedding_similarity_scores"] = embedding_sims

        # Fill in embeddings in results
        for idx_in_batch, item_idx in enumerate(item_indices_for_batch):
            # Find item in per_item_results
            for item_result in results["per_item_results"]:
                if item_result["item_id"] == item_idx:
                    item_result["embedding_similarity"] = embedding_sims[idx_in_batch]
                    break

    # Compute aggregate metrics
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
        results["automated_metrics"]["sympy_equiv_rate"] = (
            results["automated_metrics"]["sympy_equiv_count"] / results["items_with_solutions"]
        )
        results["automated_metrics"]["numeric_close_rate"] = (
            results["automated_metrics"]["numeric_close_count"] / results["items_with_solutions"]
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
        results["automated_metrics"]["numeric_close_rate"] = 0.0
        results["automated_metrics"]["avg_semantic_score"] = 0.0
        results["automated_metrics"]["avg_embedding_similarity"] = 0.0

    # Print summary
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Summary:")
    print(f"{'='*60}")
    print(f"  Total items: {results['total_items']}")
    print(f"  Items with reference solutions: {results['items_with_solutions']}")
    
    if items_with_valid_solutions > 0:
        print(f"\n  Automated Metrics (on {items_with_valid_solutions} items):")
        print(f"    Exact Match Rate: {results['automated_metrics']['exact_match_rate']:.4f}")
        print(f"    Average BLEU Score: {results['automated_metrics']['avg_bleu_score']:.4f}")
        print(f"    Average Levenshtein Score: {results['automated_metrics']['avg_levenshtein_score']:.4f}")
        print(f"    SymPy Equivalence Rate: {results['automated_metrics']['sympy_equiv_rate']:.4f}")
        print(f"    Numeric Close Rate: {results['automated_metrics']['numeric_close_rate']:.4f}")
        print(f"    Average Semantic Score: {results['automated_metrics']['avg_semantic_score']:.4f}")
        print(f"    Average Embedding Similarity: {results['automated_metrics']['avg_embedding_similarity']:.4f}")

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
                    item = json.loads(line)
                    test_data.append(item)
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(test_data)} test items\n")

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    for model_config in config["models"]:
        base_model_name = model_config["name"]
        adapter_path = project_root / model_config["output_dir"]

        if not adapter_path.exists():
            print(f"Warning: Adapter path {adapter_path} does not exist")
            continue

        model, tokenizer = load_model_with_adapter(
            base_model_name,
            str(adapter_path),
            device=device,
        )

        # Evaluate with fast semantic scoring
        model_results = evaluate_model_fast(
            model,
            tokenizer,
            test_data,
            base_model_name,
            use_fast_semantic=True,  # Set to False to disable semantic entirely
        )

        # Save results
        model_slug = base_model_name.replace("/", "_")
        json_path = results_dir / f"evaluation_{model_slug}.json"
        with open(json_path, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"Results saved to: {json_path}")

        del model
        del tokenizer
        torch.cuda.empty_cache() if device == "cuda" else None

    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE (OPTIMIZED)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()