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

# Try to import sentence-transformers for embedding similarity
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


def embedding_similarity(ref: str, pred: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Compute cosine similarity between reference and prediction using sentence embeddings.

    Args:
        ref: Reference text string
        pred: Prediction text string
        model_name: Name of the sentence transformer model to use

    Returns:
        float: Cosine similarity score between 0 and 1
    """
    if not EMBEDDINGS_AVAILABLE:
        print("Warning: sentence-transformers not available for embedding similarity")
        return 0.0

    if not ref or not pred:
        return 0.0

    try:
        # Load the sentence transformer model (cached after first load)
        if not hasattr(embedding_similarity, '_model') or embedding_similarity._model_name != model_name:
            print(f"Loading sentence transformer model: {model_name}")
            embedding_similarity._model = SentenceTransformer(model_name)
            embedding_similarity._model_name = model_name

        model = embedding_similarity._model

        # Normalize texts: remove extra whitespace, convert to lowercase
        ref_normalized = " ".join(ref.lower().split())
        pred_normalized = " ".join(pred.lower().split())

        # Generate embeddings
        ref_embedding = model.encode([ref_normalized], convert_to_tensor=True)
        pred_embedding = model.encode([pred_normalized], convert_to_tensor=True)

        # Compute cosine similarity
        similarity = cosine_similarity(
            ref_embedding.cpu().numpy(),
            pred_embedding.cpu().numpy()
        )[0][0]

        # Ensure result is in [0, 1] range
        similarity = max(0.0, min(1.0, float(similarity)))

        return similarity

    except Exception as e:
        print(f"Warning: Embedding similarity computation failed: {e}")
        return 0.0


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


def is_symbolic_expression(text: str) -> bool:
    """
    Check if text contains only valid mathematical symbols for SymPy expression.

    Args:
        text: Text to validate

    Returns:
        bool: True if text is a valid SymPy-expressible mathematical expression
    """
    if not text or not text.strip():
        return False

    # Normalize the text
    text = text.strip()

    # Define allowed characters in mathematical expressions
    # Numbers, letters (variables), mathematical operators, parentheses, brackets, braces
    allowed_chars = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/^=<>!&|.,;:_ ')
    allowed_special = set('()[]{}')

    # Check each character
    for char in text:
        if char not in allowed_chars and char not in allowed_special:
            # If character is not allowed, check if it's a common math function or constant
            if char not in 'π∞√∑∏∫±≤≥≠≈∈∉⊆⊇∪∩':
                return False

    # Check for English words (common words that shouldn't be in mathematical expressions)
    # This is a heuristic - check for common English words longer than 2 characters
    import re
    # Extract words that are just letters (not variables like x, y, n, etc.)
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())

    # Common English words that indicate this isn't a pure mathematical expression
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

    # Check if any English words are found
    found_english = [word for word in words if word in english_words]
    if found_english:
        return False

    # Check if text looks like a mathematical expression
    # Should contain at least one mathematical operator or number/variable combination
    has_math_content = (
        any(op in text for op in '+-*/^=<>!') or  # operators
        any(char.isdigit() for char in text) or   # numbers
        bool(re.search(r'[a-zA-Z]\s*[0-9]|[0-9]\s*[a-zA-Z]', text)) or  # variable-number combos
        '(' in text or ')' in text or  # parentheses
        '[' in text or ']' in text    # brackets
    )

    return has_math_content


def compute_sympy_equivalence(reference: str, prediction: str) -> bool:
    """
    Use SymPy to check symbolic equivalence between reference and prediction.
    Handles simple algebraic expressions and normalizes formatting.

    Only runs symbolic evaluation if BOTH reference and prediction are valid
    SymPy-expressible strings (no English words, only mathematical symbols).

    Returns True if expressions simplify to the same value.
    """
    if not SYMPY_AVAILABLE:
        print("Warning: SymPy not available for symbolic equivalence checking")
        return False

    # Check if both texts are valid symbolic expressions
    if not is_symbolic_expression(reference) or not is_symbolic_expression(prediction):
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


def evaluate_proof_semantically(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    prediction: str,
    reference: str,
    judge_model_name: Optional[str] = None,
    judge_tokenizer: Optional[AutoTokenizer] = None,
) -> Dict[str, Any]:
    """
    Use LLM-based semantic evaluation to assess proof quality.

    Args:
        model: The model being evaluated (or base model for generation)
        tokenizer: Tokenizer for the model being evaluated
        question: The original question/prompt
        prediction: The model's generated solution/proof
        reference: The reference solution
        judge_model_name: Optional judge model (uses base model if None)
        judge_tokenizer: Optional judge tokenizer (uses base tokenizer if None)

    Returns:
        Dictionary with rubric scores and explanation
    """
    # Use provided judge model/tokenizer or fall back to base model
    judge_model = model
    judge_tokenizer = tokenizer

    # Create grading prompt
    grading_prompt = f"""You are an expert mathematics and computer science evaluator.
Please evaluate the quality of a student's proof/solution compared to a reference solution.

**Question:**
{question}

**Reference Solution:**
{reference}

**Student's Solution:**
{prediction}

**Evaluation Rubric:**
Please score each criterion on a scale of 0.0 to 1.0:

1. **Correctness**: Is the solution mathematically/logically correct?
2. **Completeness**: Does it address all parts of the question comprehensively?
3. **Logical Coherence**: Is the reasoning clear, logical, and well-structured?
4. **Use of Definitions**: Does it properly use and reference relevant definitions/theorems?
5. **Clarity**: Is the explanation clear, readable, and easy to follow?

**Format your response as:**
Correctness: X.XX
Completeness: X.XX
Logical Coherence: X.XX
Use of Definitions: X.XX
Clarity: X.XX
Overall Score: X.XX
Explanation: [Brief explanation of your reasoning]"""

    try:
        # Generate evaluation from the judge model
        inputs = judge_tokenizer(
            grading_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3500,  # Allow longer input for evaluation
        ).to(judge_model.device)

        with torch.no_grad():
            eval_outputs = judge_model.generate(
                **inputs,
                max_new_tokens=256,  # Limit response length
                temperature=0.3,  # Lower temperature for more consistent evaluation
                do_sample=True,
                pad_token_id=judge_tokenizer.pad_token_id,
                eos_token_id=judge_tokenizer.eos_token_id,
                use_cache=False,
            )

        eval_response = judge_tokenizer.decode(
            eval_outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Parse the evaluation response
        scores = {
            "correctness": 0.0,
            "completeness": 0.0,
            "logical_coherence": 0.0,
            "use_of_definitions": 0.0,
            "clarity": 0.0,
            "overall_score": 0.0,
            "explanation": "Failed to parse evaluation",
            "raw_response": eval_response,
        }

        # Extract scores using regex
        import re

        patterns = {
            "correctness": r"Correctness:\s*(\d+\.?\d*)",
            "completeness": r"Completeness:\s*(\d+\.?\d*)",
            "logical_coherence": r"Logical Coherence:\s*(\d+\.?\d*)",
            "use_of_definitions": r"Use of Definitions:\s*(\d+\.?\d*)",
            "clarity": r"Clarity:\s*(\d+\.?\d*)",
            "overall_score": r"Overall Score:\s*(\d+\.?\d*)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, eval_response, re.IGNORECASE)
            if match:
                try:
                    scores[key] = float(match.group(1))
                    # Clamp scores to [0, 1] range
                    scores[key] = max(0.0, min(1.0, scores[key]))
                except ValueError:
                    scores[key] = 0.0

        # Extract explanation
        explanation_match = re.search(r"Explanation:\s*(.+)", eval_response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            scores["explanation"] = explanation_match.group(1).strip()

        # Calculate overall score if not provided
        if scores["overall_score"] == 0.0:
            individual_scores = [
                scores["correctness"],
                scores["completeness"],
                scores["logical_coherence"],
                scores["use_of_definitions"],
                scores["clarity"]
            ]
            scores["overall_score"] = sum(individual_scores) / len(individual_scores)

        return scores

    except Exception as e:
        print(f"Warning: Semantic evaluation failed: {e}")
        return {
            "correctness": 0.0,
            "completeness": 0.0,
            "logical_coherence": 0.0,
            "use_of_definitions": 0.0,
            "clarity": 0.0,
            "overall_score": 0.0,
            "explanation": f"Evaluation failed: {str(e)}",
            "raw_response": "",
        }


def compute_rubric_score(
    structure_score: float,
    semantic_score: float,
    embedding_score: float,
    structure_weight: float = 0.3,
    semantic_weight: float = 0.5,
    embedding_weight: float = 0.2
) -> Dict[str, float]:
    """
    Compute unified rubric score combining structure, semantic, and embedding scores.

    Args:
        structure_score: Score from structural analysis (0-1)
        semantic_score: Score from LLM semantic evaluation (0-1)
        embedding_score: Score from embedding similarity (0-1)
        structure_weight: Weight for structure score (default: 0.3)
        semantic_weight: Weight for semantic score (default: 0.5)
        embedding_weight: Weight for embedding score (default: 0.2)

    Returns:
        Dictionary with individual scores and weighted overall score
    """
    # Validate inputs are in [0, 1] range
    structure_score = max(0.0, min(1.0, structure_score))
    semantic_score = max(0.0, min(1.0, semantic_score))
    embedding_score = max(0.0, min(1.0, embedding_score))

    # Normalize weights to sum to 1.0
    total_weight = structure_weight + semantic_weight + embedding_weight
    if total_weight > 0:
        structure_weight = structure_weight / total_weight
        semantic_weight = semantic_weight / total_weight
        embedding_weight = embedding_weight / total_weight
    else:
        # Default weights if all are zero
        structure_weight = 0.3
        semantic_weight = 0.5
        embedding_weight = 0.2

    # Compute weighted overall score
    overall_score = (
        structure_score * structure_weight +
        semantic_score * semantic_weight +
        embedding_score * embedding_weight
    )

    return {
        "structure_score": structure_score,
        "semantic_score": semantic_score,
        "embedding_score": embedding_score,
        "overall_score": overall_score,
        "weights": {
            "structure_weight": structure_weight,
            "semantic_weight": semantic_weight,
            "embedding_weight": embedding_weight
        }
    }


def evaluate_proof_with_rubric(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    prediction: str,
    reference: str,
    structure_weight: float = 0.3,
    semantic_weight: float = 0.5,
    embedding_weight: float = 0.2
) -> Dict[str, Any]:
    """
    Evaluate a proof using comprehensive rubric combining multiple metrics.

    Args:
        model: The model being evaluated
        tokenizer: Tokenizer for the model
        question: The original question/prompt
        prediction: The model's generated solution/proof
        reference: The reference solution
        structure_weight: Weight for structure score (default: 0.3)
        semantic_weight: Weight for semantic score (default: 0.5)
        embedding_weight: Weight for embedding score (default: 0.2)

    Returns:
        Dictionary with comprehensive rubric evaluation
    """
    # 1. Structure Score: Based on mathematical structure analysis
    math_structure = analyze_math_structure(prediction)

    # Compute structure score based on:
    # - Has step structure (0.3)
    # - No structural issues (0.3)
    # - Has multiple steps (0.2)
    # - Has transitional phrases (0.1)
    # - No undefined variables (0.1)
    structure_score = 0.0

    if math_structure["has_step_structure"]:
        structure_score += 0.3
    if not math_structure["issues"]:  # No structural issues
        structure_score += 0.3
    if math_structure["has_multiple_steps"]:
        structure_score += 0.2
    if math_structure["has_transitional_phrases"]:
        structure_score += 0.1
    if not math_structure["undefined_variables"]:  # No undefined variables
        structure_score += 0.1

    # 2. Semantic Score: From LLM-based evaluation
    semantic_evaluation = evaluate_proof_semantically(
        model, tokenizer, question, prediction, reference
    )
    semantic_score = semantic_evaluation["overall_score"]

    # 3. Embedding Score: From embedding similarity
    embedding_score = embedding_similarity(reference, prediction)

    # 4. Compute overall rubric score
    rubric_scores = compute_rubric_score(
        structure_score, semantic_score, embedding_score,
        structure_weight, semantic_weight, embedding_weight
    )

    # 5. Add detailed breakdown and metadata
    result = {
        **rubric_scores,
        "detailed_structure": math_structure,
        "detailed_semantic": semantic_evaluation,
        "question_length": len(question),
        "prediction_length": len(prediction),
        "reference_length": len(reference) if reference else 0,
        "has_reference": bool(reference and reference.strip()),
        "evaluation_timestamp": str(__import__('datetime').datetime.now())
    }

    return result


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
            "semantic_scores": [],
            "embedding_similarity_scores": [],
            "rubric_scores": [],
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

            # Run semantic evaluation for items with reference solutions
            semantic_score = evaluate_proof_semantically(model, tokenizer, question, prediction, reference_solution)

            # Compute embedding similarity
            embedding_sim_score = embedding_similarity(reference_solution, prediction)

            # Compute comprehensive rubric score
            rubric_evaluation = evaluate_proof_with_rubric(
                model, tokenizer, question, prediction, reference_solution
            )

            # Always append scores (for both exact matches and non-matches)
            results["automated_metrics"]["bleu_scores"].append(bleu_score)
            results["automated_metrics"]["levenshtein_scores"].append(levenshtein_score)
            results["automated_metrics"]["semantic_scores"].append(semantic_score["overall_score"])
            results["automated_metrics"]["embedding_similarity_scores"].append(embedding_sim_score)
            results["automated_metrics"]["rubric_scores"].append(rubric_evaluation["overall_score"])

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
                "semantic_evaluation": semantic_score,
                "embedding_similarity": embedding_sim_score,
                "rubric_evaluation": rubric_evaluation,
            })
        else:
            # Empty reference solution - count as having solution but no automated metrics
            print(f"DEBUG: Processing item {idx} WITH empty reference solution")
            semantic_score = {
                "correctness": 0.0,
                "completeness": 0.0,
                "logical_coherence": 0.0,
                "use_of_definitions": 0.0,
                "clarity": 0.0,
                "overall_score": 0.0,
                "explanation": "No reference solution available for semantic evaluation",
                "raw_response": "",
            }
            # Create minimal rubric evaluation for cases without reference solution
            rubric_evaluation = {
                "structure_score": 0.0,
                "semantic_score": 0.0,
                "embedding_score": 0.0,
                "overall_score": 0.0,
                "weights": {"structure_weight": 0.3, "semantic_weight": 0.5, "embedding_weight": 0.2},
                "detailed_structure": {"issues": ["No reference solution"]},
                "detailed_semantic": semantic_score,
                "question_length": len(question),
                "prediction_length": len(prediction),
                "reference_length": 0,
                "has_reference": False,
                "evaluation_timestamp": str(__import__('datetime').datetime.now())
            }

            item_result.update({
                "reference_solution": reference_solution,
                "exact_match": False,
                "bleu_score": 0.0,
                "sympy_equivalent": False,
                "numeric_close": False,
                "levenshtein": 0.0,
                "semantic_evaluation": semantic_score,
                "embedding_similarity": 0.0,
                "rubric_evaluation": rubric_evaluation,
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

        # Add semantic evaluation aggregate metrics
        if results["automated_metrics"]["semantic_scores"]:
            results["automated_metrics"]["avg_semantic_score"] = (
                sum(results["automated_metrics"]["semantic_scores"]) / len(results["automated_metrics"]["semantic_scores"])
            )
        else:
            results["automated_metrics"]["avg_semantic_score"] = 0.0

        # Add embedding similarity aggregate metrics
        if results["automated_metrics"]["embedding_similarity_scores"]:
            results["automated_metrics"]["avg_embedding_similarity"] = (
                sum(results["automated_metrics"]["embedding_similarity_scores"]) / len(results["automated_metrics"]["embedding_similarity_scores"])
            )
        else:
            results["automated_metrics"]["avg_embedding_similarity"] = 0.0

        # Add rubric score aggregate metrics
        if results["automated_metrics"]["rubric_scores"]:
            results["automated_metrics"]["avg_rubric_score"] = (
                sum(results["automated_metrics"]["rubric_scores"]) / len(results["automated_metrics"]["rubric_scores"])
            )
        else:
            results["automated_metrics"]["avg_rubric_score"] = 0.0
    else:
        results["automated_metrics"]["exact_match_rate"] = 0.0
        results["automated_metrics"]["avg_bleu_score"] = 0.0
        results["automated_metrics"]["avg_levenshtein_score"] = 0.0
        results["automated_metrics"]["sympy_equiv_rate"] = 0.0
        results["automated_metrics"]["avg_levenshtein"] = 0.0
        results["automated_metrics"]["numeric_close_rate"] = 0.0
        results["automated_metrics"]["avg_semantic_score"] = 0.0
        results["automated_metrics"]["avg_embedding_similarity"] = 0.0
        results["automated_metrics"]["avg_rubric_score"] = 0.0

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
        print(f"    Average Semantic Score: {results['automated_metrics']['avg_semantic_score']:.4f}")
        print(f"    Average Embedding Similarity: {results['automated_metrics']['avg_embedding_similarity']:.4f}")
        print(f"    Average Rubric Score: {results['automated_metrics']['avg_rubric_score']:.4f}")

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