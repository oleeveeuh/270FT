def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise GenerationTimeoutError(f"Generation timed out after {GENERATION_TIMEOUT} seconds")


def detect_problem_type(question: str, reference_solution: str = None) -> str:
    """
    Detect type of problem to select appropriate evaluation method.

    Args:
        question: The problem question/prompt
        reference_solution: The reference solution (if available)

    Returns:
        str: "algorithmic", "proof", or "general"
    """
    question_lower = question.lower()
    reference_lower = reference_solution.lower() if reference_solution else ""

    # Algorithmic/complexity indicators
    complexity_patterns = [
        r'o\([^)]+\)',  # Big-O notation
        r'complexity', r'time complexity', r'space complexity',
        r'algorithm', r'runtime', r'asymptotic',
        r'recurrence.*t\(n\)',  # Recurrence relations
    ]

    # Proof indicators
    proof_patterns = [
        r'prove', r'proof', r'show that', r'demonstrate',
        r'lemma', r'theorem', r'collorary', r'proposition',
        r'by induction', r'by contradiction', r'by construction',
        r'assume', r'let.*prove', r'suppose',
        r'base case', r'inductive step', r'hypothesis',
        r'invariant', r'correctness', r'validity',
    ]

    # Check for complexity/runtime problems first
    for pattern in complexity_patterns:
        if re.search(pattern, question_lower) or re.search(pattern, reference_lower):
            return "algorithmic"

    # Then check for proof problems
    for pattern in proof_patterns:
        if re.search(pattern, question_lower) or re.search(pattern, reference_lower):
            return "proof"

    # Default to general
    return "general"