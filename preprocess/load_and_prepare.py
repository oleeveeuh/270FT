"""
Load and prepare data from raw files for fine-tuning.

Reads .txt, .json, and .pdf files, extracts Q&A pairs, chunks them into
≤2000 token segments, and splits into train/validation/test sets with temporal ordering.

Supports:
- Textbooks (PDF): Extracts text and identifies Q&A sections
- Lecture slides (PDF): Extracts slide content and examples
- Homework assignments (PDF): Extracts problems and solutions
- Temporal splitting: Past exams → validation, future exams → test
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from collections import namedtuple

from transformers import AutoTokenizer

# PDF processing
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: pdfplumber not installed. PDF processing will be disabled.")
    print("Install with: pip install pdfplumber")


# Use a base tokenizer for token counting (GPT-2 is fast and common)
TOKENIZER_NAME = "gpt2"
MAX_TOKENS = 2000

# File info for temporal/unit-based sorting
FileInfo = namedtuple('FileInfo', ['path', 'date', 'unit', 'is_exam', 'filename'])


def load_tokenizer() -> AutoTokenizer:
    """Load tokenizer for counting tokens."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def parse_json_file(file_path: Path, allow_question_only: bool = False) -> List[Dict[str, str]]:
    """
    Parse JSON file with Question/Solution blocks.

    Expected format:
    {
        "Question": "...",
        "Solution": "..."
    }
    or a list of such objects.

    Args:
        file_path: Path to JSON file
        allow_question_only: If True, allow questions without solutions (for test sets)

    Returns:
        List of Q&A pairs (or question-only dicts if allow_question_only=True)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                question = item.get("Question", item.get("question", ""))
                solution = item.get("Solution", item.get("solution", ""))
                if question and solution:
                    pairs.append({"question": question, "solution": solution})
                elif question and allow_question_only:
                    # For test sets: questions only
                    pairs.append({"question": question})
    elif isinstance(data, dict):
        question = data.get("Question", data.get("question", ""))
        solution = data.get("Solution", data.get("solution", ""))
        if question and solution:
            pairs.append({"question": question, "solution": solution})
        elif question and allow_question_only:
            # For test sets: questions only
            pairs.append({"question": question})

    return pairs


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from PDF file.
    
    Uses pdfplumber for better text extraction, especially for formatted content.
    Falls back to pypdf if pdfplumber fails.
    """
    if not PDF_AVAILABLE:
        raise ImportError("pdfplumber is required for PDF processing. Install with: pip install pdfplumber")
    
    text_content = []
    
    try:
        # Try pdfplumber first (better for formatted content)
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
    except Exception as e:
        print(f"    Warning: pdfplumber extraction failed for {file_path.name}: {e}")
        # Fallback to pypdf
        try:
            import pypdf
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
        except Exception as e2:
            print(f"    Error: Both PDF extraction methods failed: {e2}")
            return ""
    
    return '\n\n'.join(text_content)


def parse_pdf_file(file_path: Path) -> List[Dict[str, str]]:
    """
    Parse PDF file and extract Q&A pairs.

    Handles different PDF types:
    - Textbooks: Looks for chapter sections, examples, exercises
    - Lecture slides: Extracts slide content, examples, problems
    - Homework: Looks for problem/solution patterns

    Returns:
        List of Q&A pairs extracted from PDF
    """
    print(f"    Extracting text from PDF: {file_path.name}")
    text = extract_text_from_pdf(file_path)

    if not text:
        return []

    # Detect PDF type from filename and content
    filename_lower = file_path.name.lower()
    is_homework = any(kw in filename_lower for kw in ['homework', 'assignment', 'hw', 'problem_set'])
    is_lecture = any(kw in filename_lower for kw in ['lecture', 'notes', 'slides'])

    pairs = []

    # Strategy 0: For homework/assignment PDFs, extract Problem/Solution pairs
    if is_homework or 'solution' in filename_lower:
        print(f"    Detected homework/solution PDF")
        # Pattern: "Problem X. [text] Solution. [text]" until next Problem
        problem_solution_pattern = r'Problem\s+(\d+)\.\s*(.+?)\s*Solution\.\s*(.+?)(?=Problem\s+\d+\.|$)'

        matches = list(re.finditer(problem_solution_pattern, text, re.DOTALL | re.IGNORECASE))

        for match in matches:
            prob_num = match.group(1)
            prob_text = match.group(2).strip()
            sol_text = match.group(3).strip()

            # Skip if too short or too long (likely extraction error)
            if len(prob_text) < 30 or len(sol_text) < 30:
                continue
            if len(prob_text) > 15000 or len(sol_text) > 15000:
                continue

            # Clean up the text
            prob_text = clean_text(prob_text)
            sol_text = clean_text(sol_text)

            if prob_text and sol_text:
                pairs.append({
                    "question": f"Problem {prob_num}: {prob_text}",
                    "solution": sol_text
                })

        if pairs:
            print(f"    Extracted {len(pairs)} problem/solution pairs from homework")
            return pairs
        else:
            print(f"    Note: Could not extract structured Q&A from homework PDF")
    
    # For lecture slides without clear Q&A structure, skip them
    # They typically don't have extractable training data
    if is_lecture and len(text) > 100000:
        print(f"    Skipping large lecture slide deck (no clear Q&A structure)")
        return []

    # Strategy 1: Look for explicit Q: / A: or Question: / Answer: patterns
    qa_patterns = [
        r'(?:Q:|Question:)\s*(.+?)(?=\n(?:A:|Answer:))(?:\n(?:A:|Answer:)\s*(.+?))(?=\n(?:Q:|Question:)|$)',
        r'(?:Problem|Exercise)\s*\d+[.:]\s*(.+?)(?=\n(?:Solution|Answer):)(?:\n(?:Solution|Answer):\s*(.+?))(?=\n(?:Problem|Exercise)|$)',
        r'Example\s*\d+[.:]\s*(.+?)(?=\n(?:Solution|Proof):)(?:\n(?:Solution|Proof):\s*(.+?))(?=\n(?:Example|Problem)|$)',
    ]

    for pattern in qa_patterns:
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            question = match.group(1).strip() if match.lastindex >= 1 else ""
            answer = match.group(2).strip() if match.lastindex >= 2 else ""

            # Skip oversized matches (likely extraction errors)
            if len(question) > 10000 or len(answer) > 10000:
                continue

            if question and answer and len(question) > 10 and len(answer) > 10:
                pairs.append({
                    "question": clean_text(question),
                    "solution": clean_text(answer)
                })
    
    # Strategy 2: For lecture slides with explicit Problem/Solution structure
    # Only extract if we find clear delimiters
    if not pairs and is_lecture:
        # Look for explicit Problem/Solution pairs in slides
        problem_solution_pattern = r'(?:Problem|Example|Exercise)[:\s]+(.+?)\s*(?:Solution|Answer|Proof)[:\s]+(.+?)(?=(?:Problem|Example|Exercise|$))'

        matches = re.finditer(problem_solution_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            problem_text = match.group(1).strip()
            solution_text = match.group(2).strip()

            # Strict size limits to avoid document concatenation
            if len(problem_text) > 5000 or len(solution_text) > 5000:
                continue

            if len(problem_text) > 50 and len(solution_text) > 50:
                pairs.append({
                    "question": clean_text(problem_text),
                    "solution": clean_text(solution_text)
                })
    
    # Strategy 3a: For textbooks, extract Solved Exercise pairs
    # Pattern: "Solved Exercise X" followed by problem text and "Solution" section
    if not pairs:
        # Match "Solved Exercise N" ... "Solution" ... (next "Solved Exercise" or end)
        solved_ex_pattern = r'Solved Exercise\s+(\d+)\s+(.+?)\s+Solution\s+(.+?)(?=Solved Exercise|\n\n\n|$)'
        matches = re.finditer(solved_ex_pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            ex_num = match.group(1).strip()
            problem = match.group(2).strip()
            solution = match.group(3).strip()

            # Validate lengths
            if len(problem) < 50 or len(solution) < 50:
                continue
            if len(problem) > 8000 or len(solution) > 12000:
                continue

            # Clean the text
            problem = clean_text(problem)
            solution = clean_text(solution)

            # For textbook content, if it starts lowercase, try to fix by capitalizing
            # (this happens due to PDF column extraction issues)
            if problem and problem[0].islower():
                # Try to find a sentence boundary and start from there
                sentences = re.split(r'[.!?]\s+', problem)
                if len(sentences) > 1:
                    # Start from the second sentence if first is fragment
                    problem = '. '.join(sentences[1:])
                else:
                    # Just capitalize it
                    problem = problem[0].upper() + problem[1:]

            # Only add if solution is substantive (RELAXED: ≥50 chars)
            if len(solution) >= 50 and len(problem) >= 30:
                pairs.append({
                    "question": problem,
                    "solution": solution
                })

        if pairs:
            print(f"    Extracted {len(pairs)} solved exercises from textbook")

    # Strategy 3a2: For textbooks, also extract unsolved "Exercise" sections that have solutions nearby
    # ACCUMULATE with other strategies
    initial_count = len(pairs)
    exercise_pattern = r'Exercise\s+(\d+(?:\.\d+)?)[:\s]+(.+?)(?:\n\n|$)'
    matches = re.finditer(exercise_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        ex_num = match.group(1).strip()
        problem = match.group(2).strip()

        # Look for solution in the next 2000 characters
        solution_start = match.end()
        solution_search = text[solution_start:solution_start + 2000]

        # Try to find "Solution" or answer patterns
        solution_match = re.search(r'(?:Solution|Answer)[:\s]+(.+?)(?:\n\nExercise|\n\n\n|$)', solution_search, re.DOTALL | re.IGNORECASE)

        if solution_match:
            solution = solution_match.group(1).strip()

            # Validate lengths
            if len(problem) < 30 or len(solution) < 30:
                continue
            if len(problem) > 8000 or len(solution) > 12000:
                continue

            # Clean and capitalize if needed
            problem = clean_text(problem)
            solution = clean_text(solution)

            if problem and problem[0].islower():
                sentences = re.split(r'[.!?]\s+', problem)
                if len(sentences) > 1:
                    problem = '. '.join(sentences[1:])
                else:
                    problem = problem[0].upper() + problem[1:]

            pairs.append({
                "question": problem,
                "solution": solution
            })

    if len(pairs) > initial_count:
        print(f"    Extracted {len(pairs) - initial_count} additional exercises from textbook")

    # Strategy 3b: For textbooks, extract theorem/proof pairs
    if not pairs:
        # Look for theorem statements followed by proofs
        theorem_pattern = r'(?:Theorem|Proposition|Lemma|Corollary)\s*\d*[.:]\s*(.+?)(?=\n(?:Proof|Demonstration):)(?:\n(?:Proof|Demonstration):\s*(.+?))(?=\n(?:Theorem|Proposition|Lemma|Corollary)|$)'
        matches = re.finditer(theorem_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            theorem = match.group(1).strip() if match.lastindex >= 1 else ""
            proof = match.group(2).strip() if match.lastindex >= 2 else ""

            # Strict size limits
            if len(theorem) > 2000 or len(proof) > 5000:
                continue

            if theorem and proof and len(theorem) > 10 and len(proof) > 20:
                pairs.append({
                    "question": f"Prove: {clean_text(theorem)}",
                    "solution": clean_text(proof)
                })

    # Strategy 4: Extract conceptual content and frame as questions
    # For lecture slides with definitions, algorithms, concepts, etc.
    # ACCUMULATE with other strategies (removed "if not pairs")
    if is_lecture:
        initial_count = len(pairs)
        print(f"    Attempting conceptual content extraction...")

        # Pattern 1a: Explicit Definition patterns
        # "Definition: X is..." → Q: "What is X?" A: "X is..."
        definition_pattern = r'Definition[:\s]+(?:A\s+)?([A-Z][a-zA-Z\s]+?)\s+is\s+(.+?)(?=\n\n|\nDefinition|\nTheorem|\nAlgorithm|\nExample|$)'
        matches = re.finditer(definition_pattern, text, re.DOTALL)
        for match in matches:
            term = match.group(1).strip()
            definition = match.group(2).strip()

            # Validate length
            if len(definition) < 30 or len(definition) > 3000:
                continue

            question = f"What is {term}?"
            solution = f"{term} is {definition}"

            pairs.append({
                "question": clean_text(question),
                "solution": clean_text(solution)
            })

        # Pattern 1b: Inline definitions (A X is...)
        # Look for sentences that define terms inline
        inline_def_pattern = r'(?:^|\n)(?:A|An)\s+([a-z][a-z\s-]+?)\s+is\s+(?:a|an)\s+(.+?)(?:\.|;|\n)'
        matches = re.finditer(inline_def_pattern, text, re.IGNORECASE)
        for match in matches:
            term = match.group(1).strip()
            definition = match.group(2).strip()

            # Skip very short or very long
            if len(definition) < 20 or len(definition) > 500:
                continue

            # Skip if term is too long (likely not a definition)
            if len(term) > 50:
                continue

            question = f"What is a {term}?"
            solution = f"A {term} is {definition}."

            pairs.append({
                "question": clean_text(question),
                "solution": clean_text(solution)
            })

        # Pattern 1c: "X is defined as..." or "We define X as..."
        defined_as_pattern = r'(?:^|\n)(?:We define |The )?([A-Z][a-zA-Z\s-]+?)\s+is defined as\s+(.+?)(?:\.|;|\n\n)'
        matches = re.finditer(defined_as_pattern, text, re.DOTALL)
        for match in matches:
            term = match.group(1).strip()
            definition = match.group(2).strip()

            # Validate length
            if len(definition) < 20 or len(definition) > 800:
                continue

            if len(term) > 60:
                continue

            question = f"What is {term}?"
            solution = f"{term} is defined as {definition}."

            pairs.append({
                "question": clean_text(question),
                "solution": clean_text(solution)
            })

        # Pattern 2: Algorithm descriptions
        # "Algorithm: [name]. [description]" → Q: "Explain [name]" A: "[description]"
        algorithm_pattern = r'Algorithm[:\s]+([A-Z][a-zA-Z\s\-]+?)[\.:]\s*(.+?)(?=\n\n|\nAlgorithm|\nDefinition|\nTheorem|\nExample|$)'
        matches = re.finditer(algorithm_pattern, text, re.DOTALL)
        for match in matches:
            name = match.group(1).strip()
            description = match.group(2).strip()

            # Validate length
            if len(description) < 50 or len(description) > 5000:
                continue

            question = f"Explain the {name} algorithm."
            solution = clean_text(description)

            pairs.append({
                "question": question,
                "solution": solution
            })

        # Pattern 3: Concept explanations
        # "Concept: [name]. [explanation]" → Q: "Explain [name]" A: "[explanation]"
        concept_pattern = r'Concept[:\s]+([A-Z][a-zA-Z\s\-]+?)[\.:]\s*(.+?)(?=\n\n|\nConcept|\nDefinition|\nTheorem|\nAlgorithm|$)'
        matches = re.finditer(concept_pattern, text, re.DOTALL)
        for match in matches:
            name = match.group(1).strip()
            explanation = match.group(2).strip()

            # Validate length
            if len(explanation) < 50 or len(explanation) > 5000:
                continue

            question = f"Explain {name}."
            solution = clean_text(explanation)

            pairs.append({
                "question": question,
                "solution": solution
            })

        # Pattern 4: Key Points / Important Notes
        # "Key Point: [text]" → Q: "What is an important point about [topic]?" A: "[text]"
        # Extract topic from surrounding context (previous heading)
        key_point_pattern = r'(?:Key Point|Important Note|Note)[:\s]+(.+?)(?=\n\n|\nKey Point|\nImportant Note|\nNote|$)'
        matches = re.finditer(key_point_pattern, text, re.DOTALL)
        for match in matches:
            point = match.group(1).strip()

            # Validate length
            if len(point) < 30 or len(point) > 2000:
                continue

            # Generic question
            question = "Explain this key concept."
            solution = clean_text(point)

            pairs.append({
                "question": question,
                "solution": solution
            })

        if len(pairs) > initial_count:
            print(f"    Extracted {len(pairs) - initial_count} conceptual Q&A pairs from lecture slides")

    # Strategy 5: Extract examples with solutions (for any PDF type)
    # Look for "Example:" followed by solution/answer
    # REMOVED "if not pairs" to accumulate from all strategies
    example_pattern = r'Example[:\s]+\d*[:\s]*(.+?)(?:Solution|Answer|Proof)[:\s]+(.+?)(?=\n\nExample|\n\n\n|$)'
    initial_count = len(pairs)
    matches = re.finditer(example_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        example_text = match.group(1).strip()
        solution_text = match.group(2).strip()

        if len(example_text) < 30 or len(solution_text) < 50:
            continue
        if len(example_text) > 5000 or len(solution_text) > 8000:
            continue

        pairs.append({
            "question": clean_text(example_text),
            "solution": clean_text(solution_text)
        })

    if len(pairs) > initial_count:
        print(f"    Extracted {len(pairs) - initial_count} examples from PDF")

    # Strategy 6: Extract definitions (broader pattern, any PDF)
    # Capture any "X is/are defined as Y" or "We define X as Y" patterns
    # ACCUMULATE with other strategies
    initial_count = len(pairs)
    broad_def_patterns = [
        r'([A-Z][a-zA-Z\s]+?)\s+(?:is|are)\s+defined\s+(?:as|to be)\s+(.+?)(?:\.|;|\n\n)',
        r'(?:We|Let us)\s+define\s+([a-zA-Z\s]+?)\s+as\s+(.+?)(?:\.|;|\n\n)',
        r'Definition[:\s]+([A-Z][a-zA-Z\s]+?)[:\s]+(.+?)(?:\n\n|$)',
    ]

    for pattern in broad_def_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            term = match.group(1).strip()
            definition = match.group(2).strip()

            if len(term) < 3 or len(term) > 100:
                continue
            if len(definition) < 30 or len(definition) > 1000:
                continue

            question = f"What is {term}?"
            solution = clean_text(definition)

            pairs.append({
                "question": question,
                "solution": solution
            })

    if len(pairs) > initial_count:
        print(f"    Extracted {len(pairs) - initial_count} definitions from PDF")

    # Strategy 7: Extract theorem/lemma/corollary statements (even without proofs)
    # These can be used as "State the theorem" type questions
    # ACCUMULATE with other strategies
    initial_count = len(pairs)
    theorem_only_pattern = r'(Theorem|Lemma|Corollary|Proposition)\s*\d*[.:]\s*(.+?)(?=\n\n|Proof|$)'
    matches = re.finditer(theorem_only_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        thm_type = match.group(1).strip()
        statement = match.group(2).strip()

        # Skip if too short (likely continuation of previous text)
        if len(statement) < 50 or len(statement) > 3000:
            continue

        # Clean the statement
        statement = clean_text(statement)

        # If statement is actually long enough to contain explanation
        if len(statement) > 150:
            question = f"State the {thm_type}."
            solution = statement

            pairs.append({
                "question": question,
                "solution": solution
            })

    if len(pairs) > initial_count:
        print(f"    Extracted {len(pairs) - initial_count} theorems/lemmas from PDF")

    # Strategy 8: Extract algorithm pseudocode blocks (from any PDF)
    # Look for "Algorithm:", pseudocode blocks, or "Input:/Output:" patterns
    # ACCUMULATE with other strategies
    initial_count = len(pairs)

    # Pattern 8a: Algorithm with Input/Output/Steps
    algorithm_blocks = re.finditer(
        r'(?:Algorithm|Procedure)[:\s]+([A-Z][a-zA-Z\s\-()]+?)[:\s]*\n'
        r'(?:.*?(?:Input|Output|Steps|begin|for|while|if|return).*?)+'
        r'(?=\n\n|Algorithm|Theorem|Definition|Example|$)',
        text, re.DOTALL | re.IGNORECASE
    )

    for match in algorithm_blocks:
        alg_name = match.group(1).strip()
        alg_body = match.group(0).strip()

        if len(alg_body) < 100 or len(alg_body) > 5000:
            continue

        question = f"Describe the {alg_name} algorithm."
        solution = clean_text(alg_body)

        pairs.append({
            "question": question,
            "solution": solution
        })

    # Pattern 8b: Pseudocode blocks (for/while/if statements with indentation)
    code_blocks = re.finditer(
        r'(?:^|\n)((?:for|while|if|function|procedure)\s+.+?(?:\n(?:[ \t]+.+?))+)',
        text, re.MULTILINE | re.IGNORECASE
    )

    for match in code_blocks:
        code_text = match.group(1).strip()

        if len(code_text) < 80 or len(code_text) > 3000:
            continue

        # Extract what it does from context (previous line or heading)
        context_start = max(0, match.start() - 200)
        context = text[context_start:match.start()]

        # Look for heading or topic in context
        topic_match = re.search(r'([A-Z][a-zA-Z\s\-]+?)(?:\n|:)', context)
        topic = topic_match.group(1).strip() if topic_match else "this algorithm"

        question = f"Show the pseudocode for {topic}."
        solution = clean_text(code_text)

        pairs.append({
            "question": question,
            "solution": solution
        })

    if len(pairs) > initial_count:
        print(f"    Extracted {len(pairs) - initial_count} algorithms/pseudocode from PDF")

    # Strategy 9: Extract "Properties:" and "Characteristics:" sections
    # ACCUMULATE with other strategies
    initial_count = len(pairs)

    properties_pattern = r'(?:Properties|Characteristics)[:\s]+(?:of\s+)?([A-Z][a-zA-Z\s]+?)[:\s]*\n(.+?)(?=\n\n|Properties|Theorem|Definition|Algorithm|$)'
    matches = re.finditer(properties_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        topic = match.group(1).strip() if match.lastindex >= 1 else "this concept"
        properties = match.group(2).strip() if match.lastindex >= 2 else match.group(1).strip()

        if len(properties) < 50 or len(properties) > 2000:
            continue

        question = f"What are the properties of {topic}?"
        solution = clean_text(properties)

        pairs.append({
            "question": question,
            "solution": solution
        })

    if len(pairs) > initial_count:
        print(f"    Extracted {len(pairs) - initial_count} properties/characteristics from PDF")

    print(f"    Extracted {len(pairs)} Q&A pairs from PDF")
    return pairs


def parse_text_file(file_path: Path) -> List[Dict[str, str]]:
    """
    Parse text file with Q: and A: separators.
    
    Expected format:
    Q: question text
    A: answer text
    
    or variations like Question: / Answer:
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pairs = []
    # Match Q: ... A: ... pattern
    # Also handle Question: / Answer: variations
    pattern = r'(?:Q:|Question:)\s*(.+?)(?=\n(?:A:|Answer:))(?:\n(?:A:|Answer:)\s*(.+?))(?=\n(?:Q:|Question:)|$)'
    
    matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
    for match in matches:
        question = match.group(1).strip()
        answer = match.group(2).strip() if match.lastindex >= 2 else ""
        
        if question and answer:
            pairs.append({"question": question, "solution": answer})
    
    # Fallback: try simpler pattern if above doesn't match
    if not pairs:
        # Split by lines and look for Q:/A: patterns
        lines = content.split('\n')
        current_q = None
        current_a = []
        
        for line in lines:
            if re.match(r'^(?:Q:|Question:)\s*', line, re.IGNORECASE):
                if current_q and current_a:
                    pairs.append({
                        "question": current_q,
                        "solution": '\n'.join(current_a).strip()
                    })
                current_q = re.sub(r'^(?:Q:|Question:)\s*', '', line, flags=re.IGNORECASE).strip()
                current_a = []
            elif re.match(r'^(?:A:|Answer:)\s*', line, re.IGNORECASE):
                answer_line = re.sub(r'^(?:A:|Answer:)\s*', '', line, flags=re.IGNORECASE).strip()
                if answer_line:
                    current_a.append(answer_line)
            elif current_a:  # Continuation of answer
                current_a.append(line)
            elif current_q and not current_a:  # Continuation of question
                current_q += '\n' + line
        
        # Add last pair
        if current_q and current_a:
            pairs.append({
                "question": current_q,
                "solution": '\n'.join(current_a).strip()
            })
    
    return pairs


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove slide date stamps (e.g., "August23,2025 1/16")
    text = re.sub(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\d{1,2},\s*\d{4}\s+\d+/\d+', '', text)

    # Remove standalone page numbers (e.g., "1/16", "2/14")
    text = re.sub(r'\b\d+/\d+\b', '', text)

    # Remove slide header dates without page numbers (e.g., "August 23, 2025")
    text = re.sub(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}\b', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(text: str, tokenizer: AutoTokenizer, max_tokens: int = MAX_TOKENS) -> List[str]:
    """
    Split text into chunks of at most max_tokens.
    
    Tries to split at sentence boundaries when possible.
    """
    # Tokenize to count tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    # Split by sentences first
    sentences = re.split(r'([.!?]\s+)', text)
    sentence_pairs = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence_pairs.append(sentences[i] + sentences[i + 1])
        else:
            sentence_pairs.append(sentences[i])
    
    for sentence in sentence_pairs:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)
        
        if current_token_count + sentence_token_count > max_tokens:
            if current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = []
                current_token_count = 0
            
            # If single sentence is too long, split by words
            if sentence_token_count > max_tokens:
                words = sentence.split()
                for word in words:
                    word_tokens = tokenizer.encode(word, add_special_tokens=False)
                    word_token_count = len(word_tokens)
                    
                    if current_token_count + word_token_count > max_tokens:
                        if current_chunk:
                            chunks.append(''.join(current_chunk))
                            current_chunk = []
                            current_token_count = 0
                    
                    current_chunk.append(word + ' ')
                    current_token_count += word_token_count
            else:
                current_chunk.append(sentence)
                current_token_count += sentence_token_count
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks


def process_qa_pair(qa_pair: Dict[str, str], tokenizer: AutoTokenizer, max_chars: int = 20000) -> List[Dict[str, str]]:
    """
    Process a Q&A pair, with validation and length limits.

    Returns a list with a single processed pair if valid, empty list otherwise.
    """
    question = clean_text(qa_pair.get("question", ""))
    solution = clean_text(qa_pair.get("solution", ""))

    # Require both question and solution
    if not question or not solution:
        return []

    # Safety check: reject pairs that are unreasonably long (likely entire documents)
    if len(question) > max_chars or len(solution) > max_chars:
        print(f"      Warning: Skipping oversized Q&A pair (question: {len(question)} chars, solution: {len(solution)} chars)")
        return []

    # Minimum length check: require substantive content
    if len(question) < 20 or len(solution) < 20:
        return []

    # QUALITY FILTERS: VERY RELAXED for maximum data extraction
    # 1. Question should start with capital letter or digit (auto-capitalize if needed)
    if question[0].islower():
        # Auto-capitalize instead of rejecting
        question = question[0].upper() + question[1:]

    # 2. Solution should have some content (VERY RELAXED: 30 chars minimum)
    if len(solution) < 30:
        return []

    # 3. Question should look somewhat complete (VERY RELAXED)
    # Accept if: ends with punctuation, starts with common question words, or is reasonably long
    is_complete = (
        question.endswith(('?', '.', ':', ';')) or
        question.startswith(('What', 'How', 'Why', 'Explain', 'Prove', 'Show', 'Define', 'State', 'Describe', 'Give', 'Find', 'Consider', 'Let', 'Suppose')) or
        len(question) > 100  # Further reduced from 200
    )
    if not is_complete:
        return []

    # Check token count
    combined = f"Question: {question}\nSolution: {solution}"
    tokens = tokenizer.encode(combined, add_special_tokens=False)

    if len(tokens) > MAX_TOKENS:
        # If too long, truncate rather than chunk (avoids duplication)
        print(f"      Warning: Q&A pair too long ({len(tokens)} tokens), truncating to {MAX_TOKENS}")
        # Truncate by taking a reasonable portion of each
        max_q_chars = max_chars // 2
        max_s_chars = max_chars // 2
        question = question[:max_q_chars]
        solution = solution[:max_s_chars]

    return [{
        "text": combined,
        "question": question,
        "solution": solution
    }]


def extract_unit_from_filename(filename: str) -> Optional[int]:
    """
    Extract unit/chapter number from filename using common patterns.
    
    Patterns supported:
    - unit_1, unit1, unit-1, unit_01
    - chapter_1, chapter1, chapter-1, chapter_01
    - ch_1, ch1, ch-1, ch_01
    - u1, u_1, u-1
    - exam_unit1, exam_unit_1, exam_chapter1
    - midterm_unit1, final_chapter2
    
    Returns:
        int unit number if found, None otherwise
    """
    filename_lower = filename.lower()
    
    # Patterns to match unit/chapter numbers
    patterns = [
        r'unit[-_]?(\d+)',           # unit_1, unit1, unit-1
        r'chapter[-_]?(\d+)',        # chapter_1, chapter1, chapter-1
        r'ch[-_]?(\d+)',             # ch_1, ch1, ch-1
        r'\bu[-_]?(\d+)\b',          # u1, u_1, u-1 (word boundary to avoid matching in other words)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date from filename using common patterns.
    
    Patterns supported:
    - YYYY-MM-DD, YYYY_MM_DD
    - YYYYMMDD
    - YYYY-MM, YYYY_MM
    - YYYY (year only)
    - YYYY_fall, YYYY_spring, YYYY_summer, YYYY_winter
    - fall_YYYY, spring_YYYY, etc.
    
    Returns:
        datetime object if date found, None otherwise
    """
    filename_lower = filename.lower()
    
    # Pattern 1: YYYY-MM-DD or YYYY_MM_DD
    date_patterns = [
        (r'(\d{4})[-_](\d{1,2})[-_](\d{1,2})', '%Y-%m-%d'),  # Full date
        (r'(\d{4})(\d{2})(\d{2})', '%Y%m%d'),  # YYYYMMDD
        (r'(\d{4})[-_](\d{1,2})', '%Y-%m'),  # YYYY-MM
        (r'(\d{4})', '%Y'),  # Year only
    ]
    
    for pattern, date_format in date_patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                date_str = match.group(0).replace('_', '-')
                if date_format == '%Y':
                    return datetime.strptime(date_str, date_format)
                elif date_format == '%Y-%m':
                    return datetime.strptime(date_str, date_format)
                elif date_format == '%Y%m%d':
                    return datetime.strptime(date_str, date_format)
                else:
                    # Normalize separators
                    parts = re.split(r'[-_]', date_str)
                    if len(parts) == 3:
                        return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                    elif len(parts) == 2:
                        return datetime(int(parts[0]), int(parts[1]), 1)
            except (ValueError, IndexError):
                continue
    
    # Pattern 2: Season + year (fall_2023, 2023_fall, etc.)
    season_patterns = [
        (r'(?:fall|autumn)[-_]?(\d{4})', 9),  # fall_2023 -> September 2023
        (r'(\d{4})[-_]?(?:fall|autumn)', 9),
        (r'(?:spring)[-_]?(\d{4})', 3),  # spring_2023 -> March 2023
        (r'(\d{4})[-_]?(?:spring)', 3),
        (r'(?:summer)[-_]?(\d{4})', 6),  # summer_2023 -> June 2023
        (r'(\d{4})[-_]?(?:summer)', 6),
        (r'(?:winter)[-_]?(\d{4})', 12),  # winter_2023 -> December 2023
        (r'(\d{4})[-_]?(?:winter)', 12),
    ]
    
    for pattern, month in season_patterns:
        match = re.search(pattern, filename_lower)
        if match:
            year = int(match.group(1)) if match.lastindex >= 1 else int(re.search(r'\d{4}', filename).group())
            return datetime(year, month, 1)
    
    return None


def load_raw_files(
    data_dir: Path,
    validation_split: float = 0.15,
    test_split: float = 0.15,
    use_temporal_split: bool = True,
    use_unit_split: bool = False
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Load all .txt, .json, and .pdf files from data directory and split into train/validation/test.
    
    Supports two modes:
    1. Manual directory organization: If subdirectories `train/`, `validation/`, `test/` exist,
       files are loaded from those directories directly.
    2. Automatic splitting: If no subdirectories exist, automatically splits files based on
       filename patterns and ordering (temporal or unit-based).
    
    Automatic splitting behavior:
    - If use_unit_split=True: Earlier units → training/validation, later units → test
    - If use_temporal_split=True: Past exams/quizzes → validation, future exams/quizzes → test
    - All other files → training set
    
    Args:
        data_dir: Directory containing raw data files (or subdirectories train/validation/test)
        validation_split: Fraction of exam files to use for validation (default: 0.15)
        test_split: Fraction of exam files to use for test (default: 0.15)
        use_temporal_split: If True, use date-based temporal splitting for exams
        use_unit_split: If True, use unit-based splitting (earlier units → train/val, later → test)
    
    Returns:
        Tuple of (train_files, validation_files, test_files)
    """
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Check for manual directory organization
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    test_dir = data_dir / "test"
    
    # If at least one subdirectory exists, assume manual organization
    if train_dir.exists() or validation_dir.exists() or test_dir.exists():
        print("Detected manual directory organization. Loading from subdirectories...")
        train_files = []
        validation_files = []
        test_files = []
        
        # Include PDF files if PDF processing is available
        extensions = ['*.txt', '*.json']
        if PDF_AVAILABLE:
            extensions.append('*.pdf')
        
        # Load from train directory
        if train_dir.exists():
            for ext in extensions:
                train_files.extend(train_dir.glob(ext))
                train_files.extend(train_dir.glob(ext.upper()))
            print(f"  Found {len(train_files)} files in {train_dir}")
        else:
            print(f"  Warning: {train_dir} does not exist. No training files will be loaded.")
        
        # Load from validation directory
        if validation_dir.exists():
            for ext in extensions:
                validation_files.extend(validation_dir.glob(ext))
                validation_files.extend(validation_dir.glob(ext.upper()))
            print(f"  Found {len(validation_files)} files in {validation_dir}")
        else:
            print(f"  Note: {validation_dir} does not exist. No validation files will be loaded.")
        
        # Load from test directory
        if test_dir.exists():
            for ext in extensions:
                test_files.extend(test_dir.glob(ext))
                test_files.extend(test_dir.glob(ext.upper()))
            print(f"  Found {len(test_files)} files in {test_dir}")
        else:
            print(f"  Warning: {test_dir} does not exist. No test files will be loaded.")
        
        return train_files, validation_files, test_files
    
    # Automatic splitting mode: load all files from data_dir and split automatically
    print("No subdirectories detected. Using automatic splitting based on filename patterns...")
    
    all_files = []
    # Include PDF files if PDF processing is available
    extensions = ['*.txt', '*.json']
    if PDF_AVAILABLE:
        extensions.append('*.pdf')
    
    for ext in extensions:
        all_files.extend(data_dir.glob(ext))
        all_files.extend(data_dir.glob(ext.upper()))
    
    train_files = []
    exam_files = []
    validation_files = []
    test_files = []
    
    # Extract dates and units for all files
    all_file_info = []
    for file_path in all_files:
        filename_lower = file_path.name.lower()
        is_exam = any(keyword in filename_lower for keyword in ['exam', 'test', 'quiz', 'final'])
        date = extract_date_from_filename(file_path.name)
        unit = extract_unit_from_filename(file_path.name)
        
        all_file_info.append(FileInfo(
            path=file_path,
            date=date,
            unit=unit,
            is_exam=is_exam,
            filename=file_path.name
        ))
    
    # Separate exams from non-exams for non-unit-based splits
    for file_info in all_file_info:
        if file_info.is_exam:
            exam_files.append(file_info.path)
        else:
            train_files.append(file_info.path)
    
    # Split files based on strategy
    if use_unit_split:
            # Unit-based splitting: earlier units → train/validation, later units → test
            print("Using unit-based splitting: earlier units → train/validation, later units → test")
            
            # Reset train_files since we're re-splitting by unit
            train_files = []
            
            # Sort all files by unit (None units go to the end)
            all_file_info.sort(key=lambda x: (x.unit is None, x.unit or float('inf')))
            
            # Find files with units
            files_with_units = [x for x in all_file_info if x.unit is not None]
            files_without_units = [x for x in all_file_info if x.unit is None]
            
            if files_with_units:
                # Determine unit threshold: earlier units for train/val, later for test
                num_with_units = len(files_with_units)
                # Use last 15-20% of units for test
                unit_threshold_idx = max(1, int(num_with_units * (1 - test_split)))
                unit_threshold = files_with_units[unit_threshold_idx].unit
                
                print(f"  Unit threshold: Units 1-{unit_threshold-1} → train/validation, Units {unit_threshold}+ → test")
                
                # Split files by unit
                for file_info in files_with_units:
                    if file_info.unit < unit_threshold:
                        if file_info.is_exam:
                            # Earlier unit exams → validation
                            validation_files.append(file_info.path)
                        else:
                            # Earlier unit non-exams → training
                            train_files.append(file_info.path)
                    else:
                        # Later unit files → test (both exams and non-exams)
                        test_files.append(file_info.path)
                
                # Files without units: add to training (can't determine unit)
                for file_info in files_without_units:
                    if not file_info.is_exam:
                        train_files.append(file_info.path)
                    else:
                        # Exams without units: add to validation (conservative)
                        validation_files.append(file_info.path)
            else:
                # No units found, fall back to temporal or simple split
                print("  Warning: No unit numbers found in filenames. Falling back to temporal/simple split.")
                if use_temporal_split:
                    # Fall through to temporal split logic below
                    pass
                else:
                    # Simple split
                    num_exams = len(exam_files)
                    num_validation = max(1, int(num_exams * validation_split))
                    num_test = max(1, int(num_exams * test_split))
                    validation_files = exam_files[:num_validation]
                    test_files = exam_files[-num_test:] if num_validation + num_test < num_exams else exam_files[num_validation:]
    
    elif use_temporal_split and exam_files:
        # Temporal splitting: earlier dates → validation, later dates → test
        print("Using temporal splitting: earlier exams → validation, later exams → test")
        
        # Extract dates and sort exams by date
        exam_info = []
        for file_path in exam_files:
            date = extract_date_from_filename(file_path.name)
            unit = extract_unit_from_filename(file_path.name)
            exam_info.append(FileInfo(
                path=file_path,
                date=date,
                unit=unit,
                is_exam=True,
                filename=file_path.name
            ))
        
        # Sort by date (None dates go to the end)
        exam_info.sort(key=lambda x: (x.date is None, x.date or datetime.min))
        
        # Split: earlier exams → validation, later exams → test
        num_exams = len(exam_info)
        num_validation = max(1, int(num_exams * validation_split))
        num_test = max(1, int(num_exams * test_split))
        
        # If we have dates, use temporal split
        dated_exams = [x for x in exam_info if x.date is not None]
        undated_exams = [x for x in exam_info if x.date is None]
        
        if dated_exams:
            # Split dated exams temporally
            num_dated = len(dated_exams)
            num_val_dated = max(1, int(num_dated * validation_split))
            num_test_dated = max(1, int(num_dated * test_split))
            
            # Earlier exams → validation
            validation_files.extend([x.path for x in dated_exams[:num_val_dated]])
            # Later exams → test
            test_files.extend([x.path for x in dated_exams[-num_test_dated:]])
            
            # Remaining dated exams go to validation (middle ones)
            remaining = dated_exams[num_val_dated:-num_test_dated] if num_val_dated + num_test_dated < num_dated else []
            validation_files.extend([x.path for x in remaining])
        else:
            # No dates found, use simple split
            validation_files.extend([x.path for x in exam_info[:num_validation]])
            test_files.extend([x.path for x in exam_info[-num_test:]])
        
        # Undated exams: split evenly or all to test
        if undated_exams:
            num_undated = len(undated_exams)
            num_val_undated = max(1, int(num_undated * validation_split))
            validation_files.extend([x.path for x in undated_exams[:num_val_undated]])
            test_files.extend([x.path for x in undated_exams[num_val_undated:]])
    elif exam_files:
        # No splitting strategy: split exams evenly
        num_exams = len(exam_files)
        num_validation = max(1, int(num_exams * validation_split))
        num_test = max(1, int(num_exams * test_split))
        
        validation_files = exam_files[:num_validation]
        test_files = exam_files[-num_test:] if num_validation + num_test < num_exams else exam_files[num_validation:]
    
    return train_files, validation_files, test_files


def save_jsonl(data: List[Dict[str, Any]], output_path: Path):
    """Save data to JSONL format (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_files(files: List[Path], tokenizer: AutoTokenizer, set_name: str, is_test_set: bool = False) -> List[Dict[str, Any]]:
    """
    Process a list of files and extract Q&A pairs.

    Args:
        files: List of file paths to process
        tokenizer: Tokenizer for token counting
        set_name: Name of the dataset split (train/validation/test)
        is_test_set: If True, process as test set (questions only, no validation)

    Returns:
        List of processed examples
    """
    data = []
    print(f"\nProcessing {set_name} files...")
    for file_path in files:
        print(f"  Processing: {file_path.name}")
        try:
            file_ext = file_path.suffix.lower()
            if file_ext == '.json':
                qa_pairs = parse_json_file(file_path, allow_question_only=is_test_set)
            elif file_ext == '.pdf':
                if not PDF_AVAILABLE:
                    print(f"    Skipping PDF (pdfplumber not installed): {file_path.name}")
                    continue
                qa_pairs = parse_pdf_file(file_path)
            else:
                qa_pairs = parse_text_file(file_path)

            # For test set, only keep questions (no solution validation)
            if is_test_set:
                for qa_pair in qa_pairs:
                    question = clean_text(qa_pair.get("question", ""))
                    if question:
                        # Test set: questions only, no solutions
                        data.append({
                            "question": question
                        })
            else:
                # For train/validation: full validation and processing
                for qa_pair in qa_pairs:
                    processed = process_qa_pair(qa_pair, tokenizer)
                    data.extend(processed)
        except Exception as e:
            print(f"    Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return data


def main(
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
    validation_split: float = 0.15,
    test_split: float = 0.15,
    use_temporal_split: bool = True,
    use_unit_split: bool = False
):
    """
    Main preprocessing function.
    
    Args:
        raw_data_dir: Path to raw data directory (relative to project root or absolute)
        processed_data_dir: Path to processed data output directory (relative to project root or absolute)
        validation_split: Fraction of exam files for validation (default: 0.15)
        test_split: Fraction of exam files for test (default: 0.15)
        use_temporal_split: Use date-based temporal splitting (default: True)
        use_unit_split: Use unit-based splitting - earlier units → train/val, later → test (default: False)
    """
    # Resolve paths - convert to Path objects and resolve to absolute paths
    raw_dir = Path(raw_data_dir)
    processed_dir = Path(processed_data_dir)
    
    # If path is relative and doesn't exist, try relative to script's parent directory (project root)
    if not raw_dir.is_absolute():
        if not raw_dir.exists():
            # Try relative to script location (go up from preprocess/load_and_prepare.py to project root)
            script_dir = Path(__file__).parent.parent  # Go up to project root
            potential_path = script_dir / raw_data_dir
            if potential_path.exists():
                raw_dir = potential_path.resolve()
            else:
                # Try relative to current working directory
                raw_dir = raw_dir.resolve()
        else:
            raw_dir = raw_dir.resolve()
    else:
        raw_dir = raw_dir.resolve()

    # For processed directory, resolve to absolute path
    if not processed_dir.is_absolute():
        script_dir = Path(__file__).parent.parent  # Go up to project root
        potential_path = script_dir / processed_data_dir
        processed_dir = potential_path.resolve()
    else:
        processed_dir = processed_dir.resolve()
    
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = load_tokenizer()
    
    print(f"Scanning for files in: {raw_dir}")
    train_files, validation_files, test_files = load_raw_files(
        raw_dir,
        validation_split=validation_split,
        test_split=test_split,
        use_temporal_split=use_temporal_split,
        use_unit_split=use_unit_split
    )
    
    print(f"Found {len(train_files)} training files, {len(validation_files)} validation files, and {len(test_files)} test files")
    
    if validation_files:
        print(f"  Validation files (past exams): {[f.name for f in validation_files]}")
    if test_files:
        print(f"  Test files (future exams): {[f.name for f in test_files]}")
    
    # Process all file sets
    train_data = process_files(train_files, tokenizer, "training", is_test_set=False)
    validation_data = process_files(validation_files, tokenizer, "validation", is_test_set=False)
    test_data = process_files(test_files, tokenizer, "test", is_test_set=True)  # Test set: questions only
    
    # Save processed data
    print(f"\nSaving processed data to: {processed_dir}")
    save_jsonl(train_data, processed_dir / "train.jsonl")
    if validation_data:
        save_jsonl(validation_data, processed_dir / "validation.jsonl")
    save_jsonl(test_data, processed_dir / "test.jsonl")
    
    print(f"\n[OK] Processed {len(train_data)} training examples")
    if validation_data:
        print(f"[OK] Processed {len(validation_data)} validation examples")
    print(f"[OK] Processed {len(test_data)} test examples")
    print(f"[OK] Saved to {processed_dir / 'train.jsonl'}")
    if validation_data:
        print(f"[OK] Saved to {processed_dir / 'validation.jsonl'}")
    print(f"[OK] Saved to {processed_dir / 'test.jsonl'}")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess data for fine-tuning")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--validation_split", type=float, default=0.15, help="Fraction of exams for validation")
    parser.add_argument("--test_split", type=float, default=0.15, help="Fraction of exams for test")
    parser.add_argument("--no_temporal_split", action="store_true", help="Disable temporal splitting")
    parser.add_argument("--use_unit_split", action="store_true", help="Use unit-based splitting (earlier units → train/val, later → test)")
    
    args = parser.parse_args()
    
    main(
        raw_data_dir=args.raw_dir,
        processed_data_dir=args.processed_dir,
        validation_split=args.validation_split,
        test_split=args.test_split,
        use_temporal_split=not args.no_temporal_split,
        use_unit_split=args.use_unit_split
    )

