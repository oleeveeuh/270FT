#!/usr/bin/env python3
"""
Improved extraction script for creating a comprehensive test set with clear labeling.

This script processes all available data sources to extract questions and solutions,
ensuring all 24 questions are properly extracted with clear labeling like "Question 1", "Problem 1", etc.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import subprocess
import sys

# PDF processing
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")

def clean_text(text: str) -> str:
    """Clean and normalize text by removing excessive whitespace and artifacts."""
    # Remove slide date stamps and page numbers
    text = re.sub(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}\b', '', text)
    text = re.sub(r'\b\d+/\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
    if not PDF_AVAILABLE:
        raise ImportError("pdfplumber is required for PDF processing")

    text_content = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
    except Exception as e:
        print(f"    Error extracting PDF text: {e}")
        return ""

    return '\n\n'.join(text_content)

def extract_homework_questions(pdf_path: Path) -> List[Dict[str, str]]:
    """
    Extract questions and solutions from homework PDFs with improved patterns.
    """
    print(f"    Processing homework PDF: {pdf_path.name}")
    text = extract_text_from_pdf(pdf_path)

    if not text:
        return []

    questions = []

    # Pattern 1: Problem X followed by Solution
    problem_solution_pattern = r'Problem\s+(\d+)[.:]?\s*(.+?)(?=Solution\s+\d*[.:]?\s*|$)(?:Solution\s+\d*[.:]?\s*(.+?))(?=Problem\s+\d+|$)'

    matches = re.finditer(problem_solution_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        prob_num = match.group(1).strip()
        prob_text = match.group(2).strip() if match.lastindex >= 2 else ""
        sol_text = match.group(3).strip() if match.lastindex >= 3 else ""

        # Clean and validate
        prob_text = clean_text(prob_text)
        sol_text = clean_text(sol_text)

        if len(prob_text) > 50 and len(sol_text) > 50:
            questions.append({
                "question": f"Problem {prob_num}: {prob_text}",
                "solution": sol_text
            })

    # Pattern 2: Exercise X with Solution
    if not questions:
        exercise_pattern = r'Exercise\s+(\d+)[.:]?\s*(.+?)(?=Solution\s+\d*[.:]?\s*|$)(?:Solution\s+\d*[.:]?\s*(.+?))(?=Exercise\s+\d+|$)'

        matches = re.finditer(exercise_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            ex_num = match.group(1).strip()
            ex_text = match.group(2).strip() if match.lastindex >= 2 else ""
            sol_text = match.group(3).strip() if match.lastindex >= 3 else ""

            ex_text = clean_text(ex_text)
            sol_text = clean_text(sol_text)

            if len(ex_text) > 50 and len(sol_text) > 50:
                questions.append({
                    "question": f"Exercise {ex_num}: {ex_text}",
                    "solution": sol_text
                })

    # Pattern 3: Numbered questions without explicit "Problem" label
    if not questions:
        numbered_pattern = r'^(\d+)\.[\s]*(.+?)(?=\n\s*\d+\.|\n\nSolution|$)(?:\n\nSolution[:\s]*\n(.+?))?(?=\n\s*\d+\.|\n\n|\Z)'

        matches = re.finditer(numbered_pattern, text, re.MULTILINE | re.DOTALL)
        for match in matches:
            q_num = match.group(1).strip()
            q_text = match.group(2).strip() if match.lastindex >= 2 else ""
            sol_text = match.group(3).strip() if match.lastindex >= 3 else ""

            q_text = clean_text(q_text)
            sol_text = clean_text(sol_text)

            if len(q_text) > 50:
                if sol_text and len(sol_text) > 30:
                    questions.append({
                        "question": f"Question {q_num}: {q_text}",
                        "solution": sol_text
                    })
                else:
                    # Question without solution
                    questions.append({
                        "question": f"Question {q_num}: {q_text}"
                    })

    print(f"    Extracted {len(questions)} questions from {pdf_path.name}")
    return questions

def process_test_json(json_path: Path) -> List[Dict[str, str]]:
    """Process the existing test.json file and add clear labeling."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labeled_questions = []
    for i, item in enumerate(data, 1):
        question = item.get('question', '').strip()
        if question:
            labeled_questions.append({
                "question": f"Question {i}: {question}",
                "solution": item.get('solution', '') if 'solution' in item else ''
            })

    return labeled_questions

def merge_questions(all_questions: List[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Merge questions from all sources, removing duplicates and ensuring proper labeling."""
    merged = []
    seen_questions = set()

    question_counter = 1

    for source_questions in all_questions:
        for item in source_questions:
            question_text = item.get('question', '')

            # Simple duplicate detection
            question_key = re.sub(r'^(Question|Problem|Exercise)\s+\d+:\s*', '', question_text).strip().lower()
            if question_key in seen_questions:
                continue

            seen_questions.add(question_key)

            # Ensure consistent labeling
            if not re.match(r'^(Question|Problem|Exercise)\s+\d+:', question_text):
                labeled_question = f"Question {question_counter}: {question_text}"
            else:
                labeled_question = question_text

            merged_item = {"question": labeled_question}
            if 'solution' in item and item['solution']:
                merged_item["solution"] = item['solution']

            merged.append(merged_item)
            question_counter += 1

    return merged

def create_comprehensive_test_set(raw_data_dir: Path, output_path: Path) -> int:
    """
    Create a comprehensive test set by processing all available data sources.

    Args:
        raw_data_dir: Directory containing raw data
        output_path: Path to save the processed test set

    Returns:
        Number of questions extracted
    """
    print("Creating comprehensive test set...")

    all_questions = []

    # 1. Process existing test.json
    test_json_path = raw_data_dir / "test" / "test.json"
    if test_json_path.exists():
        print("Processing existing test.json...")
        test_questions = process_test_json(test_json_path)
        all_questions.append(test_questions)
        print(f"  Found {len(test_questions)} questions in test.json")

    # 2. Process homework solution PDFs from validation directory
    validation_dir = raw_data_dir / "validation"
    if validation_dir.exists():
        print("Processing homework solution PDFs...")
        pdf_questions = []

        for pdf_file in validation_dir.glob("*.pdf"):
            if "homework" in pdf_file.name.lower() or "solution" in pdf_file.name.lower():
                homework_questions = extract_homework_questions(pdf_file)
                pdf_questions.extend(homework_questions)

        if pdf_questions:
            all_questions.append(pdf_questions)
            print(f"  Found {len(pdf_questions)} questions in homework PDFs")

    # 3. Process PDF files from the root directory
    homework_pdfs = list(raw_data_dir.glob("*Homework*.pdf"))
    if homework_pdfs:
        print("Processing homework PDFs from root directory...")
        root_pdf_questions = []
        for pdf_file in homework_pdfs:
            homework_questions = extract_homework_questions(pdf_file)
            root_pdf_questions.extend(homework_questions)

        if root_pdf_questions:
            all_questions.append(root_pdf_questions)
            print(f"  Found {len(root_pdf_questions)} questions in root homework PDFs")

    # Merge and deduplicate
    print("Merging and deduplicating questions...")
    final_questions = merge_questions(all_questions)

    # Save to output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in final_questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')

    print(f"Saved {len(final_questions)} questions to {output_path}")

    # Print summary
    questions_with_solutions = sum(1 for q in final_questions if 'solution' in q and q['solution'])
    questions_without_solutions = len(final_questions) - questions_with_solutions

    print(f"\nSummary:")
    print(f"  Total questions: {len(final_questions)}")
    print(f"  With solutions: {questions_with_solutions}")
    print(f"  Without solutions: {questions_without_solutions}")

    return len(final_questions)

def create_labeled_test_with_solutions(test_path: Path, solutions_path: Path) -> None:
    """
    Create a version of the test set with reference solutions for evaluation.
    """
    questions = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Add empty solutions for questions that don't have them
    for question in questions:
        if 'solution' not in question:
            question['solution'] = ''

    # Save with solutions
    with open(solutions_path, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')

    print(f"Created test set with solutions: {solutions_path}")

def main():
    """Main function to process and create the improved test set."""
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    print(f"Project root: {project_root}")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Processed directory: {processed_dir}")

    # Create comprehensive test set
    test_output_path = processed_dir / "test_comprehensive.jsonl"
    num_questions = create_comprehensive_test_set(raw_data_dir, test_output_path)

    # Create version with solutions for evaluation
    solutions_path = processed_dir / "test_with_solutions.jsonl"
    create_labeled_test_with_solutions(test_output_path, solutions_path)

    # Update the original test.jsonl as backup
    original_test_path = processed_dir / "test.jsonl"
    backup_path = processed_dir / "test_original_backup.jsonl"
    if original_test_path.exists():
        import shutil
        shutil.copy2(original_test_path, backup_path)
        print(f"Backed up original test.jsonl to {backup_path}")

    # Replace original with comprehensive version
    import shutil
    shutil.copy2(test_output_path, original_test_path)
    print(f"Updated original test.jsonl with {num_questions} questions")

    print(f"\n✅ Successfully processed test set!")
    print(f"   - Total questions extracted: {num_questions}")
    print(f"   - Clear labeling: 'Question 1:', 'Problem 2:', etc.")
    print(f"   - Solutions included where available")
    print(f"   - Files saved:")
    print(f"     * {original_test_path} (main test set)")
    print(f"     * {solutions_path} (with solutions for evaluation)")
    print(f"     * {test_output_path} (comprehensive version)")

if __name__ == "__main__":
    main()