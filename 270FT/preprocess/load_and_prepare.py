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

# File info for temporal sorting
FileInfo = namedtuple('FileInfo', ['path', 'date', 'is_exam', 'filename'])


def load_tokenizer() -> AutoTokenizer:
    """Load tokenizer for counting tokens."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def parse_json_file(file_path: Path) -> List[Dict[str, str]]:
    """
    Parse JSON file with Question/Solution blocks.
    
    Expected format:
    {
        "Question": "...",
        "Solution": "..."
    }
    or a list of such objects.
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
    elif isinstance(data, dict):
        question = data.get("Question", data.get("question", ""))
        solution = data.get("Solution", data.get("solution", ""))
        if question and solution:
            pairs.append({"question": question, "solution": solution})
    
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
    
    pairs = []
    
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
            
            if question and answer and len(question) > 10 and len(answer) > 10:
                pairs.append({
                    "question": clean_text(question),
                    "solution": clean_text(answer)
                })
    
    # Strategy 2: For lecture slides, extract slide content as context
    # Look for slide separators or numbered sections
    if not pairs:
        # Try to identify slide-based content
        slides = re.split(r'\n\s*\n\s*\n|\n\d+\s*\n', text)  # Split by multiple newlines or slide numbers
        
        for slide in slides:
            slide = slide.strip()
            if len(slide) < 50:  # Skip very short slides (likely headers)
                continue
            
            # Look for problem statements followed by solutions
            # Common patterns in slides: "Problem:", "Example:", "Exercise:"
            problem_match = re.search(
                r'(?:Problem|Example|Exercise|Question)[:\s]+(.+?)(?=\n(?:Solution|Answer|Proof)|$)',
                slide,
                re.DOTALL | re.IGNORECASE
            )
            
            if problem_match:
                problem_text = problem_match.group(1).strip()
                
                # Look for solution in the same slide or next
                solution_match = re.search(
                    r'(?:Solution|Answer|Proof)[:\s]+(.+?)(?=\n(?:Problem|Example|Exercise|Question)|$)',
                    slide,
                    re.DOTALL | re.IGNORECASE
                )
                
                if solution_match:
                    solution_text = solution_match.group(1).strip()
                    if len(problem_text) > 10 and len(solution_text) > 10:
                        pairs.append({
                            "question": clean_text(problem_text),
                            "solution": clean_text(solution_text)
                        })
    
    # Strategy 3: For textbooks, extract theorem/proof pairs
    if not pairs:
        # Look for theorem statements followed by proofs
        theorem_pattern = r'(?:Theorem|Proposition|Lemma|Corollary)\s*\d*[.:]\s*(.+?)(?=\n(?:Proof|Demonstration):)(?:\n(?:Proof|Demonstration):\s*(.+?))(?=\n(?:Theorem|Proposition|Lemma|Corollary)|$)'
        matches = re.finditer(theorem_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            theorem = match.group(1).strip() if match.lastindex >= 1 else ""
            proof = match.group(2).strip() if match.lastindex >= 2 else ""
            
            if theorem and proof and len(theorem) > 10 and len(proof) > 20:
                pairs.append({
                    "question": f"Prove: {clean_text(theorem)}",
                    "solution": clean_text(proof)
                })
    
    # Strategy 4: If no structured patterns found, chunk by sections
    # This is useful for textbooks where content is more narrative
    if not pairs:
        # Split by common section markers
        sections = re.split(r'\n\s*(?:Chapter|Section|§)\s*\d+', text, flags=re.IGNORECASE)
        
        for section in sections:
            section = section.strip()
            if len(section) < 100:  # Skip very short sections
                continue
            
            # Look for any question-like patterns
            # This is a fallback for less structured content
            question_indicators = [
                r'What\s+(?:is|are|does|do|can|will)',
                r'How\s+(?:do|does|can|will)',
                r'Prove\s+that',
                r'Show\s+that',
                r'Explain\s+(?:why|how|what)',
                r'Find\s+(?:the|a)',
                r'Calculate\s+',
                r'Determine\s+',
            ]
            
            for indicator in question_indicators:
                matches = re.finditer(
                    f'({indicator}.+?)(?=\n\n|\n(?:Solution|Answer)|$)',
                    section,
                    re.DOTALL | re.IGNORECASE
                )
                for match in matches:
                    question_text = match.group(1).strip()
                    if len(question_text) > 20:
                        # Try to find corresponding answer nearby
                        remaining_text = section[section.find(question_text) + len(question_text):]
                        answer_match = re.search(
                            r'(?:Solution|Answer|Explanation)[:\s]*(.+?)(?=\n\n|$)',
                            remaining_text[:500],  # Look in next 500 chars
                            re.DOTALL | re.IGNORECASE
                        )
                        
                        if answer_match:
                            answer_text = answer_match.group(1).strip()
                            if len(answer_text) > 20:
                                pairs.append({
                                    "question": clean_text(question_text),
                                    "solution": clean_text(answer_text)
                                })
                                break
    
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


def process_qa_pair(qa_pair: Dict[str, str], tokenizer: AutoTokenizer) -> List[Dict[str, str]]:
    """
    Process a Q&A pair, chunking if necessary.
    
    Returns a list of processed pairs (may be multiple if chunked).
    """
    question = clean_text(qa_pair.get("question", ""))
    solution = clean_text(qa_pair.get("solution", ""))
    
    if not question or not solution:
        return []
    
    # Combine question and solution for token counting
    combined = f"Question: {question}\nSolution: {solution}"
    
    # Chunk if necessary
    chunks = chunk_text(combined, tokenizer, MAX_TOKENS)
    
    processed_pairs = []
    for chunk in chunks:
        # Try to extract question and solution from chunk
        # If chunking split them, we'll keep them together
        processed_pairs.append({
            "text": chunk,
            "question": question,
            "solution": solution
        })
    
    return processed_pairs


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
    use_temporal_split: bool = True
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Load all .txt, .json, and .pdf files from data directory and split into train/validation/test.
    
    Supports temporal splitting:
    - Past exams/quizzes → validation set
    - Future exams/quizzes → test set
    - All other files → training set
    
    Args:
        data_dir: Directory containing raw data files
        validation_split: Fraction of exam files to use for validation (default: 0.15)
        test_split: Fraction of exam files to use for test (default: 0.15)
        use_temporal_split: If True, use date-based temporal splitting for exams
    
    Returns:
        Tuple of (train_files, validation_files, test_files)
    """
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
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
    
    # First pass: separate exams from non-exams
    for file_path in all_files:
        filename_lower = file_path.name.lower()
        
        # Check if file is an exam/test/quiz
        is_exam = any(keyword in filename_lower for keyword in ['exam', 'test', 'quiz', 'final'])
        
        if is_exam:
            exam_files.append(file_path)
        else:
            # Training files: textbooks, slides, homework, assignments
            train_files.append(file_path)
    
    # Second pass: split exams into validation and test based on temporal ordering
    validation_files = []
    test_files = []
    
    if exam_files and use_temporal_split:
        # Extract dates and sort exams by date
        exam_info = []
        for file_path in exam_files:
            date = extract_date_from_filename(file_path.name)
            exam_info.append(FileInfo(
                path=file_path,
                date=date,
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
        # No temporal splitting: split exams evenly
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


def process_files(files: List[Path], tokenizer: AutoTokenizer, set_name: str) -> List[Dict[str, Any]]:
    """Process a list of files and extract Q&A pairs."""
    data = []
    print(f"\nProcessing {set_name} files...")
    for file_path in files:
        print(f"  Processing: {file_path.name}")
        try:
            file_ext = file_path.suffix.lower()
            if file_ext == '.json':
                qa_pairs = parse_json_file(file_path)
            elif file_ext == '.pdf':
                if not PDF_AVAILABLE:
                    print(f"    Skipping PDF (pdfplumber not installed): {file_path.name}")
                    continue
                qa_pairs = parse_pdf_file(file_path)
            else:
                qa_pairs = parse_text_file(file_path)
            
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
    use_temporal_split: bool = True
):
    """
    Main preprocessing function.
    
    Args:
        raw_data_dir: Path to raw data directory
        processed_data_dir: Path to processed data output directory
        validation_split: Fraction of exam files for validation (default: 0.15)
        test_split: Fraction of exam files for test (default: 0.15)
        use_temporal_split: Use date-based temporal splitting (default: True)
    """
    raw_dir = Path(raw_data_dir)
    processed_dir = Path(processed_data_dir)
    
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = load_tokenizer()
    
    print(f"Scanning for files in: {raw_dir}")
    train_files, validation_files, test_files = load_raw_files(
        raw_dir,
        validation_split=validation_split,
        test_split=test_split,
        use_temporal_split=use_temporal_split
    )
    
    print(f"Found {len(train_files)} training files, {len(validation_files)} validation files, and {len(test_files)} test files")
    
    if validation_files:
        print(f"  Validation files (past exams): {[f.name for f in validation_files]}")
    if test_files:
        print(f"  Test files (future exams): {[f.name for f in test_files]}")
    
    # Process all file sets
    train_data = process_files(train_files, tokenizer, "training")
    validation_data = process_files(validation_files, tokenizer, "validation")
    test_data = process_files(test_files, tokenizer, "test")
    
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
    
    args = parser.parse_args()
    
    main(
        raw_data_dir=args.raw_dir,
        processed_data_dir=args.processed_dir,
        validation_split=args.validation_split,
        test_split=args.test_split,
        use_temporal_split=not args.no_temporal_split
    )

