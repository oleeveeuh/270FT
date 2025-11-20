"""
Utility script to detect duplicate or similar questions between train/validation/test sets.

Helps identify data leakage where the same questions appear in multiple sets.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    """Normalize text for comparison (remove extra whitespace, lowercase, etc.)."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Lowercase
    text = text.lower()
    # Remove punctuation for more lenient matching
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity score between two texts (0-1)."""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def load_questions(file_path: Path) -> List[Dict[str, str]]:
    """Load questions from JSONL or JSON file."""
    questions = []
    
    if file_path.suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        question = item.get("question", item.get("prompt", ""))
                        if question:
                            questions.append({
                                "question": question,
                                "source": file_path.name,
                                "raw_item": item
                            })
                    except json.JSONDecodeError:
                        continue
    else:
        # JSON format
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    question = item.get("question", item.get("prompt", ""))
                    if question:
                        questions.append({
                            "question": question,
                            "source": file_path.name,
                            "raw_item": item
                        })
            elif isinstance(data, dict):
                question = data.get("question", data.get("prompt", ""))
                if question:
                    questions.append({
                        "question": question,
                        "source": file_path.name,
                        "raw_item": data
                    })
    
    return questions


def find_duplicates(
    train_path: Path,
    validation_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    similarity_threshold: float = 0.9
) -> Dict[str, List[Dict]]:
    """
    Find duplicate or similar questions between datasets.
    
    Args:
        train_path: Path to training data file
        validation_path: Path to validation data file (optional)
        test_path: Path to test data file (optional)
        similarity_threshold: Minimum similarity score to consider duplicates (0-1)
    
    Returns:
        Dictionary with duplicate findings
    """
    print("Loading questions from datasets...")
    train_questions = load_questions(train_path)
    print(f"  Loaded {len(train_questions)} questions from training set")
    
    validation_questions = []
    if validation_path and validation_path.exists():
        validation_questions = load_questions(validation_path)
        print(f"  Loaded {len(validation_questions)} questions from validation set")
    
    test_questions = []
    if test_path and test_path.exists():
        test_questions = load_questions(test_path)
        print(f"  Loaded {len(test_questions)} questions from test set")
    
    duplicates = {
        "train_vs_validation": [],
        "train_vs_test": [],
        "validation_vs_test": []
    }
    
    # Compare train vs validation
    if validation_questions:
        print("\nComparing training vs validation sets...")
        for train_q in train_questions:
            for val_q in validation_questions:
                similarity = similarity_score(train_q["question"], val_q["question"])
                if similarity >= similarity_threshold:
                    duplicates["train_vs_validation"].append({
                        "train_question": train_q["question"][:100] + "...",
                        "validation_question": val_q["question"][:100] + "...",
                        "similarity": similarity,
                        "train_source": train_q["source"],
                        "validation_source": val_q["source"]
                    })
    
    # Compare train vs test
    if test_questions:
        print("Comparing training vs test sets...")
        for train_q in train_questions:
            for test_q in test_questions:
                similarity = similarity_score(train_q["question"], test_q["question"])
                if similarity >= similarity_threshold:
                    duplicates["train_vs_test"].append({
                        "train_question": train_q["question"][:100] + "...",
                        "test_question": test_q["question"][:100] + "...",
                        "similarity": similarity,
                        "train_source": train_q["source"],
                        "test_source": test_q["source"]
                    })
    
    # Compare validation vs test
    if validation_questions and test_questions:
        print("Comparing validation vs test sets...")
        for val_q in validation_questions:
            for test_q in test_questions:
                similarity = similarity_score(val_q["question"], test_q["question"])
                if similarity >= similarity_threshold:
                    duplicates["validation_vs_test"].append({
                        "validation_question": val_q["question"][:100] + "...",
                        "test_question": test_q["question"][:100] + "...",
                        "similarity": similarity,
                        "validation_source": val_q["source"],
                        "test_source": test_q["source"]
                    })
    
    return duplicates


def print_duplicate_report(duplicates: Dict[str, List[Dict]]):
    """Print a formatted report of duplicate findings."""
    print("\n" + "="*80)
    print("DUPLICATE DETECTION REPORT")
    print("="*80)
    
    total_duplicates = 0
    
    # Train vs Validation
    if duplicates["train_vs_validation"]:
        print(f"\n⚠️  Found {len(duplicates['train_vs_validation'])} duplicates between TRAIN and VALIDATION:")
        print("-" * 80)
        for i, dup in enumerate(duplicates["train_vs_validation"][:10], 1):  # Show first 10
            print(f"\n{i}. Similarity: {dup['similarity']:.3f}")
            print(f"   Train: {dup['train_question']}")
            print(f"   Validation: {dup['validation_question']}")
        if len(duplicates["train_vs_validation"]) > 10:
            print(f"\n   ... and {len(duplicates['train_vs_validation']) - 10} more")
        total_duplicates += len(duplicates["train_vs_validation"])
    
    # Train vs Test
    if duplicates["train_vs_test"]:
        print(f"\n🚨 Found {len(duplicates['train_vs_test'])} duplicates between TRAIN and TEST:")
        print("-" * 80)
        print("   ⚠️  CRITICAL: This causes data leakage! Remove duplicates from training set.")
        for i, dup in enumerate(duplicates["train_vs_test"][:10], 1):
            print(f"\n{i}. Similarity: {dup['similarity']:.3f}")
            print(f"   Train: {dup['train_question']}")
            print(f"   Test: {dup['test_question']}")
        if len(duplicates["train_vs_test"]) > 10:
            print(f"\n   ... and {len(duplicates['train_vs_test']) - 10} more")
        total_duplicates += len(duplicates["train_vs_test"])
    
    # Validation vs Test
    if duplicates["validation_vs_test"]:
        print(f"\n⚠️  Found {len(duplicates['validation_vs_test'])} duplicates between VALIDATION and TEST:")
        print("-" * 80)
        for i, dup in enumerate(duplicates["validation_vs_test"][:10], 1):
            print(f"\n{i}. Similarity: {dup['similarity']:.3f}")
            print(f"   Validation: {dup['validation_question']}")
            print(f"   Test: {dup['test_question']}")
        if len(duplicates["validation_vs_test"]) > 10:
            print(f"\n   ... and {len(duplicates['validation_vs_test']) - 10} more")
        total_duplicates += len(duplicates["validation_vs_test"])
    
    if total_duplicates == 0:
        print("\n✅ No duplicates found! Your datasets are clean.")
    else:
        print(f"\n📊 Total duplicates found: {total_duplicates}")
        print("\n💡 Recommendation:")
        print("   - Remove duplicate questions from TRAINING set")
        print("   - Keep them in VALIDATION/TEST sets for evaluation")
        print("   - This ensures evaluation is on 'unseen' problems")
    
    print("="*80)


def main():
    """Main function to detect duplicates."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect duplicate questions between train/validation/test sets"
    )
    parser.add_argument(
        "--train",
        type=str,
        default="270FT/data/processed/train.jsonl",
        help="Path to training data file"
    )
    parser.add_argument(
        "--validation",
        type=str,
        default="270FT/data/processed/validation.jsonl",
        help="Path to validation data file (optional)"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="270FT/data/processed/test.jsonl",
        help="Path to test data file (optional)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for duplicate detection (0-1, default: 0.9)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file to save duplicate report (optional)"
    )
    
    args = parser.parse_args()
    
    train_path = Path(args.train)
    validation_path = Path(args.validation) if args.validation else None
    test_path = Path(args.test) if args.test else None
    
    if not train_path.exists():
        print(f"Error: Training file not found: {train_path}")
        return
    
    duplicates = find_duplicates(
        train_path,
        validation_path,
        test_path,
        similarity_threshold=args.threshold
    )
    
    print_duplicate_report(duplicates)
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(duplicates, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Duplicate report saved to: {output_path}")


if __name__ == "__main__":
    main()

