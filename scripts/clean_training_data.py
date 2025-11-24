#!/usr/bin/env python3
"""
Clean training data by removing entries with excessively long text.

This script filters out corrupt data entries where the entire document
was concatenated instead of individual Q&A pairs being extracted.
"""

import json
from pathlib import Path

def clean_jsonl_file(input_path: Path, output_path: Path, max_chars: int = 50000):
    """
    Clean a JSONL file by removing entries with text longer than max_chars.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        max_chars: Maximum character length for question or solution fields
    """
    cleaned_entries = []
    skipped_entries = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
                q_len = len(entry.get('question', ''))
                s_len = len(entry.get('solution', ''))

                # Skip entries with excessively long text
                if q_len > max_chars or s_len > max_chars:
                    skipped_entries.append(i)
                    continue

                cleaned_entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed entry at line {i}: {e}")
                skipped_entries.append(i)
                continue

    # Write cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in cleaned_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return len(cleaned_entries), len(skipped_entries)


def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"

    print("Cleaning training data...")
    print(f"Processing directory: {processed_dir}")
    print()

    # Clean train.jsonl
    train_input = processed_dir / "train.jsonl"
    train_output = processed_dir / "train_cleaned.jsonl"
    train_backup = processed_dir / "train_original.jsonl"

    if train_input.exists():
        print(f"Cleaning {train_input.name}...")
        kept, skipped = clean_jsonl_file(train_input, train_output, max_chars=50000)

        print(f"  Kept: {kept} entries")
        print(f"  Skipped: {skipped} entries (excessively long text)")

        # Backup original and replace with cleaned version
        if kept > 0:
            print(f"  Creating backup: {train_backup.name}")
            train_input.rename(train_backup)
            train_output.rename(train_input)
            print(f"  ✓ Replaced {train_input.name} with cleaned version")
        else:
            print(f"  ERROR: No valid entries found! Not replacing original file.")
    else:
        print(f"ERROR: {train_input} not found")

    print()
    print("=" * 60)
    print("Cleaning complete!")
    print()
    print("Summary:")
    print(f"  - Original data backed up to: {train_backup.name}")
    print(f"  - Cleaned data: {train_input.name}")
    print(f"  - Valid training examples: {kept}")
    print("=" * 60)


if __name__ == "__main__":
    main()
