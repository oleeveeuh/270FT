#!/usr/bin/env python3
"""
Unified command-line interface for the 270FT fine-tuning pipeline.

This script provides a single entry point to run the entire pipeline or individual steps:
- preprocess: Prepare data from raw files
- train: Fine-tune models with QLoRA
- evaluate: Evaluate trained models on test set
- query: Interactive CLI for querying models
- pipeline: Run the complete pipeline (preprocess → train → evaluate)
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def run_preprocess(
    raw_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
    validation_split: float = 0.15,
    test_split: float = 0.15,
    use_temporal_split: bool = True,
    use_unit_split: bool = False,
) -> int:
    """Run the preprocessing step."""
    project_root = get_project_root()
    preprocess_script = project_root / "270FT" / "preprocess" / "load_and_prepare.py"
    
    if not preprocess_script.exists():
        print(f"Error: Preprocessing script not found at {preprocess_script}")
        return 1
    
    # Set default paths relative to project root if not provided
    # Use absolute paths to avoid working directory issues
    if raw_dir is None:
        raw_dir = str((project_root / "270FT" / "data" / "raw").resolve())
    else:
        # Convert to absolute path if relative
        raw_dir_path = Path(raw_dir)
        if not raw_dir_path.is_absolute():
            raw_dir = str((project_root / raw_dir).resolve())
        else:
            raw_dir = str(raw_dir_path.resolve())
    
    if processed_dir is None:
        processed_dir = str((project_root / "270FT" / "data" / "processed").resolve())
    else:
        # Convert to absolute path if relative
        processed_dir_path = Path(processed_dir)
        if not processed_dir_path.is_absolute():
            processed_dir = str((project_root / processed_dir).resolve())
        else:
            processed_dir = str(processed_dir_path.resolve())
    
    cmd = [sys.executable, str(preprocess_script)]
    
    cmd.extend(["--raw_dir", raw_dir])
    cmd.extend(["--processed_dir", processed_dir])
    if validation_split != 0.15:
        cmd.extend(["--validation_split", str(validation_split)])
    if test_split != 0.15:
        cmd.extend(["--test_split", str(test_split)])
    if not use_temporal_split:
        cmd.append("--no_temporal_split")
    if use_unit_split:
        cmd.append("--use_unit_split")
    
    print("=" * 60)
    print("STEP 1: Data Preprocessing")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def run_training() -> int:
    """Run the training step."""
    project_root = get_project_root()
    train_script = project_root / "270FT" / "training" / "train_dual_lora.py"
    
    if not train_script.exists():
        print(f"Error: Training script not found at {train_script}")
        return 1
    
    cmd = [sys.executable, str(train_script)]
    
    print("=" * 60)
    print("STEP 2: Model Training")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}\n")
    print("Note: This may take several hours depending on your GPU and dataset size.\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def run_evaluation() -> int:
    """Run the evaluation step."""
    project_root = get_project_root()
    eval_script = project_root / "270FT" / "evaluation" / "evaluate_models.py"
    
    if not eval_script.exists():
        print(f"Error: Evaluation script not found at {eval_script}")
        return 1
    
    cmd = [sys.executable, str(eval_script)]
    
    print("=" * 60)
    print("STEP 3: Model Evaluation")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def run_query() -> int:
    """Run the interactive query CLI."""
    project_root = get_project_root()
    main_script = project_root / "main.py"
    
    if not main_script.exists():
        print(f"Error: Main CLI script not found at {main_script}")
        return 1
    
    cmd = [sys.executable, str(main_script)]
    
    print("=" * 60)
    print("Interactive Model Query CLI")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def run_pipeline(
    raw_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
    validation_split: float = 0.15,
    test_split: float = 0.15,
    use_temporal_split: bool = True,
    use_unit_split: bool = False,
    skip_preprocess: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
) -> int:
    """Run the complete pipeline."""
    print("=" * 60)
    print("270FT - Complete Pipeline")
    print("=" * 60)
    print("\nThis will run the following steps:")
    if not skip_preprocess:
        print("  1. Data Preprocessing")
    if not skip_train:
        print("  2. Model Training")
    if not skip_eval:
        print("  3. Model Evaluation")
    print()
    
    # Step 1: Preprocessing
    if not skip_preprocess:
        result = run_preprocess(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            validation_split=validation_split,
            test_split=test_split,
            use_temporal_split=use_temporal_split,
            use_unit_split=use_unit_split,
        )
        if result != 0:
            print("\n[ERROR] Preprocessing failed. Aborting pipeline.")
            return result
        print("\n[OK] Preprocessing completed successfully\n")
    else:
        print("[SKIP] Preprocessing step skipped\n")
    
    # Step 2: Training
    if not skip_train:
        result = run_training()
        if result != 0:
            print("\n[ERROR] Training failed. Aborting pipeline.")
            return result
        print("\n[OK] Training completed successfully\n")
    else:
        print("[SKIP] Training step skipped\n")
    
    # Step 3: Evaluation
    if not skip_eval:
        result = run_evaluation()
        if result != 0:
            print("\n[ERROR] Evaluation failed. Aborting pipeline.")
            return result
        print("\n[OK] Evaluation completed successfully\n")
    else:
        print("[SKIP] Evaluation step skipped\n")
    
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - View results: cat 270FT/results/metrics_report.json")
    print("  - Query models: python cli.py query")
    print("  - Visualize: jupyter notebook 270FT/notebooks/visualize_results.ipynb")
    print()
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="270FT Fine-Tuning Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python cli.py pipeline

  # Run individual steps
  python cli.py preprocess
  python cli.py train
  python cli.py evaluate
  python cli.py query

  # Preprocess with custom options
  python cli.py preprocess --raw_dir data/raw --use_unit_split

  # Run pipeline but skip preprocessing
  python cli.py pipeline --skip-preprocess
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Prepare data from raw files")
    preprocess_parser.add_argument("--raw_dir", type=str, help="Raw data directory (default: 270FT/data/raw)")
    preprocess_parser.add_argument("--processed_dir", type=str, help="Processed data directory (default: 270FT/data/processed)")
    preprocess_parser.add_argument("--validation_split", type=float, default=0.15, help="Fraction of exams for validation (default: 0.15)")
    preprocess_parser.add_argument("--test_split", type=float, default=0.15, help="Fraction of exams for test (default: 0.15)")
    preprocess_parser.add_argument("--no_temporal_split", action="store_true", help="Disable temporal splitting")
    preprocess_parser.add_argument("--use_unit_split", action="store_true", help="Use unit-based splitting")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune models with QLoRA")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models on test set")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Interactive CLI for querying models")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete pipeline (preprocess → train → evaluate)")
    pipeline_parser.add_argument("--raw_dir", type=str, help="Raw data directory (default: 270FT/data/raw)")
    pipeline_parser.add_argument("--processed_dir", type=str, help="Processed data directory (default: 270FT/data/processed)")
    pipeline_parser.add_argument("--validation_split", type=float, default=0.15, help="Fraction of exams for validation (default: 0.15)")
    pipeline_parser.add_argument("--test_split", type=float, default=0.15, help="Fraction of exams for test (default: 0.15)")
    pipeline_parser.add_argument("--no_temporal_split", action="store_true", help="Disable temporal splitting")
    pipeline_parser.add_argument("--use_unit_split", action="store_true", help="Use unit-based splitting")
    pipeline_parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    pipeline_parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    pipeline_parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate function
    if args.command == "preprocess":
        return run_preprocess(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            validation_split=args.validation_split,
            test_split=args.test_split,
            use_temporal_split=not args.no_temporal_split,
            use_unit_split=args.use_unit_split,
        )
    elif args.command == "train":
        return run_training()
    elif args.command == "evaluate":
        return run_evaluation()
    elif args.command == "query":
        return run_query()
    elif args.command == "pipeline":
        return run_pipeline(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            validation_split=args.validation_split,
            test_split=args.test_split,
            use_temporal_split=not args.no_temporal_split,
            use_unit_split=args.use_unit_split,
            skip_preprocess=args.skip_preprocess,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
        )
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

