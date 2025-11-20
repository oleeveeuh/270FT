"""
Human-in-the-loop evaluation script for test set.

This script generates an evaluation form for professors/experts to rate model outputs
on various criteria, then aggregates the results for comparison with automated metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


# Scoring Rubric
SCORING_RUBRIC = {
    "mathematical_correctness": {
        "description": "Is the mathematical content correct?",
        "scale": "1-5",
        "guidelines": {
            5: "Fully correct - all mathematical statements, formulas, and proofs are accurate",
            4: "Mostly correct - minor errors that don't affect main conclusion",
            3: "Partially correct - some correct elements but significant errors",
            2: "Mostly incorrect - few correct elements, major errors",
            1: "Completely incorrect - no valid mathematical content"
        }
    },
    "completeness": {
        "description": "Is the solution complete? Does it address all parts of the question?",
        "scale": "1-5",
        "guidelines": {
            5: "Fully complete - addresses all parts, no missing steps",
            4: "Mostly complete - addresses main parts, minor omissions",
            3: "Partially complete - addresses some parts, missing important steps",
            2: "Incomplete - addresses few parts, many missing steps",
            1: "Very incomplete - barely addresses the question"
        }
    },
    "clarity": {
        "description": "Is the explanation clear and well-structured?",
        "scale": "1-5",
        "guidelines": {
            5: "Excellent - very clear, well-organized, easy to follow",
            4: "Good - clear with minor organizational issues",
            3: "Fair - understandable but could be clearer",
            2: "Poor - confusing or poorly organized",
            1: "Very poor - difficult to understand"
        }
    },
    "overall_quality": {
        "description": "Overall assessment of the solution quality",
        "scale": "1-5",
        "guidelines": {
            5: "Excellent - would receive full credit",
            4: "Good - would receive most credit with minor deductions",
            3: "Fair - would receive partial credit",
            2: "Poor - would receive minimal credit",
            1: "Very poor - would receive no credit"
        }
    }
}


def generate_evaluation_form(
    test_results_path: str,
    output_path: str,
    model_name: Optional[str] = None
) -> str:
    """
    Generate a human evaluation form from test results.
    
    Args:
        test_results_path: Path to test results JSON (from evaluate_models.py)
        output_path: Path to save evaluation form
        model_name: Optional model name filter
    
    Returns:
        Path to generated evaluation form
    """
    # Load test results
    with open(test_results_path, 'r') as f:
        results = json.load(f)
    
    # Extract per-item results
    if model_name:
        model_results = results.get("model_results", {}).get(model_name, {})
    else:
        # Use first model if no name specified
        model_results = list(results.get("model_results", {}).values())[0]
    
    per_item_results = model_results.get("per_item_results", [])
    
    # Generate evaluation form
    form_data = {
        "instructions": {
            "title": "Human Evaluation Form - Model Output Assessment",
            "description": "Please evaluate each model-generated solution using the rubric below.",
            "rubric": SCORING_RUBRIC,
            "total_items": len(per_item_results)
        },
        "items": []
    }
    
    for idx, item in enumerate(per_item_results):
        form_item = {
            "item_id": idx + 1,
            "question": item.get("prompt", item.get("question", "")),
            "model_prediction": item.get("prediction", ""),
            "reference_solution": item.get("reference", item.get("solution", "")),
            "automated_metrics": {
                "exact_match": item.get("exact_match", False),
                "bleu_score": item.get("bleu_score", 0.0),
                "symbolic_equivalence": item.get("symbolic_equivalence", False)
            },
            "human_evaluation": {
                "mathematical_correctness": None,
                "completeness": None,
                "clarity": None,
                "overall_quality": None,
                "comments": ""
            }
        }
        form_data["items"].append(form_item)
    
    # Save form
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(form_data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated evaluation form with {len(per_item_results)} items")
    print(f"Saved to: {output_path}")
    print(f"\nRubric:")
    for criterion, details in SCORING_RUBRIC.items():
        print(f"  {criterion}: {details['description']} (Scale: {details['scale']})")
    
    return output_path


def load_human_evaluations(evaluation_form_path: str) -> Dict[str, Any]:
    """
    Load completed human evaluation form.
    
    Args:
        evaluation_form_path: Path to completed evaluation form JSON
    
    Returns:
        Dictionary with evaluation data
    """
    with open(evaluation_form_path, 'r') as f:
        return json.load(f)


def aggregate_human_evaluations(
    evaluation_form_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Aggregate human evaluation scores and compute statistics.
    
    Args:
        evaluation_form_path: Path to completed evaluation form
        output_path: Optional path to save aggregated results
    
    Returns:
        Dictionary with aggregated statistics
    """
    data = load_human_evaluations(evaluation_form_path)
    items = data.get("items", [])
    
    # Extract scores
    scores = {
        "mathematical_correctness": [],
        "completeness": [],
        "clarity": [],
        "overall_quality": []
    }
    
    completed_items = 0
    for item in items:
        human_eval = item.get("human_evaluation", {})
        if human_eval.get("overall_quality") is not None:
            completed_items += 1
            for criterion in scores.keys():
                score = human_eval.get(criterion)
                if score is not None:
                    scores[criterion].append(score)
    
    # Compute statistics
    stats = {
        "total_items": len(items),
        "completed_items": completed_items,
        "completion_rate": completed_items / len(items) if items else 0.0,
        "average_scores": {},
        "score_distributions": {}
    }
    
    for criterion, score_list in scores.items():
        if score_list:
            stats["average_scores"][criterion] = sum(score_list) / len(score_list)
            stats["score_distributions"][criterion] = {
                "min": min(score_list),
                "max": max(score_list),
                "median": sorted(score_list)[len(score_list) // 2],
                "counts": {i: score_list.count(i) for i in range(1, 6)}
            }
        else:
            stats["average_scores"][criterion] = None
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Aggregated results saved to: {output_path}")
    
    return stats


def compare_automated_vs_human(
    test_results_path: str,
    human_evaluation_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare automated metrics with human evaluations.
    
    Args:
        test_results_path: Path to automated test results
        human_evaluation_path: Path to human evaluation form
        output_path: Optional path to save comparison results
    
    Returns:
        Dictionary with comparison statistics
    """
    # Load automated results
    with open(test_results_path, 'r') as f:
        auto_results = json.load(f)
    
    # Load human evaluations
    human_data = load_human_evaluations(human_evaluation_path)
    
    # Extract per-item data
    items = human_data.get("items", [])
    
    comparison = {
        "automated_metrics": {},
        "human_metrics": {},
        "correlations": {},
        "per_item_comparison": []
    }
    
    # Aggregate automated metrics
    if "model_results" in auto_results:
        for model_name, model_data in auto_results["model_results"].items():
            per_item = model_data.get("per_item_results", [])
            if per_item:
                comparison["automated_metrics"][model_name] = {
                    "exact_match_rate": model_data.get("exact_match_rate", 0.0),
                    "symbolic_equivalence_rate": model_data.get("symbolic_equivalence_rate", 0.0),
                    "avg_bleu_score": model_data.get("avg_bleu_score", 0.0)
                }
    
    # Aggregate human metrics
    human_stats = aggregate_human_evaluations(human_evaluation_path)
    comparison["human_metrics"] = human_stats["average_scores"]
    
    # Per-item comparison
    for item in items:
        auto_metrics = item.get("automated_metrics", {})
        human_eval = item.get("human_evaluation", {})
        
        comparison["per_item_comparison"].append({
            "item_id": item.get("item_id"),
            "automated": {
                "exact_match": auto_metrics.get("exact_match", False),
                "bleu_score": auto_metrics.get("bleu_score", 0.0),
                "symbolic_equivalence": auto_metrics.get("symbolic_equivalence", False)
            },
            "human": {
                "mathematical_correctness": human_eval.get("mathematical_correctness"),
                "completeness": human_eval.get("completeness"),
                "clarity": human_eval.get("clarity"),
                "overall_quality": human_eval.get("overall_quality")
            }
        })
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison results saved to: {output_path}")
    
    return comparison


def main():
    """Main function to generate or aggregate evaluation form."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate or aggregate human evaluation form")
    parser.add_argument(
        "--test_results",
        type=str,
        default="270FT/results/metrics_report.json",
        help="Path to test results JSON (for generating form)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="270FT/results/human_evaluation_form.json",
        help="Path to save evaluation form or aggregated results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to evaluate (optional)"
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default=None,
        help="Path to completed evaluation form to aggregate"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        nargs=2,
        metavar=("TEST_RESULTS", "HUMAN_EVAL"),
        help="Compare automated results with human evaluation (provide both paths)"
    )
    
    args = parser.parse_args()
    
    if args.aggregate:
        # Aggregate completed evaluation form
        stats = aggregate_human_evaluations(
            args.aggregate,
            args.output if args.output != parser.get_default("output") else None
        )
        print(f"\nAggregation complete!")
        print(f"Average scores:")
        for criterion, score in stats["average_scores"].items():
            if score is not None:
                print(f"  {criterion}: {score:.2f}/5.0")
    
    elif args.compare:
        # Compare automated vs human
        comparison = compare_automated_vs_human(
            args.compare[0],
            args.compare[1],
            args.output
        )
        print(f"\nComparison complete!")
        print(f"Automated metrics: {comparison['automated_metrics']}")
        print(f"Human metrics: {comparison['human_metrics']}")
    
    else:
        # Generate evaluation form
        generate_evaluation_form(
            args.test_results,
            args.output,
            args.model
        )
        
        print(f"\nNext steps:")
        print(f"1. Open {args.output}")
        print(f"2. Fill in the 'human_evaluation' scores for each item (1-5 scale)")
        print(f"3. Add comments if desired")
        print(f"4. Run: python -m 270FT.evaluation.human_evaluation --aggregate {args.output}")


if __name__ == "__main__":
    main()

