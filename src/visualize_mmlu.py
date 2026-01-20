"""
MMLU results visualization script.

Generates visualization charts for MMLU evaluation results:
- Category accuracy bar chart comparing checkpoints
- Checkpoint comparison line chart
- Subject × checkpoint heatmap

Usage:
    # Visualize all results in a directory
    python src/visualize_mmlu.py --results-dir outputs/mmlu_results

    # Visualize specific result files
    python src/visualize_mmlu.py --results file1.json file2.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.mmlu_categories import MMLU_SUBJECTS, MMLU_CATEGORIES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize MMLU evaluation results")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing MMLU result JSON files",
    )
    group.add_argument(
        "--results",
        type=str,
        nargs="+",
        help="Specific result JSON files to visualize",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as results-dir or outputs/mmlu_results/plots)",
    )
    
    return parser.parse_args()


def load_results(file_paths: List[str]) -> List[Dict]:
    """
    Load MMLU results from JSON files.
    
    Args:
        file_paths: List of paths to JSON result files
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    for path in file_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                data["_source_file"] = Path(path).name
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    
    # Sort by model path (to get chronological order for checkpoints)
    results.sort(key=lambda x: x.get("model_path", x.get("_source_file", "")))
    
    return results


def get_checkpoint_label(result: Dict) -> str:
    """
    Get a short label for a checkpoint.
    
    Args:
        result: Result dictionary
        
    Returns:
        Short label string
    """
    model_path = result.get("model_path", "unknown")
    
    # Extract checkpoint name from path
    if "/" in model_path or "\\" in model_path:
        name = Path(model_path).name
    else:
        name = model_path
    
    # Shorten common patterns
    if name.startswith("allenai/") or name == "allenai/OLMo-1B-hf":
        return "Base Model"
    
    return name


def plot_category_accuracy(results: List[Dict], output_path: str):
    """
    Create a grouped bar chart comparing category accuracy across checkpoints.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Prepare data
    checkpoint_labels = [get_checkpoint_label(r) for r in results]
    categories = MMLU_CATEGORIES
    
    # Create matrix of accuracies
    accuracies = []
    for result in results:
        cat_acc = []
        for category in categories:
            if category in result.get("categories", {}):
                cat_acc.append(result["categories"][category]["accuracy"] * 100)
            else:
                cat_acc.append(0)
        accuracies.append(cat_acc)
    
    accuracies = np.array(accuracies).T  # Shape: (categories, checkpoints)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(categories))
    width = 0.8 / len(results)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
    
    for i, (label, color) in enumerate(zip(checkpoint_labels, colors)):
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accuracies[:, i], width, label=label, color=color)
    
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("MMLU Accuracy by Category Across Checkpoints", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved category accuracy chart to {output_path}")


def plot_checkpoint_comparison(results: List[Dict], output_path: str):
    """
    Create a line chart showing overall accuracy across checkpoints.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Prepare data
    checkpoint_labels = [get_checkpoint_label(r) for r in results]
    overall_accuracies = [
        r.get("overall", {}).get("accuracy", 0) * 100 
        for r in results
    ]
    
    # Also get category accuracies for multi-line plot
    category_data = {}
    for category in MMLU_CATEGORIES:
        category_data[category] = [
            r.get("categories", {}).get(category, {}).get("accuracy", 0) * 100
            for r in results
        ]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(checkpoint_labels))
    
    # Plot overall accuracy as thick line
    ax.plot(x, overall_accuracies, "ko-", linewidth=3, markersize=10, 
            label="Overall", zorder=10)
    
    # Plot category accuracies as thinner lines
    colors = plt.cm.tab10(np.linspace(0, 1, len(MMLU_CATEGORIES)))
    for category, color in zip(MMLU_CATEGORIES, colors):
        ax.plot(x, category_data[category], "o-", color=color, 
                linewidth=1, markersize=4, alpha=0.6, label=category)
    
    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("MMLU Accuracy Progression Across Checkpoints", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoint_labels, rotation=45, ha="right", fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved checkpoint comparison chart to {output_path}")


def plot_subject_heatmap(results: List[Dict], output_path: str):
    """
    Create a heatmap showing accuracy for each subject × checkpoint.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Prepare data
    checkpoint_labels = [get_checkpoint_label(r) for r in results]
    subjects = MMLU_SUBJECTS
    
    # Create matrix of accuracies
    accuracies = []
    for subject in subjects:
        subject_acc = []
        for result in results:
            if subject in result.get("subjects", {}):
                subject_acc.append(result["subjects"][subject]["accuracy"] * 100)
            else:
                subject_acc.append(0)
        accuracies.append(subject_acc)
    
    accuracies = np.array(accuracies)  # Shape: (subjects, checkpoints)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 20))
    
    im = ax.imshow(accuracies, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5)
    cbar.ax.set_ylabel("Accuracy (%)", rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xticks(np.arange(len(checkpoint_labels)))
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_xticklabels(checkpoint_labels, rotation=45, ha="right")
    ax.set_yticklabels(subjects, fontsize=8)
    
    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Subject", fontsize=12)
    ax.set_title("MMLU Subject Accuracy Heatmap", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved subject heatmap to {output_path}")


def create_summary_table(results: List[Dict], output_path: str):
    """
    Create a summary table in markdown format.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the table
    """
    if not results:
        return
    
    checkpoint_labels = [get_checkpoint_label(r) for r in results]
    
    lines = ["# MMLU Evaluation Summary\n"]
    
    # Overall accuracy table
    lines.append("## Overall Accuracy\n")
    lines.append("| Checkpoint | Accuracy | Correct | Total |")
    lines.append("|------------|----------|---------|-------|")
    
    for result, label in zip(results, checkpoint_labels):
        overall = result.get("overall", {})
        acc = overall.get("accuracy", 0) * 100
        correct = overall.get("correct", 0)
        total = overall.get("total", 0)
        lines.append(f"| {label} | {acc:.1f}% | {correct} | {total} |")
    
    lines.append("\n")
    
    # Category accuracy table
    lines.append("## Accuracy by Category\n")
    
    header = "| Category |" + " | ".join(checkpoint_labels) + " |"
    separator = "|----------|" + " | ".join(["---"] * len(checkpoint_labels)) + " |"
    lines.append(header)
    lines.append(separator)
    
    for category in MMLU_CATEGORIES:
        row = f"| {category} |"
        for result in results:
            cat_data = result.get("categories", {}).get(category, {})
            acc = cat_data.get("accuracy", 0) * 100
            row += f" {acc:.1f}% |"
        lines.append(row)
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved summary table to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine result files to load
    if args.results_dir:
        results_dir = Path(args.results_dir)
        result_files = sorted(results_dir.glob("mmlu_*.json"))
        if not result_files:
            print(f"No MMLU result files found in {results_dir}")
            return
        result_files = [str(f) for f in result_files]
    else:
        result_files = args.results
    
    print(f"Loading {len(result_files)} result files...")
    results = load_results(result_files)
    
    if not results:
        print("No valid results loaded")
        return
    
    print(f"Loaded {len(results)} result sets")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.results_dir:
        output_dir = Path(args.results_dir) / "plots"
    else:
        output_dir = Path("outputs/mmlu_results/plots")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {output_dir}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_category_accuracy(results, str(output_dir / "category_accuracy.png"))
    plot_checkpoint_comparison(results, str(output_dir / "checkpoint_comparison.png"))
    plot_subject_heatmap(results, str(output_dir / "subject_heatmap.png"))
    create_summary_table(results, str(output_dir / "summary.md"))
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
