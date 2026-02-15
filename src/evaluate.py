"""Evaluation script to fetch results from WandB and generate comparison figures."""

# [VALIDATOR FIX - Attempt 1]
# [PROBLEM]: src.evaluate was called with Hydra-style arguments (key=value) but used argparse expecting --key value format
# [CAUSE]: Workflow calls "uv run python -u -m src.evaluate results_dir=... run_ids=..." but evaluate.py uses argparse
# [FIX]: Replace argparse with Hydra to match the calling convention used by main.py and inference.py
#
# [OLD CODE]:
# import argparse
# def parse_args():
#     parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
#     parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
#     parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs")
#     parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
#     parser.add_argument("--wandb_project", type=str, default=None, help="WandB project")
#     return parser.parse_args()
#
# [NEW CODE]:
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
import hydra
from omegaconf import DictConfig
import wandb
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def fetch_wandb_run(entity: str, project: str, run_id: str) -> Optional[Dict]:
    """Fetch run data from WandB API.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run ID
        
    Returns:
        Dictionary with run config, summary, and history
    """
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Fetch history (step-by-step metrics)
        history = run.history()
        
        return {
            "run_id": run_id,
            "config": dict(run.config),
            "summary": dict(run.summary),
            "history": history,
            "name": run.name,
            "tags": run.tags
        }
    except Exception as e:
        print(f"Warning: Could not fetch WandB run {run_id}: {e}")
        return None


def load_local_metrics(results_dir: Path, run_id: str) -> Optional[Dict]:
    """Load metrics from local results directory as fallback.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
        
    Returns:
        Dictionary with metrics
    """
    metrics_file = results_dir / run_id / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    return None


def export_per_run_metrics(results_dir: Path, run_data: Dict):
    """Export per-run metrics and figures.
    
    Args:
        results_dir: Results directory
        run_data: Run data dictionary
    """
    run_id = run_data["run_id"]
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Export metrics
    metrics = {
        "run_id": run_id,
        "summary": run_data.get("summary", {}),
        "config": run_data.get("config", {})
    }
    
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics: {metrics_file}")
    
    # Generate per-run figures
    history = run_data.get("history")
    if history is not None and not history.empty:
        # Accuracy over time
        if "accuracy" in history.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(history.index, history["accuracy"], linewidth=2)
            ax.set_xlabel("Question Index")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Accuracy over Questions - {run_id}")
            ax.grid(True, alpha=0.3)
            
            accuracy_fig = run_dir / "accuracy_progress.pdf"
            fig.savefig(accuracy_fig, bbox_inches="tight")
            plt.close(fig)
            print(f"Generated figure: {accuracy_fig}")


def create_comparison_figures(results_dir: Path, all_runs: List[Dict]):
    """Generate comparison figures across runs.
    
    Args:
        results_dir: Results directory
        all_runs: List of run data dictionaries
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(all_runs))
    
    # Accuracy comparison bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    run_ids = []
    accuracies = []
    methods = []
    
    for run_data in all_runs:
        run_id = run_data["run_id"]
        summary = run_data.get("summary", {})
        config = run_data.get("config", {})
        
        accuracy = summary.get("accuracy", 0.0)
        method = config.get("run", {}).get("method", {}).get("name", "unknown")
        
        run_ids.append(run_id)
        accuracies.append(accuracy)
        methods.append(method)
    
    # Create bar plot
    x_pos = np.arange(len(run_ids))
    bars = ax.bar(x_pos, accuracies, color=colors, alpha=0.7, edgecolor="black")
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel("Run ID", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Comparison Across Runs", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    accuracy_comparison_fig = comparison_dir / "comparison_accuracy.pdf"
    fig.savefig(accuracy_comparison_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated figure: {accuracy_comparison_fig}")
    
    # Accuracy progress overlay (if history available)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    has_history = False
    for i, run_data in enumerate(all_runs):
        history = run_data.get("history")
        if history is not None and not history.empty and "accuracy" in history.columns:
            has_history = True
            run_id = run_data["run_id"]
            ax.plot(history.index, history["accuracy"], 
                   label=run_id, linewidth=2, color=colors[i], alpha=0.8)
    
    if has_history:
        ax.set_xlabel("Question Index", fontsize=12)
        ax.set_ylabel("Cumulative Accuracy", fontsize=12)
        ax.set_title("Accuracy Progress Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        progress_fig = comparison_dir / "comparison_accuracy_progress.pdf"
        fig.savefig(progress_fig, bbox_inches="tight")
        print(f"Generated figure: {progress_fig}")
    
    plt.close(fig)


def compute_aggregated_metrics(all_runs: List[Dict]) -> Dict:
    """Compute aggregated metrics across runs.
    
    Args:
        all_runs: List of run data dictionaries
        
    Returns:
        Aggregated metrics dictionary
    """
    metrics_by_run = {}
    
    for run_data in all_runs:
        run_id = run_data["run_id"]
        summary = run_data.get("summary", {})
        config = run_data.get("config", {})
        
        method_type = config.get("run", {}).get("method", {}).get("type", "unknown")
        
        metrics_by_run[run_id] = {
            "accuracy": summary.get("accuracy", 0.0),
            "correct_count": summary.get("correct_count", 0),
            "total_count": summary.get("total_count", 0),
            "method_type": method_type
        }
    
    # Find best proposed and best baseline
    proposed_accuracies = {
        rid: m["accuracy"] 
        for rid, m in metrics_by_run.items() 
        if m["method_type"] == "proposed"
    }
    
    baseline_accuracies = {
        rid: m["accuracy"] 
        for rid, m in metrics_by_run.items() 
        if m["method_type"].startswith("comparative")
    }
    
    best_proposed = max(proposed_accuracies.items(), key=lambda x: x[1]) if proposed_accuracies else (None, 0.0)
    best_baseline = max(baseline_accuracies.items(), key=lambda x: x[1]) if baseline_accuracies else (None, 0.0)
    
    gap = best_proposed[1] - best_baseline[1] if best_proposed[0] and best_baseline[0] else 0.0
    
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": {
            "run_id": best_proposed[0],
            "accuracy": best_proposed[1]
        },
        "best_baseline": {
            "run_id": best_baseline[0],
            "accuracy": best_baseline[1]
        },
        "gap": gap
    }
    
    return aggregated


# [VALIDATOR FIX - Attempt 2]
# [PROBLEM]: src.evaluate failed with "You must specify 'run'" error when called from visualization workflow
# [CAUSE]: evaluate.py uses @hydra.main with config.yaml which requires 'run: ???' parameter, but evaluate.py doesn't need run configs
# [FIX]: Use a separate evaluate.yaml config file that doesn't require 'run' parameter
#
# [OLD CODE]:
# @hydra.main(config_path="../config", config_name="config", version_base=None)
#
# [NEW CODE]:
@hydra.main(config_path="../config", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    """Main evaluation script."""
    # Extract parameters from config (with CLI overrides)
    results_dir = Path(cfg.results_dir)
    
    # Handle run_ids parameter - can be a string (JSON) or list
    run_ids_raw = cfg.run_ids
    if isinstance(run_ids_raw, str):
        # If passed as CLI arg, it will be a JSON string
        run_ids = json.loads(run_ids_raw) if run_ids_raw else []
    else:
        # If from config file, it will already be a list
        run_ids = list(run_ids_raw) if run_ids_raw else []
    
    print(f"Evaluating runs: {run_ids}")
    print(f"Results directory: {results_dir}")
    
    # Get WandB credentials from config or environment
    wandb_entity = cfg.wandb.get("entity") or os.environ.get("WANDB_ENTITY")
    wandb_project = cfg.wandb.get("project") or os.environ.get("WANDB_PROJECT")
    
    if not wandb_entity or not wandb_project:
        print("Warning: WandB entity/project not specified. Will use local metrics only.")
    
    # Fetch data for each run
    all_runs = []
    for run_id in run_ids:
        print(f"\nProcessing run: {run_id}")
        
        # Try to fetch from WandB
        run_data = None
        if wandb_entity and wandb_project:
            run_data = fetch_wandb_run(wandb_entity, wandb_project, run_id)
        
        # Fallback to local metrics
        if run_data is None:
            print(f"Falling back to local metrics for {run_id}")
            local_metrics = load_local_metrics(results_dir, run_id)
            if local_metrics:
                run_data = {
                    "run_id": run_id,
                    "config": {},
                    "summary": local_metrics,
                    "history": None
                }
        
        if run_data is None:
            print(f"Warning: No data found for run {run_id}, skipping")
            continue
        
        all_runs.append(run_data)
        
        # Export per-run metrics and figures
        export_per_run_metrics(results_dir, run_data)
    
    if not all_runs:
        print("Error: No run data available")
        sys.exit(1)
    
    # Compute aggregated metrics
    print("\nComputing aggregated metrics...")
    aggregated = compute_aggregated_metrics(all_runs)
    
    # Save aggregated metrics
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_file = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Saved aggregated metrics: {aggregated_file}")
    
    # Generate comparison figures
    print("\nGenerating comparison figures...")
    create_comparison_figures(results_dir, all_runs)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Primary metric: {aggregated['primary_metric']}")
    print(f"Best proposed: {aggregated['best_proposed']['run_id']} "
          f"(accuracy={aggregated['best_proposed']['accuracy']:.4f})")
    print(f"Best baseline: {aggregated['best_baseline']['run_id']} "
          f"(accuracy={aggregated['best_baseline']['accuracy']:.4f})")
    print(f"Gap: {aggregated['gap']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
