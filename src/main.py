"""Main orchestrator for experiment runs."""

import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Orchestrate a single run.
    
    This script acts as the entry point and delegates to the appropriate
    task-specific script based on the experiment configuration.
    """
    
    print("=" * 80)
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.name}")
    print(f"Dataset: {cfg.run.dataset.name}")
    print(f"Model: {cfg.run.model.name}")
    print("=" * 80)
    
    # Determine task type from config
    # This is an inference-only task (no training)
    task_type = "inference"
    
    # Apply mode overrides
    if cfg.mode == "sanity_check":
        print("\n[SANITY CHECK MODE]")
        print("Overrides: max_samples=5, wandb.mode=online")
        # Note: overrides are applied in inference.py
    
    # Construct command
    if task_type == "inference":
        # Run inference.py as subprocess
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.inference",
            f"run={Path(cfg.run.run_id).stem}",  # Extract base name if it's a path
            f"results_dir={cfg.results_dir}",
            f"mode={cfg.mode}"
        ]
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    print(f"\nExecuting: {' '.join(cmd)}\n")
    
    # Execute subprocess
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path.cwd(),
            env=None  # Inherit environment
        )
        
        print(f"\n{'=' * 80}")
        print(f"Run {cfg.run.run_id} completed successfully")
        print(f"{'=' * 80}")
        
        sys.exit(result.returncode)
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'=' * 80}")
        print(f"Run {cfg.run.run_id} failed with exit code {e.returncode}")
        print(f"{'=' * 80}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
