"""Utility functions for experiment launcher."""

from __future__ import annotations

import datetime
import json
import os
import random
import subprocess
from shutil import which
from typing import Any

import git
import yaml
from git import InvalidGitRepositoryError


def save_args(
    results_dir: str,
    args: dict[str, Any],
    git_repo_path: str | None = None,
    seed: int | None = None,
    save_args_as_yaml: bool = False,
    print_exp_args: bool = False,
) -> None:
    """Save experiment arguments to a file.

    Args:
        results_dir: Directory to save the args file
        args: Dictionary of arguments to save
        git_repo_path: Path to git repo for hash extraction
        seed: Optional seed for filename
        save_args_as_yaml: If True, save as YAML; otherwise JSON
        print_exp_args: If True, print the arguments
    """
    # Add git info
    args_copy = dict(args)
    try:
        repo = git.Repo(git_repo_path, search_parent_directories=True)
        args_copy["git_hash"] = repo.head.object.hexsha
        args_copy["git_url"] = repo.remotes.origin.url
    except (InvalidGitRepositoryError, ValueError):
        args_copy["git_hash"] = ""
        args_copy["git_url"] = ""

    # Convert Pydantic models to dicts for serialization
    def _make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_json_serializable(i) for i in obj]
        if hasattr(obj, "model_dump"):  # Pydantic v2
            return obj.model_dump()
        if hasattr(obj, "dict"):  # Pydantic v1
            return obj.dict()
        return obj

    serializable_args = _make_json_serializable(args_copy)

    # Save to file
    if save_args_as_yaml:
        filename = "args.yaml" if seed is None else f"args-{seed}.yaml"
        with open(os.path.join(results_dir, filename), "w") as f:
            yaml.dump(serializable_args, f, Dumper=yaml.Dumper)
    else:
        filename = "args.json" if seed is None else f"args-{seed}.json"
        with open(os.path.join(results_dir, filename), "w") as f:
            json.dump(serializable_args, f, indent=2)

    if print_exp_args:
        print("-" * 80)
        print("--------> Experiment arguments")
        print(json.dumps({k: v for k, v in serializable_args.items()
              if not k.startswith("git_")}, indent=2))
        print("-" * 80)


def is_local() -> bool:
    """Check if running locally (no SLURM available)."""
    return which("sbatch") is None


def start_wandb(
    config: "WandbConfig" | None = None,
    **kwargs: Any,
):
    """Initialize Weights & Biases run.

    Args:
        config: WandbConfig instance
        **kwargs: Additional config to log

    Returns:
        wandb.Run instance
    """
    import wandb
    from experiment_launcher.config import WandbConfig

    os.environ["WANDB__SERVICE_WAIT"] = "600"

    # If no config provided, try to create from kwargs for backward compatibility
    # or just return None/disabled if that was the intent.
    if config is None:
        return wandb.init(mode="disabled", reinit=True, config=kwargs)

    if config.mode == "disabled":
        return wandb.init(mode="disabled", reinit=True)

    return wandb.init(
        mode=config.mode,
        entity=config.entity,
        project=config.project,
        group=config.group,
        name=config.name,
        config=kwargs,
        reinit=True,
        notes=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )


def fix_random_seed(seed: int) -> None:
    """Fix random seeds for reproducibility.

    Sets seeds for random, numpy, and torch if available.

    Args:
        seed: The seed value to set
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    except ImportError:
        pass


def create_results_dir(kwargs: dict[str, Any], make_dirs_with_seed: bool = True) -> None:
    """Create results directory, optionally with seed subdirectory.

    Modifies kwargs in-place to update results_dir.

    Args:
        kwargs: Experiment kwargs containing 'results_dir' and optionally 'seed'
        make_dirs_with_seed: If True, create seed subdirectory
    """
    seed = kwargs.get("seed")
    results_dir = kwargs.get("results_dir")

    if results_dir is None:
        raise ValueError("results_dir must be specified in kwargs")

    if seed is None:
        print("Warning: No seed provided. Results will be saved without seed subdirectory.")
        make_dirs_with_seed = False

    if make_dirs_with_seed:
        results_dir = os.path.join(results_dir, str(seed))

    os.makedirs(results_dir, exist_ok=True)
    kwargs["results_dir"] = results_dir


def get_slurm_jobs_in_queue() -> int:
    """Get the number of SLURM jobs in queue (running + pending).

    Returns:
        Total number of jobs for current user
    """
    user = os.getenv("USER", "")

    running = len(
        subprocess.check_output(
            ["squeue", "-u", user, "-h", "-t", "running", "-r"],
            encoding="utf-8",
        ).splitlines()
    )
    pending = len(
        subprocess.check_output(
            ["squeue", "-u", user, "-h", "-t", "pending", "-r"],
            encoding="utf-8",
        ).splitlines()
    )

    return running + pending
