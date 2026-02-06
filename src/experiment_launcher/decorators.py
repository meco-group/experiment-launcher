"""Decorators for experiment functions."""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Any, Callable, TypeVar

from experiment_launcher.utils import create_results_dir, save_args, start_wandb

F = TypeVar("F", bound=Callable[..., Any])


def single_experiment(
    func: F | None = None,
    *,
    save_args_yaml: bool = False,
    use_logging: bool = False,
    make_dirs_with_seed: bool = True,
    print_exp_args: bool = False,
) -> F | Callable[[F], F]:
    """Decorator for single experiment functions.

    This decorator:
    1. Creates the results directory (optionally with seed subdirectory)
    2. Saves experiment arguments to a file
    3. Optionally sets up logging and Weights & Biases

    Args:
        func: The experiment function to decorate
        save_args_yaml: If True, save args as YAML instead of JSON
        use_logging: If True, set up file logging
        make_dirs_with_seed: If True, create seed subdirectories
        print_exp_args: If True, print experiment arguments

    Example:
        >>> @single_experiment
        ... def experiment(lr: float = 1e-3, seed: int = 0, results_dir: str = "logs"):
        ...     # Your experiment code
        ...     pass

        >>> @single_experiment(save_args_yaml=True)
        ... def experiment(lr: float = 1e-3, seed: int = 0, results_dir: str = "logs"):
        ...     pass
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wandb_run = None
            # Create results directory
            create_results_dir(kwargs, make_dirs_with_seed)

            results_dir = kwargs["results_dir"]

            # Setup logging
            if use_logging:
                logging.basicConfig(
                    level=logging.INFO,
                    filename=os.path.join(results_dir, "logfile"),
                    filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s",
                )

            # Save arguments
            save_args(
                results_dir,
                kwargs,
                git_repo_path="./",
                save_args_as_yaml=save_args_yaml,
                print_exp_args=print_exp_args,
            )

            # Start WandB if configured
            wandb_config = None
            if "wandb" in kwargs:
                wandb_val = kwargs["wandb"]
                if wandb_val is not None:
                    from experiment_launcher.config import WandbConfig
                    wandb_config = WandbConfig.model_validate(wandb_val)

            if wandb_config is not None:
                wandb_run = start_wandb(config=wandb_config, **kwargs)
            elif "wandb_mode" in kwargs:
                # Support legacy disabled mode if explicitly asked
                if kwargs["wandb_mode"] == "disabled":
                    wandb_run = start_wandb(config=None, **kwargs)

            try:
                # Run the experiment
                return fn(*args, **kwargs)
            finally:
                # Clean up WandB
                if wandb_run is not None:
                    wandb_run.finish()

        return wrapper  # type: ignore

    # Handle both @single_experiment and @single_experiment() syntax
    if func is not None:
        return decorator(func)
    return decorator


def single_experiment_yaml(func: F | None = None, **kwargs: Any) -> F | Callable[[F], F]:
    """Decorator that saves args as YAML and enables logging.

    Equivalent to @single_experiment(save_args_yaml=True, use_logging=True)
    """
    defaults = {"save_args_yaml": True,
                "use_logging": True, "print_exp_args": False}
    defaults.update(kwargs)
    return single_experiment(func, **defaults)


def single_experiment_flat(func: F | None = None, **kwargs: Any) -> F | Callable[[F], F]:
    """Decorator that doesn't create seed subdirectories.

    Equivalent to @single_experiment(make_dirs_with_seed=False)
    """
    defaults = {"make_dirs_with_seed": False}
    defaults.update(kwargs)
    return single_experiment(func, **defaults)


def single_experiment_flat_yaml(func: F | None = None, **kwargs: Any) -> F | Callable[[F], F]:
    """Decorator combining flat dirs and YAML saving.

    Equivalent to @single_experiment(make_dirs_with_seed=False, save_args_yaml=True)
    """
    defaults = {"make_dirs_with_seed": False, "save_args_yaml": True}
    defaults.update(kwargs)
    return single_experiment(func, **defaults)
