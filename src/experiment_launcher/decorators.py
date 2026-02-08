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
    input_is_config: bool = False,
) -> F | Callable[[F], F]:
    """Decorator for single experiment functions.

    This decorator:
    1. Creates the results directory (optionally with seed subdirectory)
    2. Saves experiment arguments to a file
    3. Optionally sets up logging and Weights & Biases
    4. Optionally handles BaseModel config with injected results_dir and seed

    Args:
        func: The experiment function to decorate
        save_args_yaml: If True, save args as YAML instead of JSON
        use_logging: If True, set up file logging
        make_dirs_with_seed: If True, create seed subdirectories
        print_exp_args: If True, print experiment arguments
        input_is_config: If True, experiment expects a single BaseModel config or dict.
            The decorator will add `results_dir` and `seed` to the config,
            then call the experiment as `experiment(cfg)`.

    Example:
        >>> @single_experiment
        ... def experiment(lr: float = 1e-3, seed: int = 0, results_dir: str = "logs"):
        ...     # Your experiment code
        ...     pass

        >>> @single_experiment(save_args_yaml=True)
        ... def experiment(lr: float = 1e-3, seed: int = 0, results_dir: str = "logs"):
        ...     pass

        >>> @single_experiment(input_is_config=True)
        ... def experiment(cfg: MyConfig):
        ...     # cfg.results_dir and cfg.seed are also available
        ...     pass
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wandb_run = None
            # Create results directory
            create_results_dir(kwargs, make_dirs_with_seed)

            results_dir = kwargs["results_dir"]
            seed = kwargs.get("seed")

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
                if input_is_config:
                    import inspect
                    from pydantic import BaseModel

                    # Find the BaseModel argument by checking type hints
                    cfg_key = None
                    cfg_value = None

                    # Try to resolve type hints (handles "from __future__ import annotations")
                    from typing import get_type_hints
                    try:
                        # get_type_hints might fail if types are not resolvable (e.g. local classes)
                        hints = get_type_hints(fn)
                    except Exception:
                        hints = {}

                    # Check resolved hints first
                    for param_name, hint in hints.items():
                        if param_name == "return":
                            continue
                        try:
                            if isinstance(hint, type) and (issubclass(hint, BaseModel) or issubclass(hint, dict)):
                                cfg_key = param_name
                                cfg_value = kwargs.get(param_name)
                                break
                        except TypeError:
                            pass

                    # Fallback to inspect.signature if not found (e.g. no future import or local class)
                    if cfg_key is None:
                        sig = inspect.signature(fn)
                        for param_name, param in sig.parameters.items():
                            hint = param.annotation
                            if hint is not inspect.Parameter.empty:
                                try:
                                    if isinstance(hint, type) and (issubclass(hint, BaseModel) or issubclass(hint, dict)):
                                        cfg_key = param_name
                                        cfg_value = kwargs.get(param_name)
                                        break
                                except TypeError:
                                    pass

                    if cfg_key is None:
                        raise ValueError(
                            "input_is_config=True but no BaseModel or dict argument found. "
                            "Experiment must have a parameter with a BaseModel or dict type hint."
                        )

                    # Add results_dir and seed to the config object
                    if isinstance(cfg_value, BaseModel):
                        # Create a copy with updated values
                        cfg_value = cfg_value.model_copy(
                            update={"results_dir": results_dir, "seed": seed}
                        )
                    elif isinstance(cfg_value, dict):
                        cfg_value = dict(cfg_value)
                        cfg_value["results_dir"] = results_dir
                        cfg_value["seed"] = seed
                    else:
                        raise TypeError(
                            f"Config argument must be a BaseModel or dict, got {type(cfg_value)}"
                        )

                    # Call experiment with (cfg)
                    return fn(cfg_value)
                else:
                    # Run the experiment normally
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
