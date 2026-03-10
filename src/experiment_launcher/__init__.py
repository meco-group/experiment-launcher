"""Experiment Launcher - Run experiments locally or on SLURM clusters.

Example:
    >>> from experiment_launcher import Launcher, LauncherConfig, Sweep
    >>> 
    >>> config = LauncherConfig(
    ...     exp_name="my_experiment",
    ...     exp_file="train",
    ...     n_seeds=3,
    ... )
    >>> launcher = Launcher(config)
    >>> 
    >>> launcher.add_experiment(
    ...     lr=Sweep(values=[1e-3, 1e-4]),
    ...     batch_size=32,
    ... )
    >>> 
    >>> launcher.run(local=True)
"""

from experiment_launcher.config import (
    DurationConfig,
    EnvironmentConfig,
    ExperimentConfig,
    LauncherConfig,
    ResourceConfig,
    SlurmConfig,
    Sweep,
    expand_sweeps,
)
from experiment_launcher.decorators import (
    single_experiment,
    single_experiment_flat,
    single_experiment_yaml,
    single_experiment_flat_yaml
)
from experiment_launcher.launcher import Launcher, run_experiment, parse_args
from experiment_launcher.utils import is_local

__version__ = "4.0.0"

__all__ = [
    # Core classes
    "Launcher",
    # Config models
    "LauncherConfig",
    "SlurmConfig",
    "DurationConfig",
    "ResourceConfig",
    "EnvironmentConfig",
    "ExperimentConfig",
    "Sweep",
    # Functions
    "run_experiment",
    "parse_args",
    "expand_sweeps",
    "is_local",
    # Decorators
    "single_experiment",
    "single_experiment_flat",
    "single_experiment_yaml",
]
