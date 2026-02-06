"""Pydantic configuration models for experiment launcher."""

from __future__ import annotations

from typing import Any, Generic, TypeVar, Optional, Literal

from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")


class Sweep(BaseModel, Generic[T]):
    """Wrapper to mark a parameter for sweeping over multiple values.

    Parameters wrapped in Sweep will:
    1. Generate a separate experiment for each value
    2. Create subdirectories based on the parameter name and value

    Example:
        >>> from experiment_launcher import Sweep
        >>> # Sweep over learning rates
        >>> lr=Sweep(values=[1e-4, 1e-3])
        >>> # With custom subdir name
        >>> lr=Sweep(values=[1e-4, 1e-3], name="learning_rate")
    """

    values: list[T]
    name: str | None = None  # Custom subdir name, defaults to param key

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)


class SlurmConfig(BaseModel):
    """SLURM-specific job configuration."""

    partition: str | None = None
    gres: str | None = None
    constraint: str | None = None
    account: str | None = None
    cluster: str | None = None
    project_name: str | None = None
    begin: str | None = None  # Start time for job (--begin flag)


class DurationConfig(BaseModel):
    """Job duration/time limit configuration."""

    days: int = 0
    hours: int = 24
    minutes: int = 0
    seconds: int = 0

    @field_validator("hours", "minutes", "seconds")
    @classmethod
    def validate_time_component(cls, v: int, info) -> int:
        if info.field_name == "hours" and not (0 <= v <= 24):
            raise ValueError("hours must be between 0 and 24")
        if info.field_name in ("minutes", "seconds") and not (0 <= v <= 59):
            raise ValueError(f"{info.field_name} must be between 0 and 59")
        return v

    def to_slurm_format(self) -> str:
        """Convert to SLURM duration format: D-HH:MM:SS."""
        h = f"{self.hours:02d}"
        m = f"{self.minutes:02d}"
        s = f"{self.seconds:02d}"
        return f"{self.days}-{h}:{m}:{s}"


class ResourceConfig(BaseModel):
    """Compute resource configuration."""

    n_cores: int = Field(
        default=1, ge=1, description="Number of CPU cores per job")
    memory_per_core: int = Field(
        default=2000, ge=1, description="Memory per core in MB")
    n_exps_in_parallel: int = Field(
        default=1, ge=1, description="Experiments to run in parallel")


class EnvironmentConfig(BaseModel):
    """Environment/conda configuration."""

    conda_env: str | None = None
    initial_module_load: list[str] | None = None


class LauncherConfig(BaseModel):
    """Main launcher configuration.

    This is the primary configuration model for creating a Launcher instance.

    Example:
        >>> from experiment_launcher import LauncherConfig, Launcher
        >>> config = LauncherConfig(
        ...     exp_name="my_experiment",
        ...     exp_file="train",
        ...     n_seeds=5,
        ... )
        >>> launcher = Launcher(config)
    """

    # Required fields
    exp_name: str = Field(..., description="Name of the experiment")
    exp_file: str = Field(...,
                          description="Python module path for the experiment file")
    n_seeds: int = Field(..., ge=1,
                         description="Number of seeds to run for each config")

    # Seed configuration
    start_seed: int = Field(default=0, ge=0, description="Starting seed value")

    # Directory configuration
    base_dir: str = Field(
        default="./logs", description="Base directory for results")
    after_run_dir: str | None = Field(
        default=None, description="Directory to copy results after run")

    # Nested configs
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    duration: DurationConfig = Field(default_factory=DurationConfig)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    # Behavior flags
    use_timestamp: bool = Field(
        default=True, description="Add timestamp to experiment name")
    compact_dirs: bool = Field(
        default=False, description="Use only values for subdir names")
    check_results_directories: bool = Field(
        default=True, description="Check for directory clashes")


class WandbConfig(BaseModel):
    project: str
    entity: str
    name: str | None = None
    mode: Literal["disabled", "online", "offline"] = "online"
    group: str | None = None


class ExperimentConfig(BaseModel):
    """Base configuration for an individual experiment.

    This can be subclassed to define strongly-typed experiment parameters.
    Extra fields are allowed for flexibility.

    Example:
        >>> class MyExperimentConfig(ExperimentConfig):
        ...     lr: float = 1e-3
        ...     batch_size: int = 32
    """

    seed: int = Field(default=0, description="Random seed")
    results_dir: str = Field(default="logs", description="Results directory")
    wandb: Optional[WandbConfig] = None

    model_config = {"extra": "allow"}


def expand_sweeps(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand sweep parameters into a list of concrete configurations.

    Recursively processes nested dicts and extracts Sweep objects,
    generating the Cartesian product of all sweep values.

    Args:
        params: Dictionary potentially containing Sweep objects

    Returns:
        List of dictionaries with Sweep objects replaced by concrete values

    Example:
        >>> params = {"lr": Sweep(values=[1e-3, 1e-4]), "batch_size": 32}
        >>> expand_sweeps(params)
        [{"lr": 1e-3, "batch_size": 32}, {"lr": 1e-4, "batch_size": 32}]
    """
    # Find all sweep parameters (including nested)
    # (flat_key, sweep, display_name)
    sweep_keys: list[tuple[str, Sweep, str | None]] = []
    fixed_params: dict[str, Any] = {}

    def extract_sweeps(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Extract sweeps from dict, returning fixed params."""
        result = {}
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, Sweep):
                display_name = value.name or key
                sweep_keys.append((full_key, value, display_name))
            elif isinstance(value, dict):
                # Recurse into nested dicts
                nested = extract_sweeps(value, full_key)
                if nested:
                    result[key] = nested
            elif isinstance(value, BaseModel) and not isinstance(value, Sweep):
                # Convert Pydantic models to dicts and process
                nested = extract_sweeps(value.model_dump(), full_key)
                if nested:
                    result[key] = nested
            else:
                result[key] = value
        return result

    fixed_params = extract_sweeps(params)

    if not sweep_keys:
        return [fixed_params]

    # Generate Cartesian product of all sweeps
    from itertools import product

    sweep_values = [list(sweep.values) for _, sweep, _ in sweep_keys]
    sweep_metadata = [(key, name) for key, _, name in sweep_keys]

    expanded = []
    for combination in product(*sweep_values):
        config = _deep_copy_dict(fixed_params)
        sweep_info = []

        for (full_key, display_name), value in zip(sweep_metadata, combination):
            _set_nested_value(config, full_key, value)
            sweep_info.append((display_name, value))

        # Store sweep metadata for directory generation
        config["_sweep_params"] = sweep_info
        expanded.append(config)

    return expanded


def _deep_copy_dict(d: dict) -> dict:
    """Deep copy a dictionary."""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _deep_copy_dict(value)
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _set_nested_value(d: dict, key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-notation key."""
    keys = key.split(".")
    current = d

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    current[keys[-1]] = value
