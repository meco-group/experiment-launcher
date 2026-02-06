"""Experiment Launcher - Run experiments locally or on SLURM clusters."""

from __future__ import annotations

import datetime
import inspect
import json
import os
import sys
import traceback
from copy import deepcopy
from importlib import import_module
from typing import Any

import git
import numpy as np
from joblib import Parallel, delayed

from experiment_launcher.config import (
    DurationConfig,
    EnvironmentConfig,
    LauncherConfig,
    ResourceConfig,
    SlurmConfig,
    Sweep,
    expand_sweeps,
)
from experiment_launcher.exceptions import ResultsDirException
from pydantic import validate_call


class Launcher:
    """Creates and starts experiments with Joblib (local) or SLURM (cluster).

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
        >>> # Add experiments with sweep parameters
        >>> launcher.add_experiment(
        ...     lr=Sweep(values=[1e-3, 1e-4]),
        ...     batch_size=32,
        ... )
        >>> 
        >>> launcher.run(local=True)
    """

    @validate_call
    def __init__(self, config: LauncherConfig) -> None:
        """Initialize the launcher with configuration.

        Args:
            config: LauncherConfig instance with all settings
        """
        self._config = config
        self._experiment_list: list[dict[str, Any]] = []
        self._results_dir_list: list[str] = []

        # Build experiment name with optional timestamp
        self._exp_name = config.exp_name
        if config.use_timestamp:
            self._exp_name += datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")

        # Setup directories
        self._base_dir = config.base_dir
        self._exp_dir_local = os.path.join(self._base_dir, self._exp_name)

        # SLURM directory (may be on scratch)
        self._exp_dir_slurm = self._exp_dir_local
        if os.getenv("VSC_SCRATCH"):
            scratch_dir = os.getenv("VSC_SCRATCH")
            if scratch_dir and os.path.isdir(scratch_dir):
                self._exp_dir_slurm = os.path.join(scratch_dir, self._exp_name)

        os.makedirs(self._exp_dir_slurm,
                    exist_ok=not config.check_results_directories)

        self._exp_dir_slurm_files = os.path.join(
            self._exp_dir_slurm, "slurm_files")
        self._exp_dir_slurm_logs = os.path.join(
            self._exp_dir_slurm, "slurm_logs")

    def add_experiment(self, **kwargs: Any) -> None:
        """Add one or more experiment configurations.

        Parameters wrapped in Sweep will generate multiple experiments
        (Cartesian product of all sweep values).

        Args:
            **kwargs: Experiment parameters. Use Sweep() to sweep over values.

        Example:
            >>> launcher.add_experiment(
            ...     lr=Sweep(values=[1e-3, 1e-4]),
            ...     batch_size=Sweep(values=[32, 64]),
            ...     epochs=100,  # Fixed parameter
            ... )
            # Generates 4 experiments: (1e-3, 32), (1e-3, 64), (1e-4, 32), (1e-4, 64)
        """
        expanded = expand_sweeps(kwargs)
        self._experiment_list.extend(expanded)

    def run(self, local: bool, test: bool = False, sequential: bool = False) -> None:
        """Run all added experiments.

        Args:
            local: If True, run locally with Joblib. If False, submit to SLURM.
            test: If True, only print commands without executing.
            sequential: If True and local, run experiments sequentially.
        """
        if self._config.check_results_directories:
            self._check_experiments_results_directories()

        if local:
            if sequential:
                self._run_sequential(test)
            else:
                self._run_local(test)
        else:
            self._run_slurm(test)

        self._experiment_list = []

    def _check_experiments_results_directories(self) -> None:
        """Check that no two experiments produce the same results directory."""
        for exp in self._experiment_list:
            results_dir = self._generate_results_dir(self._exp_dir_local, exp)
            if results_dir in self._results_dir_list:
                raise ResultsDirException(exp, results_dir)
            self._results_dir_list.append(results_dir)

    def _generate_results_dir(self, base_dir: str, exp: dict[str, Any]) -> str:
        """Generate the results directory path for an experiment.

        Uses sweep parameters to create subdirectories.
        """
        results_dir = base_dir

        # Use sweep metadata if available
        sweep_params = exp.get("_sweep_params", [])
        for name, value in sweep_params:
            if self._config.compact_dirs:
                subfolder = str(value)
            else:
                subfolder = f"{name}_{value}"
            subfolder = subfolder.replace("/", "-").replace(" ", "")
            results_dir = os.path.join(results_dir, subfolder)

        return results_dir

    def _run_local(self, test: bool) -> None:
        """Run experiments locally using Joblib."""
        if not test:
            os.makedirs(self._exp_dir_local, exist_ok=True)

        module = import_module(self._config.exp_file)
        experiment = module.experiment

        if test:
            self._test_experiment_local()
        else:
            def experiment_wrapper(params: dict[str, Any]) -> None:
                try:
                    experiment(**params)
                except Exception:
                    print("Experiment failed with parameters:")
                    print(params)
                    traceback.print_exc()

            default_params = _get_experiment_default_params(
                experiment, as_string=False)
            n_parallel = self._config.resources.n_exps_in_parallel

            Parallel(n_jobs=n_parallel)(
                delayed(experiment_wrapper)(deepcopy(params))
                for params in self._generate_exp_params(default_params)
            )

    def _run_sequential(self, test: bool) -> None:
        """Run experiments sequentially (single process)."""
        if not test:
            os.makedirs(self._exp_dir_local, exist_ok=True)

        module = import_module(self._config.exp_file)
        experiment = module.experiment

        if test:
            self._test_experiment_local()
        else:
            default_params = _get_experiment_default_params(
                experiment, as_string=False)

            for params in self._generate_exp_params(default_params):
                try:
                    experiment(**params)
                except Exception:
                    print("Experiment failed with parameters:")
                    print(params)
                    traceback.print_exc()

    def _run_slurm(self, test: bool) -> None:
        """Submit experiments to SLURM cluster."""
        os.makedirs(self._exp_dir_slurm_files, exist_ok=True)
        os.makedirs(self._exp_dir_slurm_logs, exist_ok=True)

        n_parallel = self._config.resources.n_exps_in_parallel
        slurm_files: list[str] = []

        # Chunk experiments for parallel execution within SLURM jobs
        for i in range(0, len(self._experiment_list), n_parallel):
            chunk = self._experiment_list[i:i + n_parallel]
            command_lines = []

            for exp in chunk:
                clean_exp = self._clean_exp_params(exp)
                cmd_args = _convert_to_command_line(clean_exp)
                results_dir = self._generate_results_dir(
                    self._exp_dir_slurm, exp)
                command_lines.append(f"--results_dir {results_dir} {cmd_args}")

            slurm_files.extend(
                self._save_slurm(command_lines, str(i).zfill(
                    len(str(len(self._experiment_list)))))
            )

        for slurm_file in slurm_files:
            command = f"sbatch {slurm_file}"
            if test:
                print(command)
            else:
                os.system(command)

    def _generate_slurm_script(self, command_lines: list[str]) -> str:
        """Generate a SLURM batch script."""
        cfg = self._config
        slurm = cfg.slurm
        res = cfg.resources
        env = cfg.environment

        # Build optional SBATCH directives
        directives = []
        if slurm.project_name:
            directives.append(f"#SBATCH -A {slurm.project_name}")
        if slurm.partition:
            directives.append(f"#SBATCH -p {slurm.partition}")
        if slurm.begin:
            directives.append(f"#SBATCH --begin={slurm.begin}")
        if slurm.gres:
            directives.append(f"#SBATCH --gres={slurm.gres}")
        if slurm.constraint:
            directives.append(f"#SBATCH --constraint={slurm.constraint}")
        if slurm.account:
            directives.append(f"#SBATCH --account={slurm.account}")
        if slurm.cluster:
            directives.append(f"#SBATCH --cluster={slurm.cluster}")

        optional_directives = "\n".join(directives)
        if optional_directives:
            optional_directives += "\n"

        # Environment setup
        env_setup = ""
        if env.conda_env:
            home = os.getenv("HOME", "")
            if os.path.exists(f"{home}/miniconda3"):
                env_setup = f'eval "$({home}/miniconda3/bin/conda shell.bash hook)"\n'
            elif os.path.exists(f"{home}/anaconda3"):
                env_setup = f'eval "$({home}/anaconda3/bin/conda shell.bash hook)"\n'
            else:
                raise RuntimeError(
                    "No miniconda3 or anaconda3 found in home directory")
            env_setup += f"conda activate {env.conda_env}\n"
            python_cmd = f"python {self._exp_file_path}"
        else:
            env_setup = f"source {sys.prefix}/bin/activate\n"
            repo_dir = git.Repo(
                search_parent_directories=True).working_tree_dir
            python_cmd = f"PYTHONPATH={repo_dir} python {self._exp_file_path}"

        module_loads = ""
        if env.initial_module_load:
            module_loads = "\n".join(
                f"module load {m}" for m in env.initial_module_load) + "\n"

        # Build commands
        commands = []
        for cmd in command_lines:
            seed_part = ""
            commands.append(f"{python_cmd} \\\n\t\t{seed_part}{cmd} &")

        commands_str = "\n\n".join(commands)
        duration = cfg.duration.to_slurm_format()

        script = f"""\
#!/usr/bin/env bash

###############################################################################
# SLURM Configurations
{optional_directives}#SBATCH -J {self._exp_name}
#SBATCH -a 0-0
#SBATCH -t {duration}
#SBATCH --ntasks 1
#SBATCH --cpus-per-task {res.n_cores}
#SBATCH --mem-per-cpu={res.memory_per_core}
#SBATCH -o {self._exp_dir_slurm_logs}/%A_%a.out
#SBATCH -e {self._exp_dir_slurm_logs}/%A_%a.err

###############################################################################
# Setup
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

{module_loads}module load Python
{env_setup}

###############################################################################
# Run experiments
echo "Running experiments in parallel..."
echo "########################################################################"

{commands_str}

wait
echo "########################################################################"
echo "All experiments finished."
"""

        # Copy results after run if configured
        if cfg.after_run_dir:
            repo_name = git.Repo(
                search_parent_directories=True).working_tree_dir.split("/")[-1]
            local_dir = os.path.join(
                cfg.after_run_dir, repo_name, self._base_dir)
            os.makedirs(local_dir, exist_ok=True)
            script += f"\ncp -r {self._exp_dir_slurm}/ {local_dir}/\n"

        script += '\necho "...done."\n'
        return script

    def _save_slurm(self, command_lines: list[str], idx: str) -> list[str]:
        """Save SLURM script to file and return paths."""
        cfg = self._config

        # Generate scripts for each seed if not using array tasks
        scripts = []
        cmd_batch = []

        for cmd in command_lines:
            for seed in range(cfg.start_seed, cfg.start_seed + cfg.n_seeds):
                cmd_batch.append(f"--seed {seed} {cmd}")

                if len(cmd_batch) >= cfg.resources.n_exps_in_parallel:
                    scripts.append(self._generate_slurm_script(cmd_batch))
                    cmd_batch = []

        if cmd_batch:
            scripts.append(self._generate_slurm_script(cmd_batch))

        # Save scripts
        paths = []
        for i, script in enumerate(scripts):
            filename = f"slurm_{self._exp_name}_{idx}_{i}.sh"
            path = os.path.join(self._exp_dir_slurm_files, filename)
            with open(path, "w") as f:
                f.write(script)
            paths.append(path)

        return paths

    def _generate_exp_params(self, default_params: dict[str, Any]):
        """Generate experiment parameters for all seeds and configs."""
        cfg = self._config
        seeds = np.arange(cfg.start_seed, cfg.start_seed + cfg.n_seeds)

        for exp in self._experiment_list:
            params = deepcopy(default_params)
            clean_exp = self._clean_exp_params(exp)
            params.update(clean_exp)
            params["results_dir"] = self._generate_results_dir(
                self._exp_dir_local, exp)

            for seed in seeds:
                params["seed"] = int(seed)
                yield deepcopy(params)

    def _clean_exp_params(self, exp: dict[str, Any]) -> dict[str, Any]:
        """Remove internal metadata from experiment params."""
        return {k: v for k, v in exp.items() if not k.startswith("_")}

    def _test_experiment_local(self) -> None:
        """Print experiment commands for testing."""
        for exp in self._experiment_list:
            results_dir = self._generate_results_dir(self._exp_dir_local, exp)
            clean_exp = self._clean_exp_params(exp)

            for seed in range(self._config.start_seed, self._config.n_seeds):
                params_str = ", ".join(
                    f"{k}={v!r}" for k, v in clean_exp.items())
                print(
                    f"experiment({params_str}, seed={seed}, results_dir={results_dir!r})")

    @property
    def exp_name(self) -> str:
        """Get the experiment name (with timestamp if enabled)."""
        return self._exp_name

    def log_dir(self, local: bool = True) -> str:
        """Get the log directory path."""
        return self._exp_dir_local if local else self._exp_dir_slurm

    @property
    def _exp_file_path(self) -> str:
        """Get the absolute path to the experiment file."""
        module = import_module(self._config.exp_file)
        return module.__file__


# ==============================================================================
# Utility functions
# ==============================================================================


def _get_experiment_default_params(func, as_string: bool = True) -> dict[str, Any]:
    """Extract default parameters from experiment function signature."""
    signature = inspect.signature(func)
    defaults = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            if as_string:
                defaults[k] = json.dumps(v.default, separators=(",", ":"))
            else:
                defaults[k] = v.default
    return defaults


def _convert_to_command_line(exp: dict[str, Any]) -> str:
    """Convert experiment dict to command line arguments."""
    parts = []
    for key, value in exp.items():
        json_value = json.dumps(value, separators=(",", ":"))
        parts.append(f"--{key} '{json_value}'")
    return " ".join(parts)


def _has_kwargs(func) -> bool:
    """Check if function accepts **kwargs."""
    signature = inspect.signature(func)
    return any(v.kind == v.VAR_KEYWORD for v in signature.parameters.values())


def parse_args(func) -> dict[str, Any]:
    """Parse command line arguments for an experiment function."""
    import argparse

    parser = argparse.ArgumentParser()

    # Add experiment parameters from function signature
    signature = inspect.signature(func)
    for k, v in signature.parameters.items():
        if k not in ("seed", "results_dir") and v.default is not inspect.Parameter.empty:
            json_default = json.dumps(v.default, separators=(",", ":"))
            parser.add_argument(f"--{k}", type=str, default=json_default)

    # Add base args
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results_dir", type=str)

    # Parse
    if _has_kwargs(func):
        args, unknown = parser.parse_known_args()
        kwargs = _parse_unknown_args(unknown)
        result = vars(args)
        result.update(kwargs)
    else:
        args = parser.parse_args()
        result = vars(args)

    # Decode JSON strings
    for k, v in list(result.items()):
        if isinstance(v, str):
            try:
                result[k] = json.loads(v)
            except json.JSONDecodeError:
                pass

    return result


def _parse_unknown_args(unknown: list[str]) -> dict[str, Any]:
    """Parse unknown command line arguments."""
    kwargs = {}
    key_idxs = [i for i, arg in enumerate(unknown) if arg.startswith("--")]

    if not key_idxs:
        return kwargs

    key_n_args = [
        key_idxs[i + 1] - 1 - key_idxs[i] if i < len(key_idxs) - 1
        else len(unknown) - 1 - key_idxs[i]
        for i in range(len(key_idxs))
    ]

    for i, idx in enumerate(key_idxs):
        key = unknown[idx][2:]
        n_args = key_n_args[i]

        if n_args > 1:
            kwargs[key] = [_string_to_primitive(
                unknown[idx + 1 + j]) for j in range(n_args)]
        elif n_args == 1:
            kwargs[key] = _string_to_primitive(unknown[idx + 1])

    return kwargs


def _string_to_primitive(string: str) -> int | float | bool | str:
    """Convert string to appropriate primitive type."""
    # Try int
    try:
        return int(string)
    except ValueError:
        pass

    # Try float
    try:
        return float(string)
    except ValueError:
        pass

    # Try bool
    lower = string.lower()
    if lower in ("true", "yes", "1"):
        return True
    if lower in ("false", "no", "0"):
        return False

    return string


def run_experiment(func, args: dict[str, Any] | None = None) -> None:
    """Run an experiment function with parsed or provided arguments.

    Args:
        func: The experiment function to run
        args: Optional pre-parsed arguments. If None, parses from command line.
    """
    if args is None:
        args = parse_args(func)
    func(**args)
