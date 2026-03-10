"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest

from experiment_launcher import (
    DurationConfig,
    EnvironmentConfig,
    LauncherConfig,
    ResourceConfig,
    SlurmConfig,
    Sweep,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_config(temp_dir: Path) -> LauncherConfig:
    """Create a basic launcher config for testing."""
    return LauncherConfig(
        exp_name="test_experiment",
        exp_file="test_module",
        n_seeds=2,
        base_dir=str(temp_dir),
        use_timestamp=False,  # Disable for predictable paths
    )


@pytest.fixture
def full_config(temp_dir: Path) -> LauncherConfig:
    """Create a full launcher config with all options."""
    return LauncherConfig(
        exp_name="full_test",
        exp_file="test_module",
        n_seeds=3,
        start_seed=10,
        base_dir=str(temp_dir),
        resources=ResourceConfig(
            n_cores=4,
            memory_per_core=4000,
            n_exps_in_parallel=2,
        ),
        duration=DurationConfig(
            days=1,
            hours=12,
            minutes=30,
            seconds=0,
        ),
        slurm=SlurmConfig(
            partition="gpu",
            gres="gpu:1",
            account="myproject",
        ),
        environment=EnvironmentConfig(
            conda_env="myenv",
            initial_module_load=["Python", "CUDA"],
        ),
        use_timestamp=False,
        compact_dirs=True,
    )


@pytest.fixture
def sample_experiment_params() -> dict[str, Any]:
    """Sample experiment parameters with sweeps."""
    return {
        "lr": Sweep(values=[1e-3, 1e-4]),
        "batch_size": Sweep(values=[32, 64]),
        "epochs": 100,
        "model": "resnet",
    }


@pytest.fixture
def mock_experiment_module(temp_dir: Path, monkeypatch):
    """Create a mock experiment module."""
    # Create a simple experiment module
    module_content = '''
def experiment(lr=1e-3, batch_size=32, epochs=100, seed=0, results_dir="logs", **kwargs):
    """Mock experiment function."""
    import os
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "result.txt"), "w") as f:
        f.write(f"lr={lr}, batch_size={batch_size}, epochs={epochs}, seed={seed}")
'''
    module_path = temp_dir / "test_module.py"
    module_path.write_text(module_content)
    
    # Add to path so it can be imported
    import sys
    sys.path.insert(0, str(temp_dir))
    
    yield "test_module"
    
    # Cleanup
    sys.path.remove(str(temp_dir))
