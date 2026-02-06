"""Tests for the Launcher class."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiment_launcher import Launcher, LauncherConfig, Sweep
from experiment_launcher.exceptions import ResultsDirException


class TestLauncherInit:
    """Tests for Launcher initialization."""

    def test_basic_init(self, basic_config: LauncherConfig):
        """Test basic launcher initialization."""
        launcher = Launcher(basic_config)
        
        assert launcher.exp_name == "test_experiment"
        assert launcher._experiment_list == []

    def test_creates_base_directory(self, basic_config: LauncherConfig, temp_dir: Path):
        """Test that launcher creates the experiment directory."""
        launcher = Launcher(basic_config)
        
        exp_dir = temp_dir / "test_experiment"
        assert exp_dir.exists()

    def test_timestamp_in_name(self, temp_dir: Path):
        """Test that timestamp is added when enabled."""
        config = LauncherConfig(
            exp_name="test",
            exp_file="module",
            n_seeds=1,
            base_dir=str(temp_dir),
            use_timestamp=True,
        )
        launcher = Launcher(config)
        
        # Name should contain timestamp pattern
        assert launcher.exp_name.startswith("test_")
        assert len(launcher.exp_name) > len("test_")


class TestAddExperiment:
    """Tests for add_experiment method."""

    def test_add_simple_experiment(self, basic_config: LauncherConfig):
        """Test adding a simple experiment without sweeps."""
        launcher = Launcher(basic_config)
        launcher.add_experiment(lr=1e-3, batch_size=32)
        
        assert len(launcher._experiment_list) == 1
        assert launcher._experiment_list[0]["lr"] == 1e-3
        assert launcher._experiment_list[0]["batch_size"] == 32

    def test_add_experiment_with_sweep(self, basic_config: LauncherConfig):
        """Test that sweeps expand to multiple experiments."""
        launcher = Launcher(basic_config)
        launcher.add_experiment(
            lr=Sweep(values=[1e-3, 1e-4]),
            batch_size=32,
        )
        
        assert len(launcher._experiment_list) == 2

    def test_add_experiment_multiple_sweeps(self, basic_config: LauncherConfig):
        """Test Cartesian product of multiple sweeps."""
        launcher = Launcher(basic_config)
        launcher.add_experiment(
            lr=Sweep(values=[1e-3, 1e-4]),
            batch_size=Sweep(values=[32, 64]),
        )
        
        assert len(launcher._experiment_list) == 4  # 2 x 2

    def test_add_multiple_experiments(self, basic_config: LauncherConfig):
        """Test adding multiple separate experiments."""
        launcher = Launcher(basic_config)
        launcher.add_experiment(lr=1e-3)
        launcher.add_experiment(lr=1e-4)
        
        assert len(launcher._experiment_list) == 2


class TestResultsDirectory:
    """Tests for results directory generation."""

    def test_generate_results_dir_no_sweep(self, basic_config: LauncherConfig):
        """Test results dir without sweeps."""
        launcher = Launcher(basic_config)
        exp = {"lr": 1e-3}
        
        result = launcher._generate_results_dir("/base", exp)
        assert result == "/base"  # No sweep params, no subdirs

    def test_generate_results_dir_with_sweep(self, basic_config: LauncherConfig):
        """Test results dir with sweep creates subdirectories."""
        launcher = Launcher(basic_config)
        exp = {
            "lr": 1e-3,
            "_sweep_params": [("lr", 1e-3)],
        }
        
        result = launcher._generate_results_dir("/base", exp)
        assert result == "/base/lr_0.001"

    def test_generate_results_dir_compact(self, temp_dir: Path):
        """Test compact directory mode."""
        config = LauncherConfig(
            exp_name="test",
            exp_file="module",
            n_seeds=1,
            base_dir=str(temp_dir),
            use_timestamp=False,
            compact_dirs=True,
        )
        launcher = Launcher(config)
        exp = {
            "lr": 1e-3,
            "_sweep_params": [("lr", 1e-3)],
        }
        
        result = launcher._generate_results_dir("/base", exp)
        assert result == "/base/0.001"  # Just value, no key

    def test_directory_clash_detection(self, basic_config: LauncherConfig):
        """Test that clashing directories are detected."""
        launcher = Launcher(basic_config)
        
        # Add same experiment twice (both have no sweep params, so same dir)
        launcher._experiment_list = [
            {"lr": 1e-3},
            {"lr": 1e-4},  # Different value but no sweep, same dir
        ]
        
        with pytest.raises(ResultsDirException):
            launcher._check_experiments_results_directories()


class TestLogDir:
    """Tests for log directory methods."""

    def test_log_dir_local(self, basic_config: LauncherConfig, temp_dir: Path):
        """Test local log directory path."""
        launcher = Launcher(basic_config)
        
        log_dir = launcher.log_dir(local=True)
        assert log_dir == str(temp_dir / "test_experiment")

    def test_log_dir_slurm(self, basic_config: LauncherConfig, temp_dir: Path):
        """Test SLURM log directory path."""
        launcher = Launcher(basic_config)
        
        # Without VSC_SCRATCH, should be same as local
        log_dir = launcher.log_dir(local=False)
        assert log_dir == str(temp_dir / "test_experiment")


class TestCleanExpParams:
    """Tests for _clean_exp_params method."""

    def test_removes_internal_keys(self, basic_config: LauncherConfig):
        """Test that internal keys are removed."""
        launcher = Launcher(basic_config)
        exp = {
            "lr": 1e-3,
            "batch_size": 32,
            "_sweep_params": [("lr", 1e-3)],
            "_internal": "data",
        }
        
        clean = launcher._clean_exp_params(exp)
        
        assert "lr" in clean
        assert "batch_size" in clean
        assert "_sweep_params" not in clean
        assert "_internal" not in clean
