"""Tests for experiment decorators."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from experiment_launcher.decorators import (
    single_experiment,
    single_experiment_flat,
    single_experiment_yaml,
)


class TestSingleExperiment:
    """Tests for single_experiment decorator."""

    def test_basic_decorator(self, temp_dir: Path):
        """Test basic decorator functionality."""
        results_dir = temp_dir / "results"
        
        @single_experiment
        def my_experiment(lr: float = 1e-3, seed: int = 0, results_dir: str = "logs"):
            return lr
        
        result = my_experiment(lr=1e-4, seed=42, results_dir=str(results_dir))
        
        # Should create seed subdirectory
        assert (results_dir / "42").exists()
        # Should save args
        assert (results_dir / "42" / "args.json").exists()

    def test_decorator_with_parentheses(self, temp_dir: Path):
        """Test decorator with explicit call."""
        results_dir = temp_dir / "results"
        
        @single_experiment()
        def my_experiment(seed: int = 0, results_dir: str = "logs"):
            pass
        
        my_experiment(seed=0, results_dir=str(results_dir))
        assert (results_dir / "0").exists()

    def test_decorator_saves_yaml(self, temp_dir: Path):
        """Test YAML saving option."""
        results_dir = temp_dir / "results"
        
        @single_experiment(save_args_yaml=True)
        def my_experiment(seed: int = 0, results_dir: str = "logs"):
            pass
        
        my_experiment(seed=0, results_dir=str(results_dir))
        assert (results_dir / "0" / "args.yaml").exists()


class TestSingleExperimentFlat:
    """Tests for single_experiment_flat decorator."""

    def test_no_seed_subdirectory(self, temp_dir: Path):
        """Test that no seed subdirectory is created."""
        results_dir = temp_dir / "results"
        
        @single_experiment_flat
        def my_experiment(seed: int = 0, results_dir: str = "logs"):
            pass
        
        my_experiment(seed=0, results_dir=str(results_dir))
        
        # Should create results dir directly, no seed subdir
        assert results_dir.exists()
        assert (results_dir / "args.json").exists()


class TestSingleExperimentYaml:
    """Tests for single_experiment_yaml decorator."""

    def test_saves_yaml(self, temp_dir: Path):
        """Test that args are saved as YAML."""
        results_dir = temp_dir / "results"
        
        @single_experiment_yaml
        def my_experiment(seed: int = 0, results_dir: str = "logs"):
            pass
        
        my_experiment(seed=0, results_dir=str(results_dir))
        assert (results_dir / "0" / "args.yaml").exists()


class TestDecoratorPreservesFunction:
    """Tests that decorators preserve function metadata."""

    def test_preserves_docstring(self):
        """Test that function docstring is preserved."""
        @single_experiment
        def my_experiment(seed: int = 0, results_dir: str = "logs"):
            """This is my experiment."""
            pass
        
        assert my_experiment.__doc__ == """This is my experiment."""

    def test_preserves_name(self):
        """Test that function name is preserved."""
        @single_experiment
        def my_experiment(seed: int = 0, results_dir: str = "logs"):
            pass
        
        assert my_experiment.__name__ == "my_experiment"
