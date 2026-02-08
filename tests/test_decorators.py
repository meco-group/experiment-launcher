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
from pydantic import BaseModel


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


class MyConfig(BaseModel):
    lr: float = 1e-3
    results_dir: str = ""
    seed: int = 0


class TrainSettings(BaseModel):
    batch_size: int = 32
    results_dir: str = ""
    seed: int = 0


class TestUseBaseModelCfg:
    """Tests for input_is_config option."""

    def test_basemodel_config_injection(self, temp_dir: Path):
        """Test that results_dir and seed are injected into BaseModel config."""
        results_dir = temp_dir / "results"
        captured = {}

        @single_experiment(input_is_config=True)
        def my_experiment(config: MyConfig):
            captured["config"] = config

        my_experiment(config=MyConfig(lr=1e-4), seed=42,
                      results_dir=str(results_dir))

        # Check that config has injected values
        assert captured["config"].seed == 42
        assert "42" in captured["config"].results_dir  # Contains seed subdir
        assert captured["config"].results_dir.endswith("/42")

    def test_basemodel_with_different_param_name(self, temp_dir: Path):
        """Test that BaseModel is detected regardless of parameter name."""
        results_dir = temp_dir / "results"
        captured = {}

        @single_experiment(input_is_config=True)
        def my_experiment(settings: TrainSettings):
            captured["settings"] = settings

        my_experiment(settings=TrainSettings(), seed=10,
                      results_dir=str(results_dir))

        assert captured["settings"].seed == 10
        assert captured["settings"].results_dir != ""

    def test_basemodel_raises_without_type_hint(self, temp_dir: Path):
        """Test that error is raised when no BaseModel type hint is found."""
        results_dir = temp_dir / "results"

        @single_experiment(input_is_config=True)
        def my_experiment(config, results_dir: str, seed: int):  # No type hint
            pass

        with pytest.raises(ValueError, match="no BaseModel or dict argument found"):
            my_experiment(config={}, seed=0, results_dir=str(results_dir))

    def test_basemodel_with_dict_config(self, temp_dir: Path):
        """Test that dict configs are also handled with BaseModel hint."""
        results_dir = temp_dir / "results"
        captured = {}

        @single_experiment(input_is_config=True)
        def my_experiment(config: MyConfig):
            captured["config"] = config

        # Pass a dict that matches the BaseModel schema
        my_experiment(config={"lr": 0.01}, seed=5,
                      results_dir=str(results_dir))

        # Config should still be injected with results_dir and seed
        assert captured["config"]["seed"] == 5
        assert captured["config"]["results_dir"] != ""

    def test_dict_type_hint(self, temp_dir: Path):
        """Test that dict type hint is supported."""
        results_dir = temp_dir / "results"
        captured = {}

        @single_experiment(input_is_config=True)
        def my_experiment(config: dict):
            captured["config"] = config

        my_experiment(config={"lr": 0.01}, seed=7,
                      results_dir=str(results_dir))

        assert captured["config"]["seed"] == 7
        assert captured["config"]["results_dir"] != ""
        assert captured["config"]["lr"] == 0.01
