"""Tests for utility functions."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from experiment_launcher.utils import (
    create_results_dir,
    fix_random_seed,
    is_local,
    save_args,
)


class TestSaveArgs:
    """Tests for save_args function."""

    def test_save_as_json(self, temp_dir: Path):
        """Test saving args as JSON."""
        args = {"lr": 1e-3, "batch_size": 32, "model": "resnet"}

        save_args(str(temp_dir), args, save_args_as_yaml=False)

        json_path = temp_dir / "args.json"
        assert json_path.exists()

        with open(json_path) as f:
            loaded = json.load(f)

        assert loaded["lr"] == 1e-3
        assert loaded["batch_size"] == 32

    def test_save_as_yaml(self, temp_dir: Path):
        """Test saving args as YAML."""
        args = {"lr": 1e-3, "batch_size": 32}

        save_args(str(temp_dir), args, save_args_as_yaml=True)

        yaml_path = temp_dir / "args.yaml"
        assert yaml_path.exists()

        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["lr"] == 1e-3

    def test_save_with_seed_suffix(self, temp_dir: Path):
        """Test saving args with seed in filename."""
        args = {"lr": 1e-3}

        save_args(str(temp_dir), args, seed=42, save_args_as_yaml=False)

        assert (temp_dir / "args-42.json").exists()

    def test_git_info_added(self, temp_dir: Path):
        """Test that git info is added to args."""
        args = {"lr": 1e-3}

        save_args(str(temp_dir), args)

        with open(temp_dir / "args.json") as f:
            loaded = json.load(f)

        # Git info should be in saved file
        assert "git_hash" in loaded
        assert "git_url" in loaded


class TestIsLocal:
    """Tests for is_local function."""

    def test_returns_bool(self):
        """Test that is_local returns a boolean."""
        result = is_local()
        assert isinstance(result, bool)


class TestFixRandomSeed:
    """Tests for fix_random_seed function."""

    def test_sets_python_random(self):
        """Test that Python random seed is set."""
        import random

        fix_random_seed(42)
        val1 = random.random()

        fix_random_seed(42)
        val2 = random.random()

        assert val1 == val2

    def test_sets_numpy_seed(self):
        """Test that numpy seed is set."""
        import numpy as np

        fix_random_seed(42)
        val1 = np.random.random()

        fix_random_seed(42)
        val2 = np.random.random()

        assert val1 == val2


class TestCreateResultsDir:
    """Tests for create_results_dir function."""

    def test_creates_directory(self, temp_dir: Path):
        """Test that directory is created."""
        kwargs = {"results_dir": str(temp_dir / "results"), "seed": 0}

        create_results_dir(kwargs, make_dirs_with_seed=True)

        assert (temp_dir / "results" / "0").exists()
        assert kwargs["results_dir"] == str(temp_dir / "results" / "0")

    def test_creates_without_seed(self, temp_dir: Path):
        """Test directory creation without seed subdirectory."""
        kwargs = {"results_dir": str(temp_dir / "results"), "seed": 0}

        create_results_dir(kwargs, make_dirs_with_seed=False)

        assert (temp_dir / "results").exists()
        assert kwargs["results_dir"] == str(temp_dir / "results")

    def test_raises_without_results_dir(self):
        """Test that missing results_dir raises error."""
        kwargs = {"seed": 0}

        with pytest.raises(ValueError, match="results_dir must be specified"):
            create_results_dir(kwargs)
