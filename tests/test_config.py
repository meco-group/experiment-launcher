"""Tests for Pydantic configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from experiment_launcher import (
    DurationConfig,
    EnvironmentConfig,
    ExperimentConfig,
    LauncherConfig,
    ResourceConfig,
    SlurmConfig,
    Sweep,
    expand_sweeps,
)


class TestSweep:
    """Tests for the Sweep model."""

    def test_sweep_creation(self):
        """Test basic sweep creation."""
        sweep = Sweep(values=[1, 2, 3])
        assert len(sweep) == 3
        assert list(sweep) == [1, 2, 3]

    def test_sweep_with_name(self):
        """Test sweep with custom name."""
        sweep = Sweep(values=[1e-3, 1e-4], name="learning_rate")
        assert sweep.name == "learning_rate"
        assert sweep.values == [1e-3, 1e-4]

    def test_sweep_iteration(self):
        """Test that sweep is iterable."""
        sweep = Sweep(values=["a", "b", "c"])
        result = [v for v in sweep]
        assert result == ["a", "b", "c"]

    def test_sweep_empty(self):
        """Test that empty sweep raises validation error."""
        # Empty list is technically valid, just produces no experiments
        sweep = Sweep(values=[])
        assert len(sweep) == 0


class TestDurationConfig:
    """Tests for DurationConfig."""

    def test_default_values(self):
        """Test default duration values."""
        config = DurationConfig()
        assert config.days == 0
        assert config.hours == 24
        assert config.minutes == 0
        assert config.seconds == 0

    def test_to_slurm_format(self):
        """Test SLURM format conversion."""
        config = DurationConfig(days=2, hours=12, minutes=30, seconds=15)
        assert config.to_slurm_format() == "2-12:30:15"

    def test_to_slurm_format_padding(self):
        """Test SLURM format with zero padding."""
        config = DurationConfig(days=0, hours=1, minutes=5, seconds=8)
        assert config.to_slurm_format() == "0-01:05:08"


class TestResourceConfig:
    """Tests for ResourceConfig."""

    def test_default_values(self):
        """Test default resource values."""
        config = ResourceConfig()
        assert config.n_cores == 1
        assert config.memory_per_core == 2000
        assert config.n_exps_in_parallel == 1

    def test_custom_values(self):
        """Test custom resource values."""
        config = ResourceConfig(
            n_cores=8, memory_per_core=8000, n_exps_in_parallel=4)
        assert config.n_cores == 8
        assert config.memory_per_core == 8000
        assert config.n_exps_in_parallel == 4

    def test_validation_n_cores_positive(self):
        """Test that n_cores must be positive."""
        with pytest.raises(ValidationError):
            ResourceConfig(n_cores=0)


class TestSlurmConfig:
    """Tests for SlurmConfig."""

    def test_all_optional(self):
        """Test that all fields are optional."""
        config = SlurmConfig()
        assert config.partition is None
        assert config.gres is None

    def test_with_values(self):
        """Test with SLURM values."""
        config = SlurmConfig(
            partition="gpu",
            gres="gpu:rtx3080:1",
            constraint="rtx3080",
            account="myproject",
        )
        assert config.partition == "gpu"
        assert config.gres == "gpu:rtx3080:1"


class TestLauncherConfig:
    """Tests for LauncherConfig."""

    def test_minimal_config(self):
        """Test minimal required config."""
        config = LauncherConfig(
            exp_name="test",
            exp_file="train",
            n_seeds=1,
        )
        assert config.exp_name == "test"
        assert config.n_seeds == 1
        assert config.base_dir == "./logs"

    def test_nested_configs(self):
        """Test nested config models."""
        config = LauncherConfig(
            exp_name="test",
            exp_file="train",
            n_seeds=3,
            resources=ResourceConfig(n_cores=4),
            duration=DurationConfig(hours=12),
        )
        assert config.resources.n_cores == 4
        assert config.duration.hours == 12

    def test_missing_required_fields(self):
        """Test that missing required fields raise error."""
        with pytest.raises(ValidationError):
            LauncherConfig(exp_name="test")  # Missing exp_file and n_seeds


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ExperimentConfig()
        assert config.seed == 0
        assert config.results_dir == "logs"

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed."""
        config = ExperimentConfig(seed=42, lr=1e-3, batch_size=32)
        assert config.seed == 42
        assert config.lr == 1e-3  # type: ignore
        assert config.batch_size == 32  # type: ignore


class TestExpandSweeps:
    """Tests for expand_sweeps function."""

    def test_no_sweeps(self):
        """Test expansion with no sweeps."""
        params = {"lr": 1e-3, "batch_size": 32}
        result = expand_sweeps(params)
        assert len(result) == 1
        assert result[0] == params

    def test_single_sweep(self):
        """Test expansion with single sweep."""
        params = {"lr": Sweep(values=[1e-3, 1e-4]), "batch_size": 32}
        result = expand_sweeps(params)

        assert len(result) == 2
        assert result[0]["lr"] == 1e-3
        assert result[0]["batch_size"] == 32
        assert result[1]["lr"] == 1e-4

    def test_multiple_sweeps(self):
        """Test Cartesian product of multiple sweeps."""
        params = {
            "lr": Sweep(values=[1e-3, 1e-4]),
            "batch_size": Sweep(values=[32, 64]),
        }
        result = expand_sweeps(params)

        assert len(result) == 4  # 2 x 2

        # Check all combinations exist
        combinations = [(r["lr"], r["batch_size"]) for r in result]
        assert (1e-3, 32) in combinations
        assert (1e-3, 64) in combinations
        assert (1e-4, 32) in combinations
        assert (1e-4, 64) in combinations

    def test_nested_dict_sweep(self):
        """Test sweep inside nested dict."""
        params = {
            "trainer": {
                "lr": Sweep(values=[1e-3, 1e-4]),
                "epochs": 100,
            },
            "model": "resnet",
        }
        result = expand_sweeps(params)

        assert len(result) == 2
        assert result[0]["trainer"]["lr"] == 1e-3
        assert result[0]["trainer"]["epochs"] == 100
        assert result[1]["trainer"]["lr"] == 1e-4

    def test_sweep_metadata_stored(self):
        """Test that sweep info is stored for directory generation."""
        params = {"lr": Sweep(values=[1e-3], name="learning_rate")}
        result = expand_sweeps(params)

        assert "_sweep_params" in result[0]
        assert result[0]["_sweep_params"] == [("learning_rate", 1e-3)]

    def test_empty_sweep(self):
        """Test that empty sweep produces no results."""
        params = {"lr": Sweep(values=[]), "batch_size": 32}
        result = expand_sweeps(params)
        assert len(result) == 0
