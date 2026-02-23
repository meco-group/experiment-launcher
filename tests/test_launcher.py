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


class TestGPUPool:
    """Tests for _GPUPool class."""

    def test_acquire_release(self):
        """Test basic acquire and release."""
        from experiment_launcher.launcher import _GPUPool

        pool = _GPUPool([0, 1])
        gpu0 = pool.acquire()
        gpu1 = pool.acquire()
        assert {gpu0, gpu1} == {0, 1}

        pool.release(gpu0)
        gpu_again = pool.acquire()
        assert gpu_again == gpu0

    def test_acquire_blocks_when_empty(self):
        """Test that acquire blocks when no GPUs are available."""
        import threading
        from experiment_launcher.launcher import _GPUPool

        pool = _GPUPool([0])
        pool.acquire()  # Take the only GPU

        acquired = []

        def try_acquire():
            gpu = pool.acquire()
            acquired.append(gpu)

        t = threading.Thread(target=try_acquire)
        t.start()
        t.join(timeout=0.1)
        assert len(acquired) == 0, "Should not have acquired yet"

        pool.release(0)
        t.join(timeout=1.0)
        assert acquired == [0]

    def test_all_devices_returned(self):
        """Test that all released GPUs become available again."""
        from experiment_launcher.launcher import _GPUPool

        pool = _GPUPool([0, 1, 2])
        gpus = [pool.acquire() for _ in range(3)]
        assert set(gpus) == {0, 1, 2}

        for g in gpus:
            pool.release(g)

        gpus2 = [pool.acquire() for _ in range(3)]
        assert set(gpus2) == {0, 1, 2}


class TestResolveGPUDevices:
    """Tests for _resolve_gpu_devices."""

    def test_explicit_gpu_devices(self):
        """Test with explicitly configured GPU IDs."""
        from experiment_launcher.config import ResourceConfig
        from experiment_launcher.launcher import _resolve_gpu_devices

        res = ResourceConfig(manage_gpu_devices=True, gpu_devices=[0, 2, 3])
        assert _resolve_gpu_devices(res) == [0, 2, 3]

    def test_auto_detect_with_torch(self):
        """Test auto-detection via torch.cuda.device_count()."""
        from experiment_launcher.config import ResourceConfig
        from experiment_launcher.launcher import _resolve_gpu_devices

        res = ResourceConfig(manage_gpu_devices=True, gpu_devices=None)
        mock_torch = MagicMock()
        mock_torch.cuda.device_count.return_value = 4
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _resolve_gpu_devices(res)
            assert result == [0, 1, 2, 3]

    def test_no_gpus_raises(self):
        """Test that RuntimeError is raised when no GPUs are available."""
        from experiment_launcher.config import ResourceConfig
        from experiment_launcher.launcher import _resolve_gpu_devices

        res = ResourceConfig(manage_gpu_devices=True, gpu_devices=None)
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            import sys
            sys.modules["torch"].cuda.device_count.return_value = 0
            with pytest.raises(RuntimeError, match="no GPUs detected"):
                _resolve_gpu_devices(res)


class TestGPUDeviceConfig:
    """Tests for GPU config fields."""

    def test_resource_config_gpu_defaults(self):
        """Test ResourceConfig GPU field defaults."""
        from experiment_launcher.config import ResourceConfig

        res = ResourceConfig()
        assert res.manage_gpu_devices is False
        assert res.gpu_devices is None

    def test_resource_config_gpu_set(self):
        """Test ResourceConfig with GPU fields set."""
        from experiment_launcher.config import ResourceConfig

        res = ResourceConfig(manage_gpu_devices=True, gpu_devices=[0, 1])
        assert res.manage_gpu_devices is True
        assert res.gpu_devices == [0, 1]

    def test_experiment_config_device_default(self):
        """Test ExperimentConfig device default."""
        from experiment_launcher.config import ExperimentConfig

        cfg = ExperimentConfig()
        assert cfg.device == "cpu"

    def test_experiment_config_device_set(self):
        """Test ExperimentConfig with device set."""
        from experiment_launcher.config import ExperimentConfig

        cfg = ExperimentConfig(device="cuda:1")
        assert cfg.device == "cuda:1"


class TestGPULocalRun:
    """Tests for GPU-managed local run."""

    def test_run_local_assigns_device(self, temp_dir: Path):
        """Test that _run_local with manage_gpu assigns device to experiments."""
        from experiment_launcher.config import ResourceConfig

        config = LauncherConfig(
            exp_name="gpu_test",
            exp_file="test_module",
            n_seeds=1,
            base_dir=str(temp_dir),
            use_timestamp=False,
            resources=ResourceConfig(
                manage_gpu_devices=True,
                gpu_devices=[0, 1],
            ),
        )
        launcher = Launcher(config)
        launcher.add_experiment(lr=1e-3)

        # Track devices received by experiments
        received_devices = []

        def mock_experiment(lr=1e-3, seed=0, results_dir="logs", device="cpu", **kwargs):
            received_devices.append(device)

        mock_module = MagicMock()
        mock_module.experiment = mock_experiment

        with patch("experiment_launcher.launcher.import_module", return_value=mock_module):
            launcher.run(local=True)

        assert len(received_devices) == 1
        assert received_devices[0] in ("cuda:0", "cuda:1")
