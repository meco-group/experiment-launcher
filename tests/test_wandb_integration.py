
import sys
from unittest.mock import MagicMock, patch, ANY
import pytest
from experiment_launcher.config import WandbConfig
from experiment_launcher.decorators import single_experiment
from experiment_launcher.utils import start_wandb

# Pre-mock wandb in sys.modules so local imports find it
mock_wandb_module = MagicMock()
sys.modules["wandb"] = mock_wandb_module


@pytest.fixture(autouse=True)
def reset_wandb_mock():
    mock_wandb_module.reset_mock()
    mock_wandb_module.init.reset_mock()


def test_start_wandb_with_config():
    config = WandbConfig(
        project="test_proj",
        entity="test_ent",
        mode="online",
        group="test_group",
        name="test_run"
    )
    start_wandb(config=config, param1="value1")

    # Check calling wandb.init
    mock_wandb_module.init.assert_called_with(
        mode="online",
        entity="test_ent",
        project="test_proj",
        group="test_group",
        name="test_run",
        config={"param1": "value1"},
        reinit=True,
        notes=ANY
    )


def test_start_wandb_disabled_mode():
    config = WandbConfig(
        project="test",
        entity="test",
        mode="disabled"
    )
    start_wandb(config=config)
    mock_wandb_module.init.assert_called_with(mode="disabled", reinit=True)


def test_start_wandb_none_config():
    # Should default to disabled if config is None but kwargs are passed
    start_wandb(config=None, param="val")
    # This expects the specific fallback behavior I implemented
    mock_wandb_module.init.assert_called_with(
        mode="disabled", reinit=True, config={"param": "val"})


@patch("experiment_launcher.decorators.create_results_dir")
@patch("experiment_launcher.decorators.save_args")
def test_single_experiment_with_wandb_config_object(mock_save, mock_create):
    @single_experiment
    def exp(wandb: WandbConfig = None, results_dir="."):
        pass

    wandb_cfg = WandbConfig(project="p", entity="e")
    exp(wandb=wandb_cfg, results_dir=".")

    mock_wandb_module.init.assert_called()
    call_kwargs = mock_wandb_module.init.call_args[1]
    assert call_kwargs["project"] == "p"
    assert call_kwargs["entity"] == "e"


@patch("experiment_launcher.decorators.create_results_dir")
@patch("experiment_launcher.decorators.save_args")
def test_single_experiment_with_wandb_dict(mock_save, mock_create):
    @single_experiment
    def exp(wandb: dict = None, results_dir="."):
        pass

    wandb_dict = {"project": "p_dict", "entity": "e_dict"}
    exp(wandb=wandb_dict, results_dir=".")

    mock_wandb_module.init.assert_called()
    call_kwargs = mock_wandb_module.init.call_args[1]
    assert call_kwargs["project"] == "p_dict"
    assert call_kwargs["entity"] == "e_dict"
