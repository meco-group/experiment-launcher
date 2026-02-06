"""Example using configuration files with experiment-launcher."""

import os

import yaml

from experiment_launcher import single_experiment_yaml, run_experiment


@single_experiment_yaml
def experiment(
    # Configuration file
    config_file_path: str = "./configs/config00.yaml",
    some_default_param: str = "b",
    debug: bool = True,
    
    # Required parameters
    seed: int = 0,
    results_dir: str = "logs",
    
    # Accept extra parameters (e.g., for wandb)
    **kwargs,
):
    """Experiment that loads parameters from a YAML config file.
    
    The @single_experiment_yaml decorator saves args as YAML
    and enables file logging.
    """
    print(f"DEBUG MODE: {debug}")
    
    with open(config_file_path) as f:
        configs = yaml.safe_load(f)
    
    print("Config file content:")
    print(configs)
    
    filename = os.path.join(results_dir, f"log_{seed}.txt")
    with open(filename, "w") as file:
        file.write("Some logs in a log file.\n")
        file.write(f"Running experiment with seed {seed}\n")
        file.write(f"Loaded config: {configs}\n")


if __name__ == "__main__":
    run_experiment(experiment)
