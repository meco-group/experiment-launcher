"""Basic example experiment using experiment-launcher."""

import os

from experiment_launcher import single_experiment, run_experiment


@single_experiment
def experiment(
    # Experiment parameters
    env: str = "env",
    env_param: str = "aa",
    a: int = 1,
    boolean_param: bool = True,
    some_default_param: str = "b",
    debug: bool = True,
    
    # Required parameters
    seed: int = 0,
    results_dir: str = "logs",
    
    # Accept extra parameters
    **kwargs,
):
    """Example experiment function.
    
    This function demonstrates the basic structure of an experiment.
    The @single_experiment decorator handles:
    - Creating the results directory with seed subdirectory
    - Saving experiment arguments to args.json
    """
    print(f"DEBUG MODE: {debug}")
    print(f"kwargs: {kwargs}")
    
    filename = os.path.join(results_dir, f"log_{seed}.txt")
    out_str = (
        f"Running experiment with seed {seed}, env {env}, "
        f"env_param {env_param}, a {a}, "
        f"boolean_param {boolean_param}, some_default_param {some_default_param}"
    )
    print(out_str)
    
    with open(filename, "w") as file:
        file.write("Some logs in a log file.\n")
        file.write(out_str)


if __name__ == "__main__":
    run_experiment(experiment)
