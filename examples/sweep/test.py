import os

from experiment_launcher import run_experiment, single_experiment


@single_experiment
def experiment(
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    activation: str = 'relu',
    use_batch_norm: bool = True,

    # Required parameters
    seed: int = 0,
    results_dir: str = 'logs',
    **kwargs
):
    """
    A simple experiment function to demonstrate parameter sweeps.
    This function will be launched multiple times with different
    combinations of `learning_rate`, `batch_size`, etc.
    """
    print(f"Running experiment with seed={seed}")
    print(f"Hyperparameters: LR={learning_rate}, BS={batch_size}, "
          f"Activation={activation}, BatchNorm={use_batch_norm}")

    # Simulate saving some results
    filename = os.path.join(results_dir, f'results_seed_{seed}.txt')
    with open(filename, 'w') as f:
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Activation: {activation}\n")
        f.write(f"BatchNorm: {use_batch_norm}\n")
        f.write("Status: Completed\n")


if __name__ == '__main__':
    run_experiment(experiment)
