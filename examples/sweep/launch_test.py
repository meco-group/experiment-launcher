from experiment_launcher import (
    Launcher,
    LauncherConfig,
    DurationConfig,
    ResourceConfig,
    SlurmConfig,
    Sweep,
    is_local,
)

LOCAL = is_local()
TEST = False

config = LauncherConfig(
    exp_name='test_sweep',
    exp_file='test',
    n_seeds=2,  # Run 2 seeds for each configuration
    resources=ResourceConfig(
        n_exps_in_parallel=4,  # Helpful to run jobs concurrently if running locally
        n_cores=1,
        memory_per_core=1000,
    ),
    duration=DurationConfig(
        days=0,
        hours=1,
        minutes=0,
        seconds=0,
    ),
    slurm=SlurmConfig(
        partition='cpu',
    ),
    use_timestamp=True,
    # Setting to False makes folder names explicit e.g. /learning_rate_0.001/ instead of /0.001/
    compact_dirs=False
)

launcher = Launcher(config)

# To configure a parameter sweep, pass a Sweep object with the list of values to `add_experiment`.
# The Cartesian product of all the Sweeps provided will be generated.
# In this example, we sweep over 2 learning rates, 2 batch sizes, and 2 activation functions.
# This generates 2 * 2 * 2 = 8 distinct configurations.
# Combined with our 2 seeds, we will launch 16 experiments in total.

launcher.add_experiment(
    learning_rate=Sweep(values=[1e-3, 1e-4]),
    batch_size=Sweep(values=[32, 64]),
    activation=Sweep(values=['relu', 'tanh']),

    # Not using Sweep for `use_batch_norm`. This will be fixed to True in all experiments.
    use_batch_norm=True,
)

print(f"Total spawned configurations: {len(launcher._experiment_list)}")

launcher.run(LOCAL, TEST)
