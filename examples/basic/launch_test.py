"""Launch script for basic experiment example.

This demonstrates the new Pydantic-based API with Sweep parameters.
"""

from itertools import product

from experiment_launcher import (
    Launcher,
    LauncherConfig,
    DurationConfig,
    ResourceConfig,
    SlurmConfig,
    Sweep,
    is_local,
)

# Configuration
LOCAL = is_local()
TEST = False
USE_CUDA = False
N_SEEDS = 3

# Resource settings
if LOCAL:
    N_EXPS_IN_PARALLEL = 5
else:
    N_EXPS_IN_PARALLEL = 3

N_CORES = N_EXPS_IN_PARALLEL
MEMORY_SINGLE_JOB = 1000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES

# Create launcher with Pydantic config
config = LauncherConfig(
    exp_name="test_launcher",
    exp_file="test",
    n_seeds=N_SEEDS,
    resources=ResourceConfig(
        n_cores=N_CORES,
        memory_per_core=MEMORY_PER_CORE,
        n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    ),
    duration=DurationConfig(
        days=2,
        hours=23,
        minutes=59,
    ),
    slurm=SlurmConfig(
        partition="amd2,amd",
        gres="gpu:1" if USE_CUDA else None,
    ),
    use_timestamp=True,
    compact_dirs=False,
)

launcher = Launcher(config)

# Define experiments using Sweep for parameter sweeps
# Note: Sweep replaces the old "__" suffix for directory creation

# Sweep over environments
for env, env_config in [("env_00", {"env_param": "aa"}), ("env_01", {"env_param": "bb"})]:
    launcher.add_experiment(
        # Sweep parameters create subdirectories
        env=Sweep(values=[env]),
        a=Sweep(values=[1, 2, 3]),
        boolean_param=Sweep(values=[True, False]),
        
        # Fixed parameters
        **env_config,
        some_default_param="b",
        integer_arg=10,
        debug=False,
    )

# Alternative: Use nested sweeps for cleaner code
# launcher.add_experiment(
#     env=Sweep(values=["env_00", "env_01"]),
#     a=Sweep(values=[1, 2, 3]),
#     boolean_param=Sweep(values=[True, False]),
#     debug=False,
# )

# Run experiments
launcher.run(LOCAL, TEST)
