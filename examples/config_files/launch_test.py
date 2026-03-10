"""Launch script for config files example."""

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

# Create launcher
config = LauncherConfig(
    exp_name="test_launcher_configs",
    exp_file="test",
    n_seeds=N_SEEDS,
    resources=ResourceConfig(
        n_cores=N_CORES,
        memory_per_core=MEMORY_PER_CORE,
        n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    ),
    duration=DurationConfig(days=2, hours=23, minutes=59),
    slurm=SlurmConfig(
        partition="amd2,amd",
        gres="gpu:1" if USE_CUDA else None,
    ),
    use_timestamp=True,
    compact_dirs=False,
)

launcher = Launcher(config)

# Config files to sweep over
config_files = [
    "configs/config00.yaml",
    "configs/config01.yaml",
]

# Wandb options (optional)
wandb_options = {
    "wandb_mode": "disabled",  # "online", "offline" or "disabled"
    "wandb_entity": "joaocorreiacarvalho",
    "wandb_project": "test_experiment_launcher_config_files",
}

# Add experiments - sweep over config files
for i, config_file in enumerate(config_files):
    launcher.add_experiment(
        config=Sweep(values=[f"config-{str(i).zfill(2)}"], name="config"),
        config_file_path=config_file,
        debug=False,
        **wandb_options,
        wandb_group=f"test_group-el-{config_file}",
    )

# Run
launcher.run(LOCAL, TEST)
