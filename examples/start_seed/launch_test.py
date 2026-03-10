import time
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

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 103

if LOCAL:
    N_EXPS_IN_PARALLEL = 5
else:
    N_EXPS_IN_PARALLEL = 3

N_CORES = N_EXPS_IN_PARALLEL
MEMORY_SINGLE_JOB = 1000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'el'  # None


# E.g. SLURM only allows arrays with N_SEEDS_MAX_SLURM size
N_SEEDS_MAX_SLURM = 100
start_seed = 0
if LOCAL:
    n_seeds_split = [N_SEEDS]
else:
    if N_SEEDS <= N_SEEDS_MAX_SLURM:
        n_seeds_split = [N_SEEDS]
    elif N_SEEDS % N_SEEDS_MAX_SLURM == 0:
        n_seeds_split = [N_SEEDS_MAX_SLURM for _ in range(
            N_SEEDS // N_SEEDS_MAX_SLURM)]
    else:
        n_seeds_split = [N_SEEDS_MAX_SLURM for _ in range(
            N_SEEDS // N_SEEDS_MAX_SLURM)] + [N_SEEDS % N_SEEDS_MAX_SLURM]

for _n_seeds in n_seeds_split:
    print(f"---------> Running seeds {start_seed}-{start_seed + _n_seeds}")

    time.sleep(1)

    config = LauncherConfig(
        exp_name='test_launcher',
        exp_file='test',
        # project_name='project01234',  # for hrz cluster
        n_seeds=_n_seeds,
        start_seed=start_seed,
        resources=ResourceConfig(
            n_exps_in_parallel=N_EXPS_IN_PARALLEL,
            n_cores=N_CORES,
            memory_per_core=MEMORY_PER_CORE,
        ),
        duration=DurationConfig(
            days=2,
            hours=23,
            minutes=59,
            seconds=0,
        ),
        slurm=SlurmConfig(
            partition=PARTITION,
            gres=GRES,
        ),
        environment=dict(
            conda_env=CONDA_ENV,
        ),
        use_timestamp=True,
        compact_dirs=False
    )
    launcher = Launcher(config)

    envs = {
        'env_00': {'env_param': 'aa'},
        'env_01': {'env_param': 'bb'}
    }
    a_l = [1, 2, 3]
    boolean_param_l = [True, False]
    some_default_param = 'b'

    # These arguments are kwargs of the experiment function
    unknown_args_list = [
        dict(integer_arg=10),
        # dict(floating_arg=11.0, string_arg='test')
    ]

    for env in envs:
        d = envs[env]
        for unknown_args in unknown_args_list:
            launcher.add_experiment(
                # A subdirectory will be created for parameters using the Sweep object.
                env=Sweep(values=[env]),
                a=Sweep(values=a_l),
                boolean_param=Sweep(values=boolean_param_l),

                # This value will be overwritten by the Sweep `env` above
                # (although new API does not have priority logic, Sweeps override defaults)
                # left here to maintain similar structure to old example

                **d,
                some_default_param=some_default_param,

                **unknown_args,
                debug=False,
            )

    launcher.run(LOCAL, TEST)

    ################################################################################################################
    start_seed += _n_seeds
