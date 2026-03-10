# Experiment Launcher

Launch experiments locally or on a cluster running SLURM in a single file.  
Fork from [experiment_launcher](https://git.ias.informatik.tu-darmstadt.de/common/experiment_launcher)

## Description 

The ``experiment_launcher`` package provides a simple way to run multiple experiments using SLURM or Joblib with minimum 
effort - you just have to set the `local` parameter to `True` to run locally,
or to `False` to run with a cluster using SLURM. 

It is particularly useful to run multiple seeds and/or test multiple configurations of hyperparameters such as learning rates, batch size.

## Installation

You can install the package from pip with:
```
pip install "experiment-launcher @ git+https://github.com/meco-group/experiment-launcher.git@v0.2.0"
```

If you want to install it locally, you can do so with:
```
pip install -e .
```

## How to Use

### Basic Usage

The best way to understand experiment launcher is to look at the basic example in [examples/basic](examples/basic).

- The file [examples/basic/test.py](examples/basic/test.py) contains a single experiment
- The file [examples/basic/launch_test.py](examples/basic/launch_test.py) contains the launcher file configurations and call

**Single experiment**
- [examples/basic/test.py](examples/basic/test.py) consists of:
  - The function `experiment` is the entry point of your experiment
    - It takes as arguments your experiment settings (e.g., the number of layers in a neural 
        network, the learning rate, ...)
    - The arguments need to be assigned a type and default value in the function definition
      - Current accepted types are `int`, `float`, `str`, `bool`, `list`
      - The arguments `seed` and `results_dir` **must** always be included
      - Python kwargs can also be added as `**kwargs` (accepted types are the same as above)
    - This function **must** be decorated with a decorator, e.g. `@single_experiment`. 
      This will take care of creating proper results directories.
 
  - The `if __name__ == '__main__'` block
      - This **must** contain one single line: `run_experiment(experiment)`
- You can test your code by running
    ```bash
    cd examples/basic
    python test.py
    ```

**Launch file**
- [examples/basic/launch_test.py](examples/basic/launch_test.py) consists of:
  - Creating an instance of the `LauncherConfig` object, that contains the SLURM or Joblib (if run locally) parameters. These are some of the important parameters. For more consult the class definition.
    - `exp_name` is the experiment name, under which a results directory will be created
    - `exp_file` is the path to the python file where the `experiment` is implemented (without the extension `.py`)
    - `n_seeds` is the number of random seeds for each single experiment configuration
  - Advanced configs are provided via nested Pydantic objects:
    - `ResourceConfig` specifies:
      - `n_exps_in_parallel` is the number of experiments to be run in parallel. This is useful to run multiple jobs in a single GPU in the cluster
      - `n_cores` is the number of cores for each experiment. Note that if `n_exps_in_parallel > 1`, then `n_exps_in_parallel` jobs will share `n_cores`.
      - `memory_per_core` is amount of memory in MB requested for each core in SLURM. If you specify this too low, SLURM might crash.
    - `SlurmConfig` specifies:
      - `partition` is the SLURM partition, which is cluster dependent
      - `gres` are special resources asked for a SLURM experiment
      - `project_name` is the project name in the cluster
    - `EnvironmentConfig` specifies:
      - `conda_env` if you are using a conda environment, specify its name here
    - `DurationConfig` specifies the max runtime of the SLURM job (days, hours, minutes, seconds).
  - Creating an instance of the `Launcher` object: `launcher = Launcher(config)`
  - Adding experiments with `launcher.add_experiment`
    - Use `launcher.add_experiment` to create an experiment for a particular configuration (e.g., different learning rates)
    - You can use the `Sweep` class to automatically sweep over a list of parameters. 
    - E.g. `launcher.add_experiment(learning_rate=Sweep(values=[1e-3, 1e-4]), batch_size=32)` creates two experiments: one with `1e-3` and another with `1e-4`.
    - Swept parameters automatically are used to organize the results directories. Results directories are going to be created by default as:
        - `./logs/exp_name_DATE/learning_rate_0.001/SEED/` and `./logs/exp_name_DATE/learning_rate_0.0001/SEED/`
    - If multiple sweeps are provided, the Cartesian product of all sweep values is calculated to generate the experiments.
  - Running the experiments with `launcher.run(LOCAL)`
    - This runs your experiment either locally (LOCAL: `True`) or in the cluster (LOCAL: `False`) 


**Running the experiment**
- To run the launcher simply call 
  ```bash
  cd examples/basic
  python launch_test.py
  ```
- Log files will be placed in 
  - `./logs` if running locally or in the IAS cluster
  - `/work/scratch/$USERNAME` if ran in the `Lichtenberg-Hochleistungsrechner of the TU Darmstadt` 


### Integration with Weights and Biases

The experiment launcher provides an easy way to integrate with Weights and Biases.
- In the experiment file add
  - `**kwargs` in the `experiment` function definition
- In the launcher file, create wandb options and pass them to the `launcher.add_experiment()`
  ```python
  wandb_options = dict(
    wandb_enabled=False,  # If True, runs and logs to wandb.
    wandb_entity='joaocorreiacarvalho',
    wandb_project='experiment_launcher_test',
    wandb_group='group_test'
  )
  ```
- To use wandb, you need to install it with `pip install wandb` and log in with `wandb login`
  

### Running experiments with parameters in configuration files
- If you have many parameters that you need to change, a good idea can be to use configuration files. 
- If you want to specify your parameters in a configuration file, look at the example under `examples/config_files`
  

## Notes
- For reproducibility, the seeds are created sequentially from `0` to `n_exps-1`.

