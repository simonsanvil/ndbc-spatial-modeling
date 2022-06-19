"""
Execute experiments.

Usage:
    experiments/__main__.py [options]
    experiments/__main__.py -h | --help

    python -m experiments - <experiment> - <experiment_config_name>

Examples:
    python -m experiments experiments.machine_learning.noaa.FeatureExtraction set1
    

"""

import click, subprocess
import dotenv
from ast import literal_eval
from spatial_interpolation.utils import get_object_from_str

from .registry import get_experiment_registry

@click.command(context_settings=dict(ignore_unknown_options=True,allow_extra_args=True))
@click.argument('experiment', type=str)
@click.argument('experiment_config', type=str, default=None, required=False)
@click.option('--suppress', is_flag=True, default=False)
@click.option('--all_configs', is_flag=True, default=False)
@click.pass_context
def run_experiment(ctx, experiment, experiment_config, suppress, all_configs):
    """
    Run experiments.
    """
    dotenv.load_dotenv()
    if not all_configs:
        print(f"Attempting to run experiment {experiment} with config {experiment_config}...")

    experiment_reg = get_experiment_registry()
    if experiment not in experiment_reg:
        if not experiment.startswith("experiments."):
            experiment = f"experiments.{experiment}"
        # get the experiment:
        Experiment = get_object_from_str(experiment)
        if Experiment is None:
            raise ValueError(f"Could not find experiment {experiment}")
    else:
        Experiment = experiment_reg[experiment]
    conf_args = {}
    # update config with extra command line arguments:
    kwargs = dict(
        kcmd.strip('--').split("=") 
        for kcmd in ctx.args 
        if kcmd.split("=")[0] not in ["experiment", "experiment_config"]
    )
    for key, value in kwargs.items():
        conf_args[key] = literal_eval(value)
    
    if all_configs:
        if conf_args:
            raise ValueError("optional arguments not valid with --all_configs")
        
        available_configs = Experiment.config.config.keys()
        print(f"Attempting to run experiment {experiment} with all {len(available_configs)} configs...")
        return run_configs(experiment,available_configs)
    
    print(f"Running experiment {experiment} with {experiment_config=} and params: {conf_args}")
    experiment = Experiment(experiment_config, verbose=not suppress, **conf_args)
    # run the experiment:
    experiment.run()

def run_configs(experiment, configs):
    for config_name in configs:
        p = subprocess.Popen(["python", "-m", "experiments", experiment , config_name])
    p.communicate()

if __name__ == "__main__":
    run_experiment()