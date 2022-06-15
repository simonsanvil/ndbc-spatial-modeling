"""
Execute experiments.

Usage:
    experiments/__main__.py [options]
    experiments/__main__.py -h | --help

    python -m experiments - <experiment> - <experiment_config_name>

Examples:
    python -m experiments experiments.machine_learning.noaa.FeatureExtraction set1
    

"""

import click
import dotenv
from ast import literal_eval
from spatial_interpolation.utils import get_object_from_str

@click.command(context_settings=dict(ignore_unknown_options=True,allow_extra_args=True))
@click.argument('experiment', type=str)
@click.argument('experiment_config', type=str)
@click.option('--suppress', is_flag=True, default=False)
@click.pass_context
def run_experiment(ctx, experiment, experiment_config, suppress):
    """
    Run experiments.
    """
    dotenv.load_dotenv()
    print(f"Attempting to run experiment {experiment} with config {experiment_config}...")
    if not experiment.startswith("experiments."):
        experiment = f"experiments.{experiment}"
    # get the experiment:
    Experiment = get_object_from_str(experiment)
    if Experiment is None:
        raise ValueError(f"Could not find experiment {experiment}")
    conf_args = {}
    # update config with extra command line arguments:
    kwargs = dict(
        kcmd.strip('--').split("=") 
        for kcmd in ctx.args 
        if kcmd.split("=")[0] not in ["experiment", "experiment_config"]
    )
    for key, value in kwargs.items():
        conf_args[key] = literal_eval(value)
    
    print(f"Running experiment {experiment} with {experiment_config=} and params: {conf_args}")
    experiment = Experiment(experiment_config, verbose=not suppress, **conf_args)
    # run the experiment:
    experiment.run()

if __name__ == "__main__":
    run_experiment()