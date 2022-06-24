import re
# to access dicts with dot notation
from collections import UserDict
from experiments import noaa as noaa_experiments

__registered_experiments__ = [
    noaa_experiments.FeatureExtractionExperiment,
    noaa_experiments.NOAAMLTraining,
    noaa_experiments.NOAADeterministicExperiment,
    noaa_experiments.NOAAKrigingExperiment,
    noaa_experiments.NOAARegressionKrigingExperiment,
]

snake_case_pattern_1 = re.compile('(.)([A-Z][a-z]+)')
snake_case_pattern_2 = re.compile('([a-z0-9])([A-Z])')
experiment_pattern = re.compile('(.*)([eE]xperiment)(.*)')

def get_experiment_registry():
    """
    Get all experiments from the registry
    """
    registry = {}
    experiments = __registered_experiments__
    for exp in experiments:
        exp_aliases = make_aliases(exp)
        exp_dict = {alias:exp for alias in exp_aliases}
        registry.update(exp_dict)
    return registry

def make_aliases(experiment):
    """
    Make name aliases for the Experiment 
    """
    
    cl_name = experiment.__name__
    snake = re.sub(snake_case_pattern_1, r'\1_\2', cl_name)
    snake = re.sub(snake_case_pattern_2, r'\1_\2',snake).lower()

    aliases = {
        cl_name,
        experiment.__module__,
        experiment.__module__.split("experiments.")[-1],
        experiment.__class__.__name__,
        cl_name.lower(),
        cl_name.lower().replace(" ", "_"),
        cl_name.lower().replace(" ", "-"),
        snake,
        snake.replace("_", "-"),
        snake.replace("_", ""),
        re.sub(experiment_pattern, r'\1\3', cl_name),
        re.sub(experiment_pattern, r'\1\3', experiment.__class__.__name__),
        re.sub(experiment_pattern, r'\1\3', snake).strip("_"),
    }
    if hasattr(experiment, "experiment_name"):
        name = experiment.experiment_name
        snake = re.sub(snake_case_pattern_1, r'\1_\2', name)
        snake = re.sub(snake_case_pattern_2, r'\1_\2',snake).lower().replace("-", "")
        aliases = aliases.union({
            name,
            name.lower(),
            re.sub(experiment_pattern, r'\1\3', name),
            re.sub('_|-', r'', name),
            snake,
            re.sub('_|-', r'', snake).lower(),
            snake.replace("_", ""),
            snake.replace("_", "-"),
        })
    if hasattr(experiment, "aliases"):
        aliases = aliases.union(experiment.aliases)


    return aliases