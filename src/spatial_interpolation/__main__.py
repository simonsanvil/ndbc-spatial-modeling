"""Spatial Interpolation on Environmental Data file for ensuring the package is executable
as `ml-spatial-interpolation` and `python -m spatial_interpolation`
"""
import importlib
from pathlib import Path

from kedro.framework.cli.utils import KedroCliError, load_entry_points
from kedro.framework.project import configure_project
from kedro.config import ConfigLoader

def _find_run_command(package_name):
    try:
        project_cli = importlib.import_module(f"{package_name}.cli")
        # fail gracefully if cli.py does not exist
    except ModuleNotFoundError as exc:
        if f"{package_name}.cli" not in str(exc):
            raise
        plugins = load_entry_points("project")
        run = _find_run_command_in_plugins(plugins) if plugins else None
        if run:
            # use run command from installed plugin if it exists
            return run
        # use run command from the framework project
        from kedro.framework.cli.project import run

        return run
    # fail badly if cli.py exists, but has no `cli` in it
    if not hasattr(project_cli, "cli"):
        raise KedroCliError(f"Cannot load commands from {package_name}.cli")
    return project_cli.run


def _find_run_command_in_plugins(plugins):
    for group in plugins:
        if "run" in group.commands:
            return group.commands["run"]

def get_flow_run_func(task_runner=None,**flow_params):
    '''
    To run the experiment as a prefect flow
    Returns a function that can be used as a prefect flow run function
    '''
    from prefect import flow
    from prefect.task_runners import DaskTaskRunner, ConcurrentTaskRunner

    if runner is None:
        runner = ConcurrentTaskRunner
    
    @flow(task_runner=task_runner,**flow_params)
    def run_prefect_flow():
        main()
    
    return run_prefect_flow    

def main():
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    run = _find_run_command(package_name)
    run()


if __name__ == "__main__":
    conf_loader = ConfigLoader(["conf/base", "conf/local"])
    prefect_run = False #conf_loader.get("run_as_prefect_flow")

    if prefect_run:
        from prefect.task_runners import DaskTaskRunner, ConcurrentTaskRunner
        task_runner = conf_loader.get("prefect_task_runner")
        flow_params = conf_loader.get("prefect_flow_params")
        if task_runner is None:
            task_runner = DaskTaskRunner
        elif task_runner.lower().startswith("dask"):
            task_runner = DaskTaskRunner
        elif task_runner.lower().startswith("concurrent"):
            task_runner = ConcurrentTaskRunner
        
        flow_run = get_flow_run_func(task_runner,**flow_params)
        flow_run()
    else:
        main()
