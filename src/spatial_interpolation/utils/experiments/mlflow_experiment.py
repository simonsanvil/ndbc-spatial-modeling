from functools import wraps
import json
import os, tempfile

from typing import Any
import mlflow
import yaml

from spatial_interpolation.utils.experiments import conf
from .experiment import Experiment


class MLFlowExperiment(Experiment):
    """
    An experiment that logs to MLFlow
    """

    def __init__(self, config=None, experiment_name=None, mlflow_tracking_uri=None, **kwargs):
        super().__init__(config, **kwargs)
        self._mlflow_tracking_uri = mlflow_tracking_uri or self.__dict__.get("mlflow_tracking_uri")
        self._experiment_name = experiment_name or getattr(self,"experiment_name",None)
    
    @property
    def tracking_uri(self):
        if self._mlflow_tracking_uri is not None:
            return self._mlflow_tracking_uri
        config = self.get_config()
        tracking_uri = config.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri is None:
            tracking_uri = mlflow.get_tracking_uri()
        return tracking_uri
    
    @property
    def name(self):
        if self._experiment_name is not None:
            return self._experiment_name
        config = self.get_config()
        return config.get("experiment_name") or os.environ.get("MLFLOW_EXPERIMENT_NAME", self.__class__.__name__)
    
    def get_mlflow_experiment(self):
        return mlflow.get_experiment_by_name(self.name)
        
    def pre_run(self):
        super().pre_run()
        config = self.get_config()
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.name)
        mlflow.start_run(
            run_name=getattr(self,"run_name",None), 
            description=getattr(self,"description",None) or config.get("description"),
        )

        if getattr(self,"tags",None):
            for tag_tup in self.tags:
                if len(tag_tup)==1:
                    mlflow.set_tag(tag_tup[0], "")
                else:
                    mlflow.set_tag(tag_tup[0], tag_tup[1])               
        
        if getattr(self,"params_to_log") or config.get("params_to_log"):
            params_to_log = getattr(self,"params_to_log") or config.get("params_to_log")
            for param in params_to_log:
                if isinstance(param, dict):
                    param_name = list(param.keys())[0]
                    param_value = param[param_name]
                elif isinstance(param, tuple):
                    param_name = param[0]
                    param_value = config.get(param[1]) or getattr(self, param[1])
                else:
                    param_name = param
                    param_value = config.get(param) or getattr(self, param)
                mlflow.log_param(param_name, json.dumps(param_value, default=str))
        
        self._log_config_as_yaml(config)
    
    def _log_config_as_yaml(self, config, fname="config.yaml"):
        tdir = tempfile.TemporaryDirectory()
        with open(os.path.join(tdir.name, fname), "w") as f:
            config_json = json.loads(config.to_json_best_effort())
            yaml.safe_dump(config_json, f, default_flow_style=False)
            f.seek(0); mlflow.log_artifact(f.name)
        tdir.cleanup()

    def post_run(self, result=None):
        super().post_run(result)
        try:
            mlflow.end_run()
        except Exception as e:
            self.logger.error(f"Failed to end MLFlow run: {e}")
    
    def wrap_run(self):
        """
        Wraps the run method to allow for pre- and post-run hooks.
        """
        run_func = self.run
        
        @wraps(run_func)
        def run_(*args, **kwargs):
            try:
                self.pre_run()
                result = run_func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to run experiment: {e}")
                if mlflow.active_run():
                    mlflow.end_run(status="FAILED")
                raise e
            self.post_run(result)
            return result
        
        return run_
