"""
An experiment is an abstract class to define 
the pipeline and parameters of an experiment as a class
that can be used to run the experiment.
"""

from ast import literal_eval
import logging, importlib
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from types import ModuleType

from spatial_interpolation.utils.experiments import conf
from ml_collections import config_dict

class Experiment(ABC):
    """
    Abstract class to define the pipeline and parameters of an experiment.

    Parameters
    ----------
    cfg : str or dict or Config or ConfigDict
        The configuration of the experiment.
    verbose : bool
        Whether to print the logs.
    kwards : dict
        Additional parameters to be passed to the config.

    Usage:
    -------
    class MyExperiment(Experiment):
        config = my_config_file_or_module

        def run(self):
            self.pipe.run()
    
    >>> exp = MyExperiment(config)
    >>> exp.run()
    """

    def __init__(self, cfg=None, verbose=True, **config_kwargs):
        self._cfg = cfg
        self.config_kwargs = config_kwargs
        # change the run function to the one that allows for pre- and post-run hooks
        self.run = self.wrap_run()
        self.logger = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
        self.verbose = verbose
    
    def get_config(self, cfg=None, as_conf=False):
        self.pre_config()
        cfg = cfg or self._cfg
        if cfg is None and not hasattr(self, "config") and not hasattr(self.__class__, "config"):
            raise ValueError("Either config or a config must be provided.")
        
        if hasattr(self, "config") and isinstance(self.config, ModuleType):
            importlib.reload(self.config)

        if cfg is None:
            if isinstance(self.config,str) and Path(self.config).is_file():
                config_ = conf.Config.from_yaml(self.config).to_config_dict()
            elif hasattr(self.config, "get_config"):
                config_ = self.config.get_config()
            elif isinstance(self.config, config_dict.ConfigDict) or isinstance(self.config, conf.Config):
                config_ = self.config
            elif isinstance(self.config,dict):
                config_ = config_dict.ConfigDict(self.config)
            elif isinstance(self.config, type):
                config_ = self.config()
            else:
                raise ValueError("config is not a valid config and config was not provided at instance time.")
        else:
            if isinstance(cfg, dict):
                config_ = config_dict.ConfigDict(cfg).config_dict
            elif isinstance(cfg, str) and Path(cfg).is_file():
                config_ = conf.Config.from_yaml(cfg).config_dict
            elif isinstance(cfg, str) and hasattr(self.config, cfg):
                config_ = getattr(self.config, cfg)
            elif isinstance(cfg, str) and hasattr(self.config, "get_config"):
                configs = self.config.get_config()
                if configs.get(cfg):
                    config_ = configs[cfg]     
                else:
                    raise ValueError("{} is not a valid config name for config {}".format(cfg, self.config))
            elif isinstance(cfg, conf.Config) or isinstance(cfg, config_dict.ConfigDict):
                config_ = cfg
            else:
                raise ValueError("Invalid cfg given")

        if isinstance(config_, conf.Config):
            config_ = config_.config_dict
        elif isinstance(config_, type) and hasattr(config_, "to_config_dict"):
            config_ = config_.to_config_dict()
        elif isinstance(config_, dict):
            config_ = config_dict.ConfigDict(config_)
        
         # set the configs of config_kwargs:
        for k,v in self.config_kwargs.items():
            self._set_conf_params(config_, k, v)

        if as_conf:
            if isinstance(self.config, type):
                return self.config(config_)
            return conf.Config(config_)

        return config_
    
    def set_config_param(self, param, value):
        """
        Set a config parameter.
        """
        self.config_kwargs[param] = value
    
    def _set_conf_params(self, conf_dict, param, value):
        """
        Set a config parameter.
        """
        if "." in param:
            conf_d = conf_dict
            for k in param.split(".")[:-1]:
                conf_d = conf_d[k]
            conf_d[param.split(".")[-1]] = value
        else:
            conf_dict[param] = value
    
    @property
    def experiment_name(self):
        return self.__class__.__name__

    @abstractmethod
    def run(self):
        """
        Run the experiment.
        """
        raise NotImplementedError("The run method must be implemented.")
        # self.pipe.run(), self.pipe.transform(), ...
    
    def pre_config(self):
        """
        Called before the config is loaded.
        """
        pass
    
    def pre_run(self):
        """
        Called before the experiment is run.
        """
        self.logger.info("Running experiment \"{}\"...".format(self.experiment_name))
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)
        pass

    def post_run(self, result=None):
        """
        Called after the experiment is run.
        """
        self.logger.info("Finished running experiment \"{}\"".format(self.experiment_name))
        pass

    def wrap_run(self):
        """
        Wraps the run method to allow for pre- and post-run hooks.
        """

        run_func = self.run

        @wraps(run_func)
        def run_(*args, **kwargs):
            self.pre_run()
            result = run_func(*args, **kwargs)
            self.post_run(result)
            return result
        
        return run_

