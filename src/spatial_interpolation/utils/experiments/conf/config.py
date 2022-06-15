"""
Classes and helper methods to configure the parameters of machine learning experiments.
"""

import inspect
from functools import partial, wraps
from typing import Any
import typing
from ml_collections import config_dict

from ._conf_utils import confmethod, is_confmethod

class Config:
    """
    A class to define the parameters of an experiment as a class.
    The class can be instantiated with keyword arguments to set the parameters.

    Usage:
    >>> class MyConfig(Config):
    >>>     param1:str = "default value"
    >>>     param2:str = "default value"
    >>>     param3:dict = {"type": "list", "params": {"values": [1,2,3]}}

    >>> my_experiment = MyExperiment(param1="value1", param2="value2")
    >>> my_experiment.param1
    "value1"
    >>> my_experiment.param2
    "value2"
    >>> my_experiment_dict.param3.values
    [1, 2, 3]
    >>> my_experiment.to_dict()
    {'param1': 'value1', 'param2': 'value2', 'param3': {'type': 'list', 'params': {'values': [1, 2, 3]}}}
    """

    def __init__(self,*args,**kwargs):
        if len(args)==1 and not kwargs:
            if isinstance(args[0], self.__class__):
                kwargs = args[0].config_dict
            else:
                kwargs = args[0]
        elif len(args)>1:
            raise ValueError("__init__() can only take a single positional argument")
        elif len(args)==1 and kwargs:
            raise ValueError("__init__() can only take a single positional argument or multiple keyword arguments")

        cls_fields = get_class_fields(self)
        kwargs.update(cls_fields)
        self.config_dict = config_dict.ConfigDict(kwargs)
        self.to_config_dict = lambda : self.config_dict

    def __getattribute__(self, __name: str) -> Any:
        if __name == "config_dict":
            return object.__getattribute__(self, __name)
        elif __name.startswith("_"):
            return object.__getattribute__(self, __name)
        elif __name in self.config_dict:
            return self.config_dict[__name]            
            
        return object.__getattribute__(self, __name)
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "config_dict":
            object.__setattr__(self, __name, __value)
        elif __name.startswith("_"):
            object.__setattr__(self, __name, __value)
        else:
            self.config_dict[__name] = __value
    
    def __deepcopy__(self, memo):
        return self.copy()

    @confmethod
    def copy(self):
        return type(self)(self.to_dict())

    @confmethod
    def to_dict(self) -> dict:
        """
        Convert to a dictionary.
        """
        return self.config_dict.to_dict()
    
    @classmethod
    def to_config_dict(cls, frozen=False) -> dict:
        """
        Convert this config to a `ml_collections.config_dict.ConfigDict`.

        See: https://ml-collections.readthedocs.io/en/latest/config_dict.html
        """
        return as_config_dict(cls, frozen=frozen)

    @classmethod
    def from_yaml(cls, yaml_file, *, as_base_cls=False, name=None, loader=None) -> type:
        import yaml
        name = name or cls.__name__
        if loader is None:
            loader = yaml.safe_load
        elif isinstance(loader, type):
            loader = partial(yaml.load, Loader=loader)
        with open(yaml_file, "r") as f:
            d = loader(f)
        if as_base_cls:
            return type(name, (cls,), d)
        return type(name, (cls,), {})(d)

    def to_yaml(self, yaml_file, *, dump=None, **kwargs):
        import yaml
        if dump is None:
            dump = partial(yaml.safe_dump, width=float("inf"))
        elif isinstance(dump, type):
            dump = partial(yaml.dump, Dumper=dump, width=float("inf"))
        with open(yaml_file, "w") as f:
            dump(self.to_dict(), f, **kwargs)

def as_config_dict(clss=None,*, frozen=False, _name=False):
    """
    Decorator to convert a class to a `ml_collections.config_dict.ConfigDict`.

    See: https://ml-collections.readthedocs.io/en/latest/config_dict.html
    """
    if clss is None:
        return lambda clss: as_config_dict(clss, frozen=frozen, _name=_name)
    elif not isinstance(clss, type):
        raise ValueError("clss argument must be a class")
    if _name:
        clss._name = clss.__name__
    cls_fields = get_class_fields(clss)
    for name, annotation in typing.get_type_hints(clss).items():
        if name in cls_fields: continue
        cls_fields[name] = config_dict.placeholder(annotation)
    if frozen:
        return config_dict.FrozenConfigDict(cls_fields)
    return config_dict.ConfigDict(cls_fields)

def get_class_fields(clss):
    get_the_attr = lambda x: getattr(clss, x) if isinstance(clss,type) else object.__getattribute__(clss,x)
    cls_fields = {
        k:get_the_attr(k) for k in dir(clss) 
        if not k.startswith("_") 
        and not inspect.ismethod(get_the_attr(k))
        and not is_confmethod(get_the_attr(k))
        and ((not callable(get_the_attr(k))) or (k in typing.get_type_hints(clss)))
    }
    return cls_fields