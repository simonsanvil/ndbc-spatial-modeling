from pykrige.rk import Krige
from spatial_interpolation.utils.experiments import conf
from spatial_interpolation.interpolators import kriging as kr

from dataclasses import dataclass

BaseConfig = conf.Config.from_yaml("conf/kriging/parameters.yaml", as_base_cls=True)
config = conf.config_dict.ConfigDict()
# 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
@conf.as_config_dict
class ok_linear(BaseConfig):
    """
    Ordinary Kriging on the eval data
    """
    eval_set:str = "set1"
    dimensions:list = ["longitude","latitude"]
    krige:object = kr.pyKrige
    krige_params=dict(
        method="ordinary",
        variogram_model="gaussian",
        n_closest_points=5,
    )

@conf.as_config_dict
class ok_gaussian(BaseConfig):
    """
    Ordinary Kriging on the eval data
    """
    eval_set:str = "set1"
    dimensions = ["longitude","latitude"]
    krige:object = kr.SkGKrige
    krige_params=dict(
        method="ordinary",
        variogram_model="gaussian",
        n_closest_points=5,
    )

config["ordinary_kriging"] = ok_linear

def get_config():
    return config