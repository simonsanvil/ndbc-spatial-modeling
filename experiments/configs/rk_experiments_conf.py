import gstools as gs

from pykrige.rk import Krige
from spatial_interpolation.utils.experiments import conf
from spatial_interpolation.interpolators import kriging as kr

from dataclasses import dataclass

# BaseConfig = conf.Config.from_yaml("conf/kriging/parameters.yaml", as_base_cls=True)
BaseConfig = conf.Config.from_yaml("conf/ml_experiments/ndbc/train_model/parameters.yml", as_base_cls=True)
config = conf.config_dict.ConfigDict()

class rk_config(BaseConfig):
    """
    This experiments trains a ordinary kriging model on the 
    entire training dataset since 2011 to predict the wave height.
    Uses the locations at each evaluation area to evaluate the model.
    """
    n_jobs:int = 25
    eval_set:str = "set1" # placeholder
    regressor_uri: str = 'runs:/4322b34fb81a407a962eb39fd34a4225/lgbmregressor'
    interpolator:object = kr.GsKrige
    dimensions:list = ["y","x"]
    interpolator_params = dict(
        model="gaussian", # placeholder for now
        latlon=True,
        krige_params = dict(exact=True),
        model_params = dict(rescale=gs.EARTH_RADIUS)
    )
    search_parameters:dict = {
        "model": list(kr.GsKrige.gs_models.values()),
        "num_bins": [8, 10, 20, 30, 40, 50, 60],
    }




for model_name in list(kr.GsKrige.gs_models.keys()):
    for eval_set in ["set1","set2","set3"]:
        # experiment with ordinary kriging on this eval area and this model
        
        rk_config_dict = conf.as_config_dict(rk_config)
        rk_config_dict.interpolator_params["model"] = model_name
        rk_config_dict.eval_set = eval_set
        rk_config_dict.input = dict(
            eval_dir=f"data/05_model_input/ml/noaa/{eval_set}/eval/", 
            train_dir=f"data/05_model_input/ml/noaa/{eval_set}/train/")
        config[f"rk_{model_name.lower()}_{eval_set}"] = rk_config_dict
        # experiment with regression kriging on this eval area and this model
        #TODO: implement regression kriging


def get_config():
    return config