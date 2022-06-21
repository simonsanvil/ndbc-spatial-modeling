"""
Defines the parameters of each experiment
"""

from spatial_interpolation.utils.experiments import conf

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

BaseConfig = conf.Config.from_yaml("conf/ml_experiments/ndbc/train_model/parameters.yml", as_base_cls=True)

class lgbm_config(BaseConfig):
    """
    This experiments trains a gradient boosting model on the 
    entire training dataset since 2011 to predict the wave height.
    Uses the locations at each evaluation area to evaluate the model.
    """
    eval_set:str = "set1" # placeholder
    model:object = lgb.LGBMRegressor
    model_params:dict = dict(
        # best params selecte from doing grid search
        learning_rate=0.075,
        n_estimators=650,
        max_depth=10,
        num_leaves=15,
        subsample_freq=30,
        subsample=0.75,
        colsample_bytree=0.8,
    )
    fit_params:dict = dict(
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=350,
                verbose=True
            ),
            lgb.log_evaluation(
                period=50,
            )
        ],
        eval_set=True,
        eval_metric="rmse",
    )

class rf_config(BaseConfig):
    """
    This experiments trains a random forest model on the
    entire training dataset since 2011 to predict the wave height.
    Uses the locations at each evaluation area to evaluate the model.
    """
    eval_set:str = "set1" # placehoder
    model:object = RandomForestRegressor
    fit_params:dict = dict(
        n_estimators=1000,
        max_depth=None,
    )
    model_params:dict = dict()
    param_search:dict = dict(
        strategy= "random",
        size=30,
        parameters_to_search = dict(
            n_estimators = [400, 500, 750, 1000, 1500], # Number of trees in random forest
            max_features = ['auto', 'sqrt'], # Number of features to consider at every split
            max_depth = [int(x) for x in np.linspace(10, 110, num = 8)], # Maximum number of levels in tree
            min_samples_split = [2, 5, 10], # Minimum number of samples required to split a node
            min_samples_leaf = [1, 2, 4], # Minimum number of samples in newly created leaf
            bootstrap = [True, False], # Method of selecting samples for training each tree
        )
    )
    search_strategy:str = "random"


config = conf.config_dict.ConfigDict()
for exp_cls in [lgbm_config, rf_config]:
    for eval_set in ["set1","set2","set3"]:
        exp_config = conf.as_config_dict(exp_cls)
        exp_cls.input = dict(
            eval_dir=f"data/05_model_input/ml/noaa/{eval_set}/eval/", 
            train_dir=f"data/05_model_input/ml/noaa/{eval_set}/train/")
        config[f"{exp_cls.__name__}_{eval_set}"] = conf.as_config_dict(exp_cls)

def get_config():
    return config