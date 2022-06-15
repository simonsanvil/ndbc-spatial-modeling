"""
Defines the parameters of each experiment
"""

from spatial_interpolation.utils.experiments import conf

import lightgbm as lgb

BaseConfig = conf.Config.from_yaml("conf/ml_experiments/ndbc/train_model/parameters.yml", as_base_cls=True)

class experiment1(BaseConfig):
    """
    This experiments trains a gradient boosting model on the 
    entire training dataset since 2011 to predict the wave height.
    Uses the locations at evaluation set 1 to evaluate the model.
    """
    eval_set:str = "set1"
    model:object = lgb.LGBMRegressor
    fit_params:dict = dict(
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=150,
                verbose=True
            ),
            lgb.log_evaluation(
                period=25,
            )
            
        ],
        eval_set=True,
        eval_metric="rmse",
    )


class experiment2(BaseConfig):
    """
    This experiments trains a gradient boosting model on the 
    entire training dataset since 2011 to predict the wave height.
    Uses the locations at evaluation set 2 to evaluate the model.
    """
    eval_set = "set2"
    model:object = lgb.LGBMRegressor
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
    
class experiment3(BaseConfig):
    """
    This experiments trains a gradient boosting model on the 
    entire training dataset since 2011 to predict the wave height.
    Uses the locations at evaluation set 2 to evaluate the model.
    """
    eval_set = "set3"
    model:object = lgb.LGBMRegressor
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

class experiment4(BaseConfig):
    """
    This experiments trains a gradient boosting model on the 
    entire training dataset since 2011 to predict the wave height.
    Uses the locations at evaluation set 2 to evaluate the model.
    """
    eval_set = "set4"
    model:object = lgb.LGBMRegressor
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

# class experiment4(BaseConfig):
#     """
#     This experiments trains a gradient boosting model on the 
#     entire training dataset since 2011 to predict the wave height.
#     Uses the locations at evaluation set 2 to evaluate the model.
#     """
#     eval_set = "set1"

class experiment5(BaseConfig):
    """
    This experiments trains a gradient boosting model on the 
    entire training dataset since 2011 to predict the wave height.
    Uses the locations at evaluation set 2 to evaluate the model.
    """
    eval_set = "set2"

class experiment6(BaseConfig):
    """
    This experiments trains a gradient boosting model on the 
    entire training dataset since 2011 to predict the wave height.
    Uses the locations at evaluation set 2 to evaluate the model.
    """
    eval_set = "set3"

config = conf.config_dict.ConfigDict()
for exp_cls in [experiment1, experiment2, experiment3, experiment4, experiment5, experiment6]:
    exp_cls.input = dict(
        eval_dir=f"data/05_model_input/ml/noaa/{exp_cls.eval_set}/eval/", 
        train_dir=f"data/05_model_input/ml/noaa/{exp_cls.eval_set}/train/")

    config[exp_cls.__name__] = conf.as_config_dict(exp_cls)

def get_config():
    return config