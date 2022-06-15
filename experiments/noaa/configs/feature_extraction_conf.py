"""
Defines the parameters to extract the features
for each train-eval split of the data that is used to train
the machine learning models.
"""
from spatial_interpolation.utils.experiments import conf
from experiments.evaluation import eval_sets

config = conf.config_dict.ConfigDict()
base_config = conf.Config.from_yaml("conf/ml_experiments/ndbc/feature_extraction/parameters.yml")

for set_name,val_set in eval_sets["ndbc"].items():
    val_set = conf.config_dict.ConfigDict(val_set)
    set_config_dict = base_config.copy().config_dict
    set_config_dict.split_strategy = {}
    set_config_dict.split_strategy.method = "split_slice"
    set_config_dict.split_strategy.params = {
        "eval": val_set.eval,
        "train": val_set.train if val_set.get("train") else None,
        **{k:v for k,v in val_set.items() if k not in ["eval","train"]}
    }
    set_config_dict.locations_full = val_set.locations_full
    set_config_dict.area = val_set.area
    
    set_config_dict.output = dict(
        eval_dir=f"data/05_model_input/ml/noaa/{val_set.name}/eval/", 
        train_dir=f"data/05_model_input/ml/noaa/{val_set.name}/train/")

    config[val_set.name] = set_config_dict

config["set5"] = config["set4"].copy_and_resolve_references()
config.set5.output.eval_dir = "data/05_model_input/ml/noaa/set5/eval/"
config.set5.output.train_dir = "data/05_model_input/ml/noaa/set5/train/"

def get_config(as_conf=False):
    if as_conf:
        return conf.Config(config)
    else:
        return config
    

