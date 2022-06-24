import numpy as np
from scipy import interpolate

from spatial_interpolation.utils.experiments import conf
from spatial_interpolation.interpolators import IDWInterpolator

BaseConfig = conf.Config.from_yaml("conf/deterministic/parameters.yaml", as_base_cls=True)
config = conf.config_dict.ConfigDict()

@conf.as_config_dict
class idw_interpolation(BaseConfig):
    """
    Linear Barycentric Interpolation on the eval data
    """
    eval_set:str = "set1"
    interpolator:object = IDWInterpolator
    dimensions = [["longitude","latitude"]]
    interpolator_params = dict()

@conf.as_config_dict
class linear_interpolation(BaseConfig):
    """
    Linear Barycentric Interpolation on the eval data
    """
    eval_set:str = "set1"
    interpolator:object = interpolate.LinearNDInterpolator
    dimensions = [["longitude","latitude"]]
    interpolator_params = dict()

@conf.as_config_dict
class linearnd_time_interpolation(BaseConfig):
    """
    Linear Barycentric Interpolation on the eval data
    including the time dimension
    """
    eval_set:str = "set1"
    interpolator:object = interpolate.LinearNDInterpolator
    dimensions = [["longitude","latitude","time_step"]]
    interpolator_params = dict()

@conf.as_config_dict
class rbf_interpolation(BaseConfig):
    """
    Radial Basis Function Interpolation on the eval data
    with Gaussian kernel
    """
    eval_set:str = "set2"
    interpolator:object = interpolate.RBFInterpolator
    dimensions = [["longitude","latitude"]]
    interpolator_params = dict(
        epsilon=0.5, # placeholder for now
        kernel="gaussian", # placeholder for now
        # max_neighbors=100,
    )


@conf.as_config_dict
class rbf_time_interpolation(BaseConfig):
    """
    Radial Basis Function Interpolation on the eval data
    including time as the dimension
    """
    eval_set:str = "set1"
    interpolator:object = interpolate.RBFInterpolator
    dimensions = [["longitude","latitude","time_step"]]
    interpolator_params = dict(
        epsilon=0.5, # placeholder for now
        kernel="gaussian", # placeholder for now
        neighbors=100,
    )



# Make configs for each rbf kernel that we want to use
rbfs = ["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline", "cubic"]
# epsilons to search for:
epsilons = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5]
epsilon_per_kernel = {
    "gaussian": [0.5],
    "inverse_multiquadric": [0.5],
    "multiquadric": [1],
}
rbf_configs = {}
for rbf in rbfs:
    rbf_config = rbf_interpolation.copy_and_resolve_references()
    rbf_config.interpolator_params["kernel"] = rbf
    rbf_temporal_spatial = rbf_time_interpolation.copy_and_resolve_references()
    rbf_temporal_spatial.interpolator_params["kernel"] = rbf
    
    if rbf in ["gaussian", "inverse_multiquadric", "multiquadric"]:
        # for epsilon in epsilons: # TO MAKE EXPERIMENTS THAT SEARCH FOR ALL EPSILONS
        for epsilon in epsilon_per_kernel[rbf]:
            rbf_config = rbf_config.copy_and_resolve_references()
            rbf_temporal_spatial = rbf_temporal_spatial.copy_and_resolve_references()
            rbf_config.interpolator_params["epsilon"] = epsilon
            rbf_temporal_spatial.interpolator_params["epsilon"] = epsilon
            rbf_configs[f"rbf_{rbf}_eps_{str(epsilon).replace('.','_')}"] = rbf_config
            rbf_configs[f"rbf_time_{rbf}_eps_{str(epsilon).replace('.','_')}"] = rbf_temporal_spatial
    else:
        rbf_configs[f"rbf_{rbf}"] = rbf_config
        rbf_configs[f"rbf_time_{rbf}"] = rbf_temporal_spatial

# Make config for each eval set/area
for eval_set in ["set1","set2","set3"]:
    config[f"linear_{eval_set}"] = linear_interpolation.copy_and_resolve_references()
    config[f"idw_{eval_set}"] = idw_interpolation.copy_and_resolve_references()
    config[f"linear_time_{eval_set}"] = linearnd_time_interpolation.copy_and_resolve_references()
    config[f"linear_{eval_set}"].eval_set = eval_set
    config[f"idw_{eval_set}"].eval_set = eval_set
    config[f"linear_time_{eval_set}"].eval_set = eval_set
    for rbf_conf_name, rbf_config in rbf_configs.items():
        config[f"{rbf_conf_name}_{eval_set}"] = rbf_config.copy_and_resolve_references()
        config[f"{rbf_conf_name}_{eval_set}"].eval_set = eval_set

def get_config():
    return config