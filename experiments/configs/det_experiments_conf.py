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
class rbf_interpolation(BaseConfig):
    """
    Radial Basis Function Interpolation on the eval data
    with Gaussian kernel
    """
    eval_set:str = "set2"
    interpolator:object = interpolate.RBFInterpolator
    dimensions = [["longitude","latitude"]]
    interpolator_params = dict(
        epsilon=1.0,
        kernel="gaussian",
    )

# Make configs for each rbf kernel that we want to use
rbfs = ["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline", "cubic"]
# epsilons to search for:
epsilons = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5]
epsilon_per_kernel = {
    "gaussian": [0.5, 1, 0.1, 0.05],
    "inverse_multiquadric": [0.5, 1, 2, 0.05],
    "multiquadric": [5, 0.5, 1, 0.05],
}
rbf_configs = {}
for rbf in rbfs:
    if rbf in ["gaussian", "inverse_multiquadric", "multiquadric"]:
        # for i, epsilon in enumerate(epsilons): # TO MAKE EXPERIMENTS THAT SEARCH FOR ALL EPSILONS
        for epsilon in epsilon_per_kernel[rbf]:
            rbf_config = rbf_interpolation.copy_and_resolve_references()
            rbf_config.interpolator_params["kernel"] = rbf
            rbf_config.interpolator_params["epsilon"] = epsilon
            rbf_configs[f"rbf_{rbf}_eps_{str(epsilon).replace('.','_')}"] = rbf_config
    else:
        rbf_config = rbf_interpolation.copy_and_resolve_references()
        rbf_config.interpolator_params["kernel"] = rbf
        rbf_configs[f"rbf_{rbf}"] = rbf_config

for eval_set in ["set1","set2","set3"]:
    config[f"linear_{eval_set}"] = linear_interpolation.copy_and_resolve_references()
    config[f"idw_{eval_set}"] = idw_interpolation.copy_and_resolve_references()
    config[f"linear_{eval_set}"].eval_set = eval_set
    config[f"idw_{eval_set}"].eval_set = eval_set

    for rbf_conf_name, rbf_config in rbf_configs.items():
        config[f"{rbf_conf_name}_{eval_set}"] = rbf_config.copy_and_resolve_references()
        config[f"{rbf_conf_name}_{eval_set}"].eval_set = eval_set

def get_config():
    return config