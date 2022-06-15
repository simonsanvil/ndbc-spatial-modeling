from spatial_interpolation.utils.experiments import conf
from scipy import interpolate

BaseConfig = conf.Config.from_yaml("conf/deterministic/parameters.yaml", as_base_cls=True)
config = conf.config_dict.ConfigDict()

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
class rbf_interpolation_gauss(BaseConfig):
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

@conf.as_config_dict
class rbf_interpolation_multiquadric(BaseConfig):
    """
    Radial Basis Function Interpolation on the eval data
    with Multiquadric kernel
    """
    eval_set:str = "set2"
    interpolator:object = interpolate.RBFInterpolator
    dimensions = [["longitude","latitude"]]
    interpolator_params = dict(
        epsilon=1.0,
        kernel='multiquadric',
    )

@conf.as_config_dict
class rbf_interpolation_linear(BaseConfig):
    """
    Radial Basis Function Interpolation on the eval data
    with Linear kernel
    """
    eval_set:str = "set2"
    interpolator:object = interpolate.RBFInterpolator
    dimensions = [["longitude","latitude"]]
    interpolator_params = dict(
        epsilon=1.0,
        kernel='linear',
    )

@conf.as_config_dict
class rbf_interpolation_inverse_multiquadric(BaseConfig):
    """
    Radial Basis Function Interpolation on the eval data
    with Inverse Multiquadric kernel
    """
    eval_set:str = "set2"
    interpolator:object = interpolate.RBFInterpolator
    dimensions = [["longitude","latitude"]]
    interpolator_params = dict(
        epsilon=1.0,
        kernel='inverse_multiquadric',
    )
    


for eval_set in ["set1","set2","set3","set4"]:
    config[f"linear_{eval_set}"] = linear_interpolation.copy_and_resolve_references()
    config[f"linear_{eval_set}"].eval_set = eval_set
    config[f"rbf_gauss_{eval_set}"] = rbf_interpolation_gauss.copy_and_resolve_references()
    config[f"rbf_gauss_{eval_set}"].eval_set = eval_set
    config[f"rbf_mult_{eval_set}"] = rbf_interpolation_multiquadric.copy_and_resolve_references()
    config[f"rbf_mult_{eval_set}"].eval_set = eval_set
    config[f"rbf_linear_{eval_set}"] = rbf_interpolation_linear.copy_and_resolve_references()
    config[f"rbf_linear_{eval_set}"].eval_set = eval_set
    config[f"rbf_invmult_{eval_set}"] = rbf_interpolation_inverse_multiquadric.copy_and_resolve_references()
    config[f"rbf_invmult_{eval_set}"].eval_set = eval_set

def get_config():
    return config