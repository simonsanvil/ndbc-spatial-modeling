from typing import Dict, List

import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import metrics

from spatial_interpolation import features, utils

def tweak_features(functions:Dict,*dfs:List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Apply the given functions to the given dataframes.
    
    """
    tweaked_dfs = []
    for df in dfs:
        if len(df) > 0:
            df_tweaked = features.apply_functions_to_df(df,functions)
        else:
            df_tweaked = df            
        tweaked_dfs.append(df_tweaked)
    return tweaked_dfs

def fit_estimator_with_params(estimator,X,y,X_eval,y_eval, params, fit_params):
    estimator.set_params(**params)
    estimator.fit(X, y, **fit_params)
    results={
        "estimator":estimator,
        "params":params,
        "score":r2_score(y_eval,estimator.predict(X_eval)),
        "est_best_score":estimator.best_score_["valid_0"]["rmse"]
    }
    return results

def compute_metrics(true, pred,*funcs):
    metrics_dict = dict(
        rmse=mean_squared_error(true, pred, squared=False),
        mae=mean_absolute_error(true, pred),
        r2=r2_score(true, pred),
    )
    for func in funcs:
        if isinstance(func,str):
            if hasattr(metrics,func):
                func = getattr(metrics,func)
            else:
                func = utils.get_object_from_str(func)
        metrics_dict[func.__name__] = func(true, pred)
    return pd.Series(metrics_dict)

def get_interpolations_at_time(interpolator, t, X_coords, y,  X_eval_coords, min_points=4):
    train_coords_at_time = X_coords.loc[X_coords.index.get_level_values("time") == t]
    y_train_at_time = y.loc[y.index.get_level_values("time") == t]
    train_coords_at_time = train_coords_at_time.values
    y_train_at_time = y_train_at_time.loc[y_train_at_time.index].values
    eval_coords_at_time = X_eval_coords.loc[X_eval_coords.index.get_level_values("time") == t]
    if len(eval_coords_at_time) == 0:
        return None
    if len(train_coords_at_time) < min_points:
        return None
    try:
        interpolator.fit(train_coords_at_time, y_train_at_time)
        pred = interpolator.predict(eval_coords_at_time.values)
    except Exception as err:
        print(f"Error at time {t} with coordinates:\n{eval_coords_at_time.values}\n{err}")
        raise err
    return pd.Series(pred.ravel(), index=eval_coords_at_time.index)

def infer_regression_kriging_at_time(
    model, 
    interpolator, 
    t,
    X_eval_coords,
    X_eval, 
    X, 
    X_coords, 
    y, 
    min_points=4
) -> pd.Series:
    train_coords_at_time = X_coords.loc[X_coords.index.get_level_values("time") == t]
    x_train_at_time = X.loc[X.index.get_level_values("time") == t]
    y_train_at_time = y.loc[y.index.get_level_values("time") == t].iloc[:,0]
    x_train_at_time = x_train_at_time.loc[train_coords_at_time.index]
    y_train_at_time = y_train_at_time.loc[train_coords_at_time.index].values
    train_coords_at_time = train_coords_at_time.values
    eval_coords_at_time = X_eval_coords.loc[X_eval_coords.index.get_level_values("time") == t]
    if len(eval_coords_at_time) == 0:
        return None
    if len(train_coords_at_time) < min_points:
        return None
    reg_pred_train = model.predict(x_train_at_time)
    interpolator.fit(train_coords_at_time, y=y_train_at_time-reg_pred_train)
    X_eval_at_time = X_eval.loc[X_eval.index.get_level_values("time") == t].loc[eval_coords_at_time.index]
    reg_pred_eval = model.predict(X_eval_at_time)
    krigging_residuals = interpolator.predict(eval_coords_at_time.values)
    pred = reg_pred_eval + krigging_residuals
    return pd.Series(pred.ravel(), index=eval_coords_at_time.index)


def infer_regression_krigging(model, interpolator, coords, X, y, min_points=4):
    reg_pred = model.predict(X)
    interpolator.fit(coords.values, y=y-reg_pred)
    pred = reg_pred + interpolator.predict(coords.values)
    return pred.ravel()

def search_params(experiment):
    from sklearn.model_selection import RandomizedSearchCV
    config = experiment.get_config()

    train_df = pd.concat(
        [pd.read_parquet(f"{config.input.train_dir}/{year}.parquet") for year in range(2011,2022)],
        axis=0).sort_index()
    test_df = pd.concat(
        [pd.read_parquet(f"{config.input.eval_dir}/{year}.parquet") for year in range(2011,2022)],
        axis=0).sort_index()
    X_train = train_df.drop(columns=[config.target]).copy()
    y_train = train_df[config.target]
    X_eval = test_df.drop(columns=[config.target]).copy()
    y_eval = test_df[config.target]

    X_train, X_eval = tweak_features(
        config.pretrain_funcs,
        X_train, X_eval
    )
    y_train = y_train.loc[X_train.index]
    y_eval = y_eval.loc[X_eval.index]

    target = config.target
    mod = config.model()

    parameters_to_search = config.parameters_to_search.to_dict() 
    random = RandomizedSearchCV(estimator = mod, param_distributions = parameters_to_search, n_iter = 80, cv = 2, verbose=2, random_state=42, n_jobs = 35)
    random.fit(X_train, y_train.values)
    return random