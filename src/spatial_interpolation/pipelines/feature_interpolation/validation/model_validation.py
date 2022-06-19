from functools import partial
import logging
import os
import tempfile
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from tqdm import tqdm
from spatial_interpolation import utils
from spatial_interpolation import features

def evaluate_model(
    model,
    X_eval,
    y_eval,
    X_train,
    y_train,
    evaluation_params:dict=None,
    log_mlflow:bool=True,
    log_shap:bool=False,
    **kwargs
    ):
    evaluation_params = evaluation_params or {}
    evaluation_params.update(kwargs)
    log_mlflow = evaluation_params.get("log_mlflow", log_mlflow)
    log_shap = evaluation_params.get("log_shap", log_shap)
    if len(X_eval)==0 and len(X_train)==0:
        raise ValueError("No data to evaluate on.")
    if evaluation_params.get("eval_locations"):
        eval_locations = evaluation_params["eval_locations"]
    else:
        if len(X_eval)>0:
            eval_locations = X_eval.index.get_level_values("location_id").unique().tolist()
        else:
            eval_locations = []
    logging.info(f"Computing test and eval metrics...")
    logging.info(f"X_eval shape: {X_eval.shape}")
    Xs, ys, set_names = (X_train, X_eval), (y_train, y_eval), ("train", "eval")
    metrics_dict = {}
    preds_dfs = []
    for X, y, set_name in zip(Xs, ys, set_names):
        if len(X)==0:
            logging.info(f"No data to evaluate on for {set_name} set.")
            continue
        y_pred = model.predict(X)
        metrics_dict[set_name] = compute_metrics(y, y_pred, *evaluation_params.get("metrics", []))
        logging.info(f"{set_name} set metrics: {metrics_dict[set_name]}")
        preds_df = pd.DataFrame({"y_pred":y_pred,"y_true":y.values},index=X.index)
        preds_df["set"] = set_name
        preds_dfs.append(preds_df)
        if log_mlflow:
            mlflow.log_metrics({f"{set_name}_{k}":v for k,v in metrics_dict[set_name].to_dict().items()})
            log_locations_by_time(X,set_name)
            if log_shap:
                logging.info(f"Logging SHAP values of {set_name} set predictions...")
                log_shap_values(model, X, set_name, **evaluation_params.get("shap_params", {}))
    # combine predictions
    preds_df = pd.concat(preds_dfs)
    logging.info("Across sets")
    logging.info("\n"+str(pd.DataFrame(metrics_dict)))
    logging.info("Eval locations and non-eval locations")
    metrics_by_set = make_metrics_by_group(preds_df, "set")
    logging.info("\n"+str(metrics_by_set))
    logging.info("By location id")
    metrics_by_station = make_metrics_by_group(preds_df, "location_id")\
        .assign(is_eval=lambda df: df.index.isin(eval_locations))
    metrics_by_station_train = make_metrics_by_group(preds_df[preds_df.set=="train"], "location_id")
    metrics_by_station_eval = make_metrics_by_group(preds_df[preds_df.set=="eval"], "location_id")
    logging.info("\n"+str(metrics_by_station))
    log_metric_by_time_plot(preds_df, set_name="eval", metric="r2")
    log_metric_by_time_plot(preds_df, set_name="eval", metric="rmse")

    if log_shap:
        logging.info("Computing SHAP values...")
        if evaluation_params.get("shap_by_station"):
            shap_by_station = evaluation_params.get("shap_by_station")
            if shap_by_station=="eval":
                X = X_eval
            elif shap_by_station=="train":
                X = X_train
            else:
                X = pd.concat([X_train,X_eval], axis=0)
            for location_id in X.index.get_level_values("location_id").unique():
                log_shap_values(model,X.loc[location_id],set_name=f"{location_id}",log_mlflow=log_mlflow)

    if log_mlflow:
        mlflow.log_param("eval_locations", eval_locations)
        mlflow.log_param("eval_data_shape", X_eval.shape)
        mlflow.log_param("metrics_target_var",y_eval.name)
        metrics_by_set.to_html("data/07_model_output/set_metrics.html",index=True)
        metrics_by_station.to_html("data/07_model_output/location_metrics.html",index=True)
        metrics_by_station_train.to_html("data/07_model_output/location_metrics_train.html",index=True)
        metrics_by_station_eval.to_html("data/07_model_output/location_metrics_eval.html",index=True)
        mlflow.log_artifact("data/07_model_output/set_metrics.html")
        mlflow.log_artifact("data/07_model_output/location_metrics.html")
        mlflow.log_artifact("data/07_model_output/location_metrics_train.html")
        mlflow.log_artifact("data/07_model_output/location_metrics_eval.html")

    return preds_df

def make_metrics_by_group(preds_df,group_by):
    metrics_by_group = (
        preds_df
        .groupby(group_by)
        .apply(lambda g: compute_metrics(g["y_true"], g["y_pred"]))
        .assign(prop=preds_df.groupby(group_by).y_pred.count()/len(preds_df))
        .assign(size=preds_df.groupby(group_by).y_pred.count())
        .round(4)
    )
    return metrics_by_group

def log_trained_model(
    model, 
    model_path:str,
    X_train=None, 
    y_train=None,
    model_name:dict=None,
    model_params:dict=None,
    fit_params:dict=None,
    fit_time:float=None,
    log_feature_importances:bool=False,
    log_sklearn_model:bool=True,
    **extra_params
    ):
    import mlflow
    model_name = model_name or model.__class__.__name__
    if model_name.lower().startswith("gridsearch"):
        mlflow.log_param("best_params", model.best_params_)
        model = model.best_estimator_
        model_name = model_name+f"[{model.__class__.__name__}]"
    
    # mlflow.log_param("model", model_name)
    if model_params:
        mlflow.log_param("model_params", model_params)
    if fit_params:
        mlflow.log_param("fit_params", fit_params)
    if fit_time:
        mlflow.log_param("fit_time", fit_time)
    if X_train is not None:
        mlflow.log_param("model_input_features", X_train.columns.tolist())
        mlflow.log_param("train_data_shape", X_train.shape)
    if X_train is not None and y_train is not None:
        mlflow.log_metrics({f"train_{k}":v for k,v in compute_metrics(y_train, model.predict(X_train)).to_dict().items()})
    if extra_params:
        mlflow.log_params(extra_params)
    if log_feature_importances:
        import matplotlib.pyplot as plt
        # plot feature importances
        fig, ax = plt.subplots(figsize=(10,10))
        pd.Series(
            model.feature_importances_, 
            index=X_train.columns,
        ).sort_values(
            ascending=True
        ).plot(
            kind='barh',
            figsize=(10,10),
            ax=ax,
            title="Feature Importances of {} model".format(model_name),
        )
        fig.tight_layout(); fig.set_figwidth(10)
        mlflow.log_figure(fig, "feature_importances.png")
    if log_sklearn_model:
        # log model artifact
        mlflow.sklearn.log_model(model, model_path)


def log_shap_values(
    model,
    X,
    set_name,
    explainer="shap.TreeExplainer",
    log_mlflow=True):
    if explainer!="pred_contrib":
        if isinstance(explainer,str):
            Explainer = utils.get_object_from_str(explainer)
        else:
            Explainer = explainer
        explainer = Explainer(model)
        shap_values = explainer.shap_values(X)
    else:
        shap_values = model.predict(X,pred_contrib=True)[:,:-1]
    if log_mlflow:
        import mlflow, shap
        fig, ax = plt.subplots(figsize=(10,10),facecolor="white")
        shap.summary_plot(shap_values, X)
        ax.set_title(f"SHAP feature importances of the {set_name} set predictions")
        with tempfile.TemporaryDirectory() as tmpdir:
            fig.savefig(os.path.join(tmpdir,  f"{set_name}_shaps.png"), dpi=300, bbox_inches="tight")
            mlflow.log_artifact(os.path.join(tmpdir,  f"{set_name}_shaps.png"), "shap")


def log_locations_by_time(X,set_name, resample="1D", **kwargs):
    x_resampled = utils.resample_multiindex(X,interval=resample, agg="count", **kwargs)
    locations_count_by_time = x_resampled.groupby("time").count().iloc[:,0]
    fig, ax = plt.subplots(figsize=(10,10),facecolor="white")
    locations_count_by_time.plot(ax=ax,title=f"Count of locations by time for the {set_name} set")
    fig.tight_layout(); fig.set_figwidth(10)
    mlflow.log_figure(fig, f"locations_count_by_time_{set_name}.png")

def get_metric_by_time_plot(preds_df, metric, set_name=None, resample="1D", ax=None, **kwargs):
    if set_name:
        preds_df = preds_df[preds_df.set==set_name]
    title = kwargs.pop("title", None)
    preds_by_time = (
        preds_df
        .pipe(
            utils.resample_multiindex, interval=resample,
            agg=lambda g: compute_metrics(g["y_true"], g["y_pred"]),
            with_apply=True,
            **kwargs
        )
        .reset_index()
    )
    if not ax:
        fig, ax = plt.subplots(figsize=(15,10),facecolor="white")
    else:
        fig = ax.get_figure()
    sns.lineplot(x="time", y=metric, data=preds_by_time.astype({"location_id":int}), hue="location_id", ax=ax)
    if not title:
        title = f"{metric} by time for the {set_name} set" if set_name else f"{metric} by time"
    ax.set(title=title, xlabel="")
    return fig, ax

def log_metric_by_time_plot(preds_df, metric, set_name, *args,**kwargs):
    fig, ax = get_metric_by_time_plot(preds_df=preds_df, metric=metric, set_name=set_name, *args,**kwargs)
    with tempfile.TemporaryDirectory() as tmpdir:
        fig.savefig(os.path.join(tmpdir,  f"{metric}_by_time_{set_name}.png"), dpi=300, bbox_inches="tight")
        mlflow.log_artifact(os.path.join(tmpdir,  f"{metric}_by_time_{set_name}.png"))

def compute_metrics(true, pred,*funcs):
    metrics_dict = dict(
        rmse=np.sqrt(mean_squared_error(true, pred)),
        mae=mean_absolute_error(true, pred),
        r2=r2_score(true, pred),
    )
    for func in funcs:
        metrics_dict[func.__name__] = func(true, pred)
    return pd.Series(metrics_dict)

def log_params(*params:List[dict]):
    import mlflow
    for param_dict in params:
        mlflow.log_param(param_dict)

def tweak_features(functions:Dict,*dfs:List[pd.DataFrame]) -> List[pd.DataFrame]:
    tweaked_dfs = []
    for df in dfs:
        if len(df) > 0:
            df_tweaked = features.apply_functions_to_df(df,functions)
        else:
            df_tweaked = df            
        tweaked_dfs.append(df_tweaked)
    return tweaked_dfs

def search_best_estimator(
    model,
    X_train,
    y_train,
    X_eval,
    y_eval,
    search_strategy:str="grid",
    fit_params:dict=None,
    parameters_to_search:dict=None,
    functions_to_search:dict=None,
    scoring:str=r2_score,
    n_jobs:int=1,
    log_mlflow:bool=True,
) -> Tuple[Any, dict]:
    """
    Search for the best estimator using the specified search strategy.
    """
    logging.info(f"Attempting to find best {model} estimator with {search_strategy} search...")
    parameters_to_search = parameters_to_search or {}
    fit_params = fit_params or {}
    functions_to_search = functions_to_search or {}

    if search_strategy!="grid":
        raise NotImplementedError(f"Search strategy {search_strategy} not implemented. Use \"grid\"")
    if isinstance(scoring,str):
        scoring = {
            "r2":r2_score,
            "mae":mean_absolute_error,
            "rmse":partial(mean_squared_error, squared=False),
            "mse":mean_squared_error,
        }.get(scoring,utils.get_object_from_str(scoring))
    if isinstance(model,str):
        model = utils.get_object_from_str(model)
    
    def fit_estimator_with_params(estimator,X,y, params,**kwargs):
        estimator.set_params(**params)
        estimator.fit(X, y, **fit_params)
        results={
            "estimator":estimator,
            "params":params,
            "score":scoring(y_eval,estimator.predict(X_eval)),
        }
        results.update(**kwargs)
        return results

    logging.info("number of parameters to search: {}".format(len(parameters_to_search)))
    logging.info("number of functions to search: {}".format(len(functions_to_search)))
    if search_strategy=="grid":
        params_grid = list(ParameterGrid(parameters_to_search))
        funcs_grid = list(ParameterGrid(functions_to_search))
        logging.info(f"parameter grid is of size {len(params_grid)}")
        logging.info(f"function grid is of size {len(funcs_grid)}")
        results_list = []
        for i,funcs in enumerate(funcs_grid):
            X_train, X_eval = tweak_features(funcs, X_train, X_eval)
            with utils.tqdm_joblib(tqdm(desc="training grid...",total=len(params_grid))) as pbar:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(fit_estimator_with_params)(
                        model,
                        X_train,
                        y_train,
                        params,
                        functions=funcs,
                        **fit_params,
                    ) for params in params_grid
                )
            results_list.extend(results)
        
        results_df = pd.DataFrame(results_list).sort_values("score",ascending=False)
        best_estimator = results_df.iloc[0]["estimator"]
        best_params = results_df.iloc[0]["params"]
        best_score = results_df.iloc[0]["score"]
        best_funcs = results_df.iloc[0]["functions"]
        logging.info(f"Best estimator: {best_estimator}")
        logging.info(f"Best params: {best_params}")
        logging.info(f"Best score: {best_score}")
        logging.info(f"Best functions: {best_funcs}")
        logging.info(f'results head:\n{results_df[["estimator","score","functions"]].head()}')
        logging.info(f'results tail:\n{results_df[["estimator","score","functions"]].tail()}')
        if log_mlflow:
            mlflow.log_param("best_estimator",best_estimator)
            mlflow.log_param("best_params",best_params)
            mlflow.log_param("best_score",best_score)
            mlflow.log_param("best_funcs",best_funcs)
            results_df.to_html("data/07_model_output/best_results.html",index=True)
            mlflow.log_artifact("data/07_model_output/best_results.html")
    else:
        raise NotImplementedError(f"Search strategy {search_strategy} not implemented. Use \"grid\"")
    
    return best_estimator, best_params
        
def get_map_of_locations(
    X_train:pd.DataFrame,
    X_eval: pd.DataFrame,
    locations_gdf,
    location_cols = None,
    point_locations:List[str]=None,
    map_params: dict = None,
    location_level: str = "location_id",
    mp = None,
    **kwargs
) -> None:
    from spatial_interpolation.visualization import map_viz
    from folium import LayerControl
    map_params = map_params or kwargs.pop("folium_map",{})
    if location_cols is None:
        location_cols = X_train.columns[X_train.columns.str.match("location_\\d$")]
    if point_locations is None:
        test_locations = np.concatenate([X_eval[col].values for col in location_cols])
        train_locations = np.concatenate([X_train[col].values for col in location_cols])
        pcs = np.concatenate([train_locations, test_locations])
    else:
        pcs = point_locations

    eval_locations = X_eval.index.get_level_values(location_level).unique()
    eval_gdf = locations_gdf[locations_gdf.index.isin(eval_locations)]

    # make a map with all relevant locations
    map_ = map_viz.add_geodf_to_map(
        locations_gdf.loc[locations_gdf.index.isin(set(pcs))],
        map=mp,
        map_args = map_params,
        color="blue", radius=1, 
        weight=5, 
        layer_name="Available Locations", 
        **kwargs
    )
    # show the test locations in the map
    map_viz.add_geodf_to_map(
        eval_gdf, map_, 
        radius=1, 
        weight=5, 
        color="red", 
        layer_name="Eval Locations",
        **kwargs
    )
    LayerControl().add_to(map_)
    return map_

def log_map_of_locations(*args,**kwargs):
    """"
    Logs a map of buoy locations to the mlflow experiment.
    Receives the same arguments as `get_map_of_locations`.
    """
    map_ = get_map_of_locations(*args,**kwargs)
    log_folium_map(map_, "map_of_locations.html")

def log_folium_map(mp, fname):
    if not fname.endswith(".html"):
        fname += ".html"
    with tempfile.TemporaryDirectory() as tmpdir:
        mp.save(os.path.join(tmpdir,fname))
        mlflow.log_artifact(os.path.join(tmpdir,fname))

def log_pandas_table(df, fname):
    """
    Log a pandas dataframe as an html table to mlflow.
    """
    if not fname.endswith(".html"):
        fname += ".html"
    with tempfile.TemporaryDirectory() as tmpdir:
        df.to_html(os.path.join(tmpdir,fname))
        mlflow.log_artifact(os.path.join(tmpdir,fname))

def log_fig(fig, fname, **kwargs):
    """
    Log a matplotlib figure as an image to mlflow.
    """
    if not fname.endswith(".png"):
        fname += ".png"
    with tempfile.TemporaryDirectory() as tmpdir:
        fig.savefig(os.path.join(tmpdir,fname), dpi=300, bbox_inches="tight", **kwargs)
        mlflow.log_artifact(os.path.join(tmpdir,fname))
