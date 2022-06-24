"""
An experiment to train and evaluate machine learning models on geospatial and time
features extracted from the NDBC buoy data.
"""
import json
import os
import time
import warnings, logging
from functools import lru_cache

import numpy as np
import pandas as pd
import mlflow
from pandas import IndexSlice as idx
from joblib import Parallel, delayed

from spatial_interpolation import utils, data
from spatial_interpolation.pipelines.feature_interpolation import validation
from spatial_interpolation.utils.experiments import MLFlowExperiment
from spatial_interpolation.utils.modeling import (
    tweak_features,
    compute_metrics,
)
from pyhandy import subplotted
from experiments.configs import ml_experiments_conf
from experiments.configs.evaluation import eval_sets as eval_conf

class NOAAMLTraining(MLFlowExperiment):
    """
    An experiment to train and evaluate machine learning models on geospatial and time
    features extracted from the NDBC buoy data.
    """
    config = ml_experiments_conf
    experiment_name = "NOAA-ML-Interpolation"
    params_to_log = ["model", "target", "eval_set"]

    @property
    def description(self):
        config = self.get_config()
        desc = f"""
        Interpolation experiment for the NOAA dataset using interpolation from machine-learning models
        with {config.model.__name__} in the {config.eval_set} evaluation dataset.
        """.strip()
        return desc

    @property
    def tags(self):
        tags = [
            ("noaa",), ("machine_learning",),
            ("config", str(self._cfg)),
        ]
        return tags
    
    @property
    def run_name(self):
        return str(self._cfg)
    
    def run(self, **kwargs):
        """
        Run the experiment.
        """
        config = self.get_config()
        if config.get("seed"):
            np.random.seed(config.seed)

        params_dict = json.loads(json.dumps(dict(config.get("model_params")), default=str))
        mlflow.log_params(params_dict)

        train_df = pd.concat(
            [pd.read_parquet(f"{config.input.train_dir}/{year}.parquet") for year in range(2011,2022)],
            axis=0).sort_index()
        test_df = pd.concat(
            [pd.read_parquet(f"{config.input.eval_dir}/{year}.parquet") for year in range(2011,2022)],
            axis=0).sort_index()
        
        if config.get("time"):
            self.logger.info(f"Filtering data on time {config.time.to_dict()}")
            train_df = train_df.loc[idx[:,config.time.start:config.time.end],:]
            test_df = test_df.loc[idx[:,config.time.start:config.time.end],:]
        
        # Make the X and Y train and test sets that will be passed to the model
        X_train = train_df.drop(columns=[config.target]).copy()
        y_train = train_df[config.target]
        X_eval = test_df.drop(columns=[config.target]).copy()
        y_eval = test_df[config.target]

        # log the map of buoys 
        buoy_gdf = data.load_buoys_geo()
        locations_gdf = buoy_gdf[
            ~buoy_gdf.index.get_level_values("buoy_id").duplicated(keep='last')
        ].reset_index("year",drop=True)
        validation.log_map_of_locations(
            train_df, test_df, locations_gdf,
            location_cols = train_df.columns[train_df.columns.str.match("location_\\d$")],
            map_params = config.get("map_params",{}),
        )

        if config.get("pretrain_funcs"):
            X_train, X_eval = tweak_features(
                config.pretrain_funcs,
                X_train, X_eval
            )
        y_train = y_train.loc[X_train.index]
        y_eval = y_eval.loc[X_eval.index]
        
        # Fit the model
        fit_start = time.time()
        if config.get("make_grid_search"):
            self.logger.info(f"Gridsearching on {config.gridsearch_params}...")
            mod, best_params = validation.search_best_estimator(
                config.model,
                X_train, y_train,
                X_eval, y_eval,
                search_strategy="grid",
                **config.gridsearch_params
            )
            self.logger.info(f"Best params: {best_params}")
        else:
            mod_cls = config.model if not isinstance(config.model,str) else utils.get_object_from_str(config.model)
            mod = mod_cls(**config.model_params)
            fit_params = config.get("fit_params",{}).to_dict()
            if fit_params.get("eval_set"):
                fit_params["eval_set"] = [(X_eval, y_eval)]
            mod.fit(X_train, y_train, **fit_params)
        fit_time = time.time() - fit_start
        mlflow.log_metric("fit_time", fit_time)
        self.logger.info(f"Model was fitted in: {fit_time:.2f}s")
        self.logger.info(f"Model fit with score {mod.score(X_train, y_train):.4f}")
        validation.log_trained_model(
            mod,
            model_path=f"{mod_cls.__name__.lower()}",
            X_train=X_train,
            y_train=y_train,
            model_name=mod_cls.__name__,
            **config.to_dict().get("mlflow_logging",{}).get("log_model",{}),
        )

        # Evaluate the model
        inference_start = time.time()
        y_pred = mod.predict(X_eval)
        mlflow.log_metric("inference_time", time.time() - inference_start)
        metrics = compute_metrics(y_eval, y_pred, **config.get("metrics",{}))
        self.logger.info(f"Evaluation metrics: {metrics.to_dict()}")
        mlflow.log_metrics(metrics.to_dict())
        preds_df = validation.evaluate_model(mod, X_eval, y_eval, X_train, y_train, **config.get("model_validation",{}))
        # log partial evaluation sets
        eval_set_config = eval_conf.ndbc[config.eval_set].copy_and_resolve_references()
        del eval_set_config["eval"]
        if "partial" in eval_set_config:
            self.log_partial_sets(preds_df, eval_set_config)
        # log eval predictions
        model_name = mod_cls.__name__.lower()
        preds_df[preds_df["set"]=="eval"].to_csv(f"data/07_model_output/{model_name}_predictions.csv")
        # mlflow.log_artifact(f"data/07_model_output/{model_name}_predictions.csv")
    
    def log_partial_sets(self, preds_df, eval_set_config):
        """
        Log insights about the predictions on the partial sets of the evaluation set.
        """
        preds_df = preds_df.copy().sort_index()
        # get metrics for each partial evaluation set
        partial_evals = []
        partial_sets = {d:eval_set_config["partial"][d] for d in eval_set_config["partial"] if eval_set_config["partial"][d]}
        for fig, ax, name in subplotted(partial_sets, ncols=1, figsize=(15,len(partial_sets)*7)):
            partial_set = partial_sets[name]
            eval_locs = partial_set.get("locations")
            eval_time = partial_set.get("time")
            preds_on_set = preds_df.loc[idx[eval_locs, eval_time.start:eval_time.end],:]
            if len(preds_on_set)<2:
                continue
            set_eval_metrics = dict(
                **validation.compute_metrics(true=preds_on_set.y_true, pred=preds_on_set.y_pred),
                set=name,
                locations=eval_locs,
                time_start=eval_time.start,
                time_end=eval_time.end,
            )
            # validation.get_metric_by_time_plot(
            #     preds_on_set, 
            #     metric="rmse", 
            #     ax=ax, 
            #     time_level="time",
            #     title=f"Timeseries of RMSE on partial set {name}",
            # )
            partial_evals.append(set_eval_metrics)
        partial_evals_df = pd.DataFrame(partial_evals).set_index("set")
        validation.log_pandas_table(partial_evals_df, "partial_eval_metrics")
        validation.log_fig(fig, "timeseries_of_partial_eval_metrics")


        



        