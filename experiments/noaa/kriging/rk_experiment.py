"""
An experiment to evaluate kriging statistical models to make interpolation on the NDBC buoy data.
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
from pydantic import validate_arguments
from sympy import evaluate
from tqdm import tqdm

from pyhandy import subplotted
from spatial_interpolation import utils, data
from spatial_interpolation.visualization import map_viz
from spatial_interpolation.utils.experiments import MLFlowExperiment
from spatial_interpolation.utils.modeling import compute_metrics
from spatial_interpolation.pipelines.feature_interpolation import validation
from experiments.configs import rk_experiments_conf
from experiments.configs.evaluation import eval_sets as eval_conf
from spatial_interpolation.utils.modeling import tweak_features

class NOAARegressionKrigingExperiment(MLFlowExperiment):
    """
    An experiment to evaluate regression kriging to make interpolation on the NDBC buoy data.
    """
    config = rk_experiments_conf
    experiment_name = "NOAA-Regression-Kriging"
    params_to_log = ["target", "eval_set", "n_jobs"]

    @property
    def description(self):
        config = self.get_config()
        desc = f"""
        Interpolation experiment for the NOAA dataset using kriging interpolation
        with {config.krige} in the {config.eval_set} evaluation dataset.
        """.strip()
        return desc

    @property
    def tags(self):
        tags = [
            ("noaa",), ("kriging",),
            ("config", str(self._cfg)),
        ]
        return tags
    
    @property
    def run_name(self):
        return str(self._cfg)

    """
    An experiment to evaluate kriging statistical models to make interpolation on the NDBC buoy data.
    """
    config = rk_experiments_conf
    experiment_name = "NOAA-Kriging-Interpolation"
    params_to_log = ["target", "eval_set", "n_jobs"]

    @property
    def description(self):
        config = self.get_config()
        desc = f"""
        Interpolation experiment for the NOAA dataset using regression kriging interpolation
        with {config.krige} and lightgbm in the {config.eval_set} evaluation dataset.
        """.strip()
        return desc

    @property
    def tags(self):
        tags = [
            ("noaa",), ("kriging",),
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

        if config.get("interpolator_params"):
            params_dict = json.loads(json.dumps(dict(config.get("interpolator_params")), default=str))
            mlflow.log_params(params_dict)

         # log evaluation set as yaml
        eval_set_config = eval_conf.ndbc[config.eval_set].copy_and_resolve_references()
        del eval_set_config["eval"]
        self._log_config_as_yaml(eval_set_config, "eval_set.yaml")

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
        if config.get("pretrain_funcs"):
            X_train, X_eval = tweak_features(
                config.pretrain_funcs,
                X_train, X_eval
            )
        y_train = y_train.loc[X_train.index]
        y_eval = y_eval.loc[X_eval.index]
        train_df = train_df.loc[X_train.index,:]
        test_df = test_df.loc[X_eval.index,:]
  
        # get rid of times that have less than 4 data points
        train_times = X_train.index.get_level_values("time").unique()
        test_times = X_eval.index.get_level_values("time").unique()
        
        if config.get("eval_frac",1) < 1:
            self.logger.info(f"A random subset of {config.eval_frac*100}% of the test data will be used for evaluation")
            test_times = test_times.to_series().sample(frac=config.eval_frac)
        elif config.get("eval_frac",1) > 1:
            self.logger.info(f"A random subset of {config.eval_frac} observations of the test data will be used for evaluation")
            test_times = test_times.to_series().sample(n=config.eval_frac)
        
        mlflow.log_param("eval_frac", config.get("eval_frac",1))

        mod = mlflow.pyfunc.load_model(config.regressor_uri)
        kriger = config.interpolator
        dims = config.dimensions
        krige = kriger(
            dimensions=dims,
            **config.interpolator_params
        )
        target = config.target
        def fit_evaluate_krige_on_time(time, **kwargs):
            """
            Fit a model on all the points at the time given
            and return the result of evaluating the model on the test
            set.
            """
            if time not in train_times or len(X_train.loc[idx[:,time],:])<=2:
                return
            try:
                tr_pred = mod.predict(X_train.loc[idx[:,time],:])
                train_true = y_train.loc[idx[:,time]]
                krige.fit(train_df.loc[idx[:,time],:], train_true-tr_pred)
                tst_pred = mod.predict(X_eval.loc[idx[:,time],:])
                rk_pred = krige.predict(test_df.loc[idx[:,time],:], with_error=False)
                pred = tst_pred + rk_pred
            except Exception as e:
                self.logger.warning(f"Failed to fit krige {krige} on time {time}: {e}")
                return
            return pd.DataFrame({target:pred},index=test_df.loc[idx[:,time],:].index)

        eval_start = time.time()
        if config.n_jobs > 1:
            self.logger.info(f"Using {config.n_jobs} parallel jobs to evaluate the model")
            # divide the test times into chunks
            test_times_chunks = np.array_split(test_times, config.n_jobs)
            # run the evaluation in parallel
            with utils.tqdm_joblib(tqdm(desc="computing kriging interpolations...",total=len(test_times_chunks))):
                pred_lsts = Parallel(n_jobs=config.n_jobs)(
                    delayed(lambda chunk: [fit_evaluate_krige_on_time(time) for time in tqdm(chunk)])(c) for c in test_times_chunks
                )
            preds = [p for l in pred_lsts for p in l]
        else:
            preds = [fit_evaluate_krige_on_time(time) for time in  tqdm(test_times, desc="fitting krigging at eval times...")]
        mlflow.log_metric("time_to_eval", time.time() - eval_start)

        # contatenate predictions
        preds_df = (
            pd.concat(preds)
            .join(test_df[[target]], rsuffix="_true")
            .rename(columns={target:"y_pred", f"{target}_true":"y_true"})
            .dropna()
            .sort_index()
        )
        preds_df["set"] = "eval"
        mlflow.log_param("num_evaluated_points", preds_df.shape[0])
        # evaluate and log insights about the predictions
        eval_metrics = validation.compute_metrics(true=preds_df.y_true, pred=preds_df.y_pred)
        mlflow.log_metrics(eval_metrics)
        eval_locations = test_df.index.get_level_values("location_id").unique().tolist()
        metrics_by_loc = validation.make_metrics_by_group(preds_df, "location_id")\
            .assign(is_eval=lambda df: df.index.isin(eval_locations))
        metrics_by_loc.to_html("data/07_model_output/location_metrics.html",index=True)
        mlflow.log_artifact("data/07_model_output/location_metrics.html")
        validation.log_locations_by_time(test_df, time_level="time", set_name="eval")
        validation.log_locations_by_time(train_df, time_level="time", set_name="train")

        preds_df[preds_df["set"]=="eval"].to_csv(f"data/07_model_output/{self.run_name}_eval_preds.csv")
        
        # log partial evaluation sets
        if "partial" in eval_set_config:
            self.log_partial_sets(preds_df, eval_set_config)
    

    def log_partial_sets(self, preds_df, eval_set_config):
        """
        Log insights about the predictions on the partial sets of the evaluation set.
        """
        preds_df = preds_df.copy()
        # get metrics for each partial evaluation set
        partial_evals = []
        partial_sets = {d:eval_set_config["partial"][d] for d in eval_set_config["partial"] if eval_set_config["partial"][d]}
        for fig, ax, name in subplotted(partial_sets, ncols=1, figsize=(15,len(partial_sets)*7)):
            partial_set = partial_sets[name]
            eval_locs = partial_set.get("locations")
            eval_time = partial_set.get("time")
            preds_on_set = preds_df.loc[idx[eval_locs,eval_time.start:eval_time.end],:]
            if len(preds_on_set)<2:
                continue
            set_eval_metrics = dict(
                **validation.compute_metrics(true=preds_on_set.y_true, pred=preds_on_set.y_pred),
                set=name,
                locations=eval_locs,
                time_start=eval_time.start,
                time_end=eval_time.end,
            )
            validation.get_metric_by_time_plot(
                preds_on_set, 
                metric="rmse", 
                ax=ax, 
                time_level="time",
                title=f"Timeseries of RMSE on partial set {name}",
            )
            partial_evals.append(set_eval_metrics)
        partial_evals_df = pd.DataFrame(partial_evals).set_index("set")
        validation.log_pandas_table(partial_evals_df, "partial_eval_metrics")
        validation.log_fig(fig, "timeseries_of_partial_eval_metrics")
        

                
    

