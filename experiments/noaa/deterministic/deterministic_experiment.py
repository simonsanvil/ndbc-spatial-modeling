"""
An experiment to evaluate deterministic interpolation methods on the NDBC buoys dataset.
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
from spatial_interpolation.interpolators import ScipyInterpolator
from spatial_interpolation.utils.experiments import MLFlowExperiment
from spatial_interpolation.utils.modeling import compute_metrics
from spatial_interpolation.pipelines.feature_interpolation import validation
from experiments.configs import det_experiments_conf
from experiments.configs.evaluation import eval_sets as eval_conf

class NOAADeterministicExperiment(MLFlowExperiment):
    """
    An experiment to evaluate deterministic interpolation methods on the NDBC buoys dataset.
    """
    config = det_experiments_conf
    experiment_name = "NOAA-Deterministic-Interpolation"
    params_to_log = ["interpolator", "target", "eval_set", "n_jobs"]

    @property
    def description(self):
        config = self.get_config()
        desc = f"""
        Interpolation experiment for the NOAA dataset using determinstic interpolation
        with {config.interpolator.__name__} in the {config.eval_set} evaluation dataset.
        """.strip()
        return desc

    @property
    def tags(self):
        tags = [
            ("noaa",), ("deterministic",),
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

        params_dict = json.loads(json.dumps(dict(config.get("interpolator_params")), default=str))
        mlflow.log_params(params_dict)

         # log evaluation set as yaml
        eval_set_config = eval_conf.ndbc[config.eval_set].copy_and_resolve_references()
        del eval_set_config["eval"]
        self._log_config_as_yaml(eval_set_config, "eval_set.yaml")

        dataset = data.NDBCDataLoader(**config.data_loading).load()
        if eval_set_config.get("area"):
            self.logger.info(f"Filtering data on area {eval_set_config.area}")
            df, gdf = dataset.buoys_data, dataset.buoys_geo
            locations_within_area = gdf.loc[gdf.within(eval_set_config.area)].index.get_level_values("buoy_id").unique()
            df = df.loc[df.index.get_level_values("buoy_id").isin(locations_within_area)]
            gdf = gdf.loc[idx[:, locations_within_area],:]
            self.logger.info(f"Filtered data has shapes {df.shape} and {gdf.shape}")
            dataset = data.NDBCData(df, gdf)

        available_time = dataset.buoys_data.index.get_level_values("time").unique()
        time_range = pd.date_range(available_time.min(), available_time.max(), freq="H")
        train, test = dataset.split_slice(test=eval_conf.ndbc[config.eval_set].eval)
        train_df = train.join()
        test_df = test.join()

        target = config.target
        time_steps = (
            time_range
            .to_series(name="time_step")
            .rename_axis("time")
            .rank(ascending=True)
        )
        train_by_times = (
            train_df
            .reset_index()
            .rename(columns={"buoy_id": "location_id"})
            .set_index(["time", "location_id"])
            .sort_index()
            .dropna(subset=[target])
            .join(time_steps)
        )
        test_by_times = (
            test_df
            .reset_index()
            .rename(columns={"buoy_id": "location_id"})
            .set_index(["time", "location_id"])
            .sort_index()
            .dropna(subset=[target])
            .join(time_steps)
        )
        # get rid of times that have less than 4 data points
        available_times = train_by_times.groupby("time").count()[target]
        train_by_times = train_by_times.loc[available_times.index[available_times >= 4]]

        interpolator = ScipyInterpolator(
            config.interpolator,
            dimensions=config.dimensions,
            **config.interpolator_params
        )
        is_temporal = "time" in interpolator.dim_cols or "time_step" in interpolator.dim_cols
        self.logger.info(f"Interpolator is temporal?: {is_temporal}")
        delta = pd.Timedelta(config.temp_interpolation.delta)
        train_times = train_by_times.index.get_level_values("time").unique()
        test_times = test_by_times.index.get_level_values("time").unique()
        
        if config.get("eval_frac",1) < 1:
            self.logger.info(f"A random subset of {config.eval_frac*100}% of the test data will be used for evaluation")
            test_times = test_times.to_series().sample(frac=config.eval_frac)
        elif config.get("eval_frac",1) > 1:
            self.logger.info(f"A random subset of {config.eval_frac} observations of the test data will be used for evaluation")
            test_times = test_times.to_series().sample(n=config.eval_frac)
        
        mlflow.log_param("eval_frac", config.get("eval_frac",1))
        mlflow.log_param("num_test_times", len(test_times))
        mlflow.log_param("num_train_times", len(train_times))
        mlflow.log_param("prop_test_obs_by_time", len(test_times)/test_by_times.shape[0])
        mlflow.log_param("prop_train_obs_by_time", len(train_times)/train_by_times.shape[0])
    
        def fit_evaluate_interpolator_on_time(time, **kwargs):
            """
            Fit a model on all the points at the time given
            and return the result of evaluating the model on the test
            set.
            """
            if time not in train_times or len(train_by_times.loc[time])<2:
                return
            if is_temporal:
                try:
                    interpolator.fit(train_by_times.loc[(time-delta):time],y=target)
                    pred = interpolator.predict(test_by_times.loc[time])
                except Exception as e:
                    self.logger.warning(f"Failed to fit {interpolator} on time {time} with data {train_by_times.loc[(time-delta):time]}")
                    self.logger.warning(f"Exception: {e}")
                    raise e
            else:
                interpolator.fit(train_by_times.loc[time],y=target)
                pred = interpolator.predict(test_by_times.loc[time])
                # return
            return pd.DataFrame({target:pred},index=test_by_times.loc[[time]].index)

        eval_start = time.time()
        if config.n_jobs > 1:
            self.logger.info(f"Using {config.n_jobs} parallel jobs to evaluate the model")
            # divide the test times into chunks
            test_times_chunks = np.array_split(test_times, config.n_jobs)
            # run the evaluation in parallel
            with utils.tqdm_joblib(tqdm(desc="computing deterministic interpolations...",total=len(test_times_chunks))):
                pred_lsts = Parallel(n_jobs=config.n_jobs)(
                    delayed(lambda chunk: [fit_evaluate_interpolator_on_time(time) for time in tqdm(chunk)])(c) for c in test_times_chunks
                )
            preds = [p for l in pred_lsts for p in l]
        else:
            preds = [
                fit_evaluate_interpolator_on_time(time) 
                for time in  
                tqdm(test_times, desc="fitting interpolator at eval times...")
            ]
        mlflow.log_metric("time_to_eval", time.time() - eval_start)

        # contatenate predictions
        preds_df = (
            pd.concat(preds)
            .join(test_by_times[[target]], rsuffix="_true")
            .rename(columns={target:"y_pred", f"{target}_true":"y_true"})
            .dropna()
            .sort_index()
        )
        preds_df["set"] = "eval"
        mlflow.log_param("num_evaluated_points", preds_df.shape[0])
        # evaluate and log insights about the predictions
        eval_metrics = validation.compute_metrics(true=preds_df.y_true, pred=preds_df.y_pred)
        mlflow.log_metrics(eval_metrics)
        eval_locations = test_by_times.index.get_level_values("location_id").unique().tolist()
        metrics_by_loc = validation.make_metrics_by_group(preds_df, "location_id")\
            .assign(is_eval=lambda df: df.index.isin(eval_locations))
        metrics_by_loc.to_html("data/07_model_output/location_metrics.html",index=True)
        mlflow.log_artifact("data/07_model_output/location_metrics.html")
        validation.log_metric_by_time_plot(
            preds_df.sort_index(), 
            set_name="eval", 
            metric="rmse",
            time_level="time",
        )
        validation.log_locations_by_time(test_by_times, time_level="time", set_name="eval")
        validation.log_locations_by_time(train_by_times, time_level="time", set_name="train")

        preds_df[preds_df["set"]=="eval"].to_csv(f"data/07_model_output/{self.run_name}_eval_preds.csv")
        
        # log partial evaluation sets
        if "partial" in eval_set_config:
            self.log_partial_sets(preds_df, eval_set_config)

        # log map of buoys with heatmap along time
        buoy_gdf = data.load_buoys_geo()
        locations_gdf = buoy_gdf[
            ~buoy_gdf.index.get_level_values("buoy_id").duplicated(keep='last')
        ].reset_index("year",drop=True)
        tr_locations = train_by_times.index.get_level_values("location_id").unique()
        te_locations = test_by_times.index.get_level_values("location_id").unique()
        df_times = pd.concat([train_by_times, test_by_times], axis=0)\
            .pipe(utils.resample_multiindex, interval="D", time_level="time")
        map_ = map_viz.heatmap_with_time(
            df_times.sort_index(level="time"), 
            time_col="time", 
            name="heatmap in time",
            format="%b %d, %Y",
            **config.get("map_params",{})
        )
        validation.log_folium_map(map_, "map_of_locations_by_time")
        validation.log_map_of_locations(
            train_df, test_df, locations_gdf,
            location_level="buoy_id",
            point_locations=np.concatenate([tr_locations, te_locations]).tolist(),
            # mp=map_,
            **config.get("map_params",{})
        )
    
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
            preds_on_set = preds_df.loc[idx[eval_time.start:eval_time.end,eval_locs],:]
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
        

                
    

