"""
Functions to train and evaluate spatial interpolation model for the interluft project.
"""

from functools import partial
import logging
import time
import warnings
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import numpy as np
import geopandas as gpd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

from spatial_interpolation import utils
from spatial_interpolation.pipelines.feature_interpolation import feature_extraction
from spatial_interpolation.features import *
from spatial_interpolation.features import apply_functions_to_df
from spatial_interpolation import utils
from spatial_interpolation.utils import get_object_from_str

from . import validation
from .validation.splits import ValidationSplit

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_model(
    model:Union[str,object],
    X_train:pd.DataFrame,
    y_train:pd.Series,
    X_eval=None,
    y_eval=None,
    model_params:dict=None,
    fit_params:dict=None,
    log_mlflow:bool=False,
    gridsearch_params:dict=None,
    log_model_params:dict=None,
    **kwargs
    ) -> Tuple[object,pd.Series,pd.Series]:
    """
    Train a spatial interpolation model for the interluft project.
    """
    model_params = model_params or {}
    fit_params = fit_params or {}
    logger.info(f"Attempting to train model {model}...")
    logger.info("X_train shape: {}".format(X_train.shape))
    logger.info("model_params: {}".format(model_params))
    logger.info("train columns: {}".format(X_train.columns.tolist()))

    if isinstance(model,str):
        model_cls = get_object_from_str(model)
        model = model_cls(**model_params)
    if gridsearch_params:
        logger.info("Fitting to a gridsearch to find best hyperparameters...")
        logger.info("gridsearch_params: {}".format(gridsearch_params))
        if X_eval is not None:
            logger.info("using evaluation data for gridsearch")
            X_train, y_train = pd.concat([X_train,X_eval]), pd.concat([y_train,y_eval])
            gridsearch_params["cv"] = [(np.zeros(len(X_train)),np.ones(len(X_eval)))]   
        model = GridSearchCV(model,**gridsearch_params)
    elif X_eval is not None and len(X_eval)>0:
        fit_params["eval_set"] = (X_eval,y_eval)
    
    logger.info(f"Training {model.__class__.__name__} model...")
    fit_start = time.time()
    model.fit(X_train, y_train.iloc[:,0].values,**fit_params)
    fit_end = time.time()
    logger.info(f"Fitting of {model.__class__.__name__} model took {fit_end - fit_start} seconds")

    if log_mlflow:
        validation.log_trained_model(
            model if not gridsearch_params else model.best_estimator_,
            model_path=kwargs.pop("model_path",f"{y_train.iloc[:,0].name}_{model.__class__.__name__}.mod"),
            X_train=X_train,
            y_train=y_train,
            fit_time=fit_end - fit_start,
            **(log_model_params or {})
        )
    if gridsearch_params:
        logger.info(f"Best parameters: {model.best_params_}")
        logger.info(f"Best score: {model.best_score_}")
        model = model.best_estimator_
    return model


def preprocess_training_data(
    buoys_df:pd.DataFrame,
    buoys_gdf:gpd.GeoDataFrame,
    preprocess_funcs:Dict[str,Any],
    feature_params:dict,
    **func_args,
):
    # get params of feature extraction
    train_vars:List[str] = feature_params.get("train_vars")
    # project the geographic data to epsg 4326
    buoys_gdf.geometry.crs = 'epsg:4326'
    
    if train_vars:
        buoys_df = buoys_df.dropna(how="any",subset=train_vars)

    buoys_df = apply_functions_to_df(buoys_df,preprocess_funcs,**func_args)
    
    return buoys_df, buoys_gdf


def make_training_features(
    buoys_df:pd.DataFrame,
    buoys_gdf:gpd.GeoDataFrame,
    target_var: str,
    validation_splits:ValidationSplit,
    make_features_args:dict,
    num_jobs:int=1,
) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    #compute the features
    make_features_with_split = partial(
        compute_features_of_split,
        (buoys_df, buoys_gdf),
        validation_splits=validation_splits,
        n_jobs=num_jobs,
        **make_features_args)
    # compute the training features
    train_features_df = make_features_with_split(split="train")
    # compute the test features
    test_features_df = make_features_with_split(split="test")
    # make X and y
    X_train,y_train = train_features_df.drop(columns=[target_var]), train_features_df[target_var]
    X_test, y_test = test_features_df.drop(columns=[target_var]), test_features_df[target_var]
    y_train = pd.DataFrame(y_train, columns=[target_var])
    y_test = pd.DataFrame(y_test, columns=[target_var])

    X = pd.concat([X_train,X_test],axis=0).assign(is_test = [False]*X_train.shape[0] + [True]*X_test.shape[0])
    y = pd.concat([y_train,y_test],axis=0).assign(is_test = X.is_test)

    return X, y
    

def make_features_of_split(
    validation_splits:List[ValidationSplit],
    buoys_df:pd.DataFrame,
    buoys_gdf:gpd.GeoDataFrame,
    target_var:str,
    make_features_args:dict,
    is_location_split:bool=False,
    num_jobs:int=1,
    apply_functions:dict=None,
    **kwargs,
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    # variables to use for training
    k_nearest = make_features_args.pop("k_nearest")
    feature_vars = make_features_args.pop("traffic_vars",None) or buoys_df.columns.tolist()

    if all(len(s.values)==0 for s in validation_splits):
        logging.info("No data in this validation split. Returning empty X,y dataframes")
        return pd.DataFrame(), pd.DataFrame()

    # define function to compute features for a single split
    def extract_single_split(buoys_df, split_name:str, **kwargs) -> pd.DataFrame:
        logger.info(f"Extracting features for split \"{split_name}\"...")
        features_df = feature_extraction.make_features(
            (buoys_df, buoys_gdf),
            k_nearest=k_nearest,
            feature_vars=feature_vars,
            **kwargs
        )
        logger.info(f"The computed features for this split has shape {features_df.shape}")
        return features_df
    
    # compute the features
    if num_jobs>1:
        # If parallel the function generator must be delayed
        extract_single_split = delayed(extract_single_split)
    # if the number of jobs is greater than the number of splits, we need to 
    # further divide the validation splits into chunks so that all jobs run at the same time
    if num_jobs>len(validation_splits):    
        logging.info(f"Splitting the {len(validation_splits)} splits into {num_jobs} chunks for parallel processing...")
        jobs_per_split = utils.int_partition(len(validation_splits), num_jobs)
        valsplit_chunks = []
        for split,n_splits in zip(validation_splits, jobs_per_split):
            new_time_splits = utils.make_time_splits(split.start, split.end, num_splits=n_splits, group_by_pairs=True)
            valsplit_chunks += [
                ValidationSplit(
                    start=start, end=end,
                    values=split.values.copy(),
                    name = split.name,
                    validation_strategy=split.validation_strategy.copy(),
                )
                for start,end in new_time_splits
            ]
        validation_splits = valsplit_chunks
        logging.info(f"The splits have been divided into {len(validation_splits)} chunks.")
    # make features for each (train or test) split
    logger.info(f"Attempting to make features for each of the {len(validation_splits)} splits...")
    if not is_location_split:
        features_dfs = (
            extract_single_split(
                buoys_df.loc[split.values],
                split_name=split.name,
                **kwargs
            )
            for split in validation_splits
            if len(split.values)>0
        )
    else:
        features_dfs = (
            extract_single_split(
                buoys_df.loc[split.values],
                split_name=split.name,
                **kwargs
            )
            for split in validation_splits
            if len(split.values)>0
        )
    
    if num_jobs>1:
        # Make the features in parallel
        logging.info(f"Computing features in parallel w/ {num_jobs} jobs...")
        with utils.tqdm_joblib(tqdm(desc="Computing features...",total=len(validation_splits))) as pbar:
            features_dfs = Parallel(n_jobs=num_jobs)(features_dfs)
    else:
        # Make the features in serial
        logging.info(f"Computing features in serial...")
        features_dfs = []
        for feature_df in tqdm(features_dfs, desc="Computing features..."):
            features_dfs.append(feature_df)
    
    logger.info(f"Concatenating features ...")
    # concatenate the features and remove missing values
    features_df = pd.concat(features_dfs, axis=0)\
        .dropna(how="any",subset=feature_vars)\
        .sort_index()
    logger.info(f"Final shape {features_df.shape}")
    # get rid of duplicates
    features_df = features_df.loc[~features_df.index.duplicated(keep="first")]
    # make X and y
    X,y = features_df.drop(columns=[target_var]), features_df[[target_var]]
    if apply_functions is not None:
        X = features.apply_functions_to_df(X,functions=apply_functions)

    return X, y
    

def split_train_from_test(X,y)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    X_train, X_test = X[X.is_test==False], X[X.is_test==True]
    y_train, y_test = y[y.is_test==False], y[y.is_test==True]
    return X_train, X_test, y_train, y_test

def compute_features_of_split(
    buoys_df:pd.DataFrame,
    buoys_gdf:gpd.GeoDataFrame,
    validation_splits:List[ValidationSplit],
    split:str,
    k_nearest:int=4,
    feature_vars:List[str]=None,
    n_jobs:int=1,
    **kwargs
    ) -> None:
    """
    Make features for the training splits of a spatial interpolation model for the interluft project.
    """
    # define function to compute features for a single split
    def make_features_of_split(split_buoys_df,split_name:str, **kwargs) -> pd.DataFrame:
        logger.info(f"Making {split} features for split \"{split_name}\"...")
        features_df = feature_extraction.make_features(
            split_buoys_df,
            buoys_gdf,
            k_nearest=k_nearest,
            feature_vars=feature_vars,
            **kwargs
        )
        logger.info(f"The computed features for this split has shape {features_df.shape}")
        return features_df
    
    if n_jobs>1:
        # If parallel the function generator must be delayed
        make_features_of_split = delayed(make_features_of_split)
    # if the number of jobs is greater than the number of splits, we need to 
    # further divide the validation splits into chunks so that all jobs run at the same time
    if n_jobs>len(validation_splits):    
        logging.info(f"Splitting the {len(validation_splits)} splits into {n_jobs} chunks for parallel processing...")
        jobs_per_split = utils.int_partition(len(validation_splits), n_jobs)
        valsplit_chunks = []
        for split,n_splits in zip(validation_splits, jobs_per_split):
            new_time_splits = utils.make_time_splits(split.start, split.end, num_splits=n_splits, group_by_pairs=True)
            valsplit_chunks += [
                ValidationSplit(
                    start=start, end=end,
                    air_quality=split.air_quality.copy(),
                    validation_strategy=split.validation_strategy.copy(),
                ) 
                for start,end in new_time_splits
            ]
        validation_splits = valsplit_chunks
        logging.info(f"The splits have been divided into {len(validation_splits)} chunks.")
        
    # make features for each (train or test) split
    logger.info(f"Attempting to make {split} features for each of the {len(validation_splits)} splits...")
    if split=="train":
        features_dfs = (
            make_features_of_split(
                split_buoys_df=buoys_df.loc[split.train],
                split_name=f"Split #{i+1} {split.start}-{split.end} "\
                f"train ids: {split.train.tolist()}, test ids {split.test.tolist()} ",
                **kwargs
            )
            for i,split in enumerate(validation_splits)
            if len(split.train)>0
        )
    elif split=="test":
        features_dfs = (
            make_features_of_split(
                split_buoys_df=buoys_df.loc[split.test],
                split_name=f"Split #{i+1} {split.start}-{split.end} "\
                f"train ids: {split.train.tolist()}, test ids {split.test.tolist()} ",
                **kwargs
            )
            for i,split in enumerate(validation_splits)
            if len(split.air_quality.test)>0
        )
        
    if n_jobs>1:
        # Make the features in parallel
        with utils.tqdm_joblib(tqdm(desc="Computing features...",total=len(validation_splits))) as pbar:
            features_dfs = Parallel(n_jobs=n_jobs)(features_dfs)
    else:
        # Make the features in serial
        features_dfs = list(features_dfs)
    
    logger.info(f"Concatenating features ...")
    # concatenate the features and remove missing values
    features_df = pd.concat(features_dfs, axis=0)\
        .dropna(how="any",subset=feature_vars)\
        .sort_index()
    logger.info(f"Final shape {features_df.shape}")
    
    return features_df


def make_evaluation_features(
    aq_eval_stations,
    traffic_data,
    buoys_gdf,
    aq_df,
    aq_gdf,
    weather_df,
    target:str,
    make_features_args:dict,
    n_jobs=1,
) ->  None: 
    """
    Evaluate the model on the validation splits.
    """

    if len(aq_eval_stations)==0:
        logger.info("No stations to evaluate on.")
        return pd.DataFrame(), pd.DataFrame()
    
    k_nearest_traffic = make_features_args.pop("k_nearest_traffic")
    k_nearest_air_quality = make_features_args.pop("k_nearest_air_quality")
    traffic_vars = make_features_args.pop("traffic_vars")
    air_quality_vars = make_features_args.pop("air_quality_vars")
    weather_vars = make_features_args.pop("weather_vars")

    # project the geographic data to epsg 4326
    aq_gdf.geometry.crs = {'init':'epsg:4326'}
    buoys_gdf.geometry.crs = {'init':'epsg:4326'}
    
    # variables to use for training
    traffic_vars = traffic_vars or traffic_data.columns.tolist()
    air_quality_vars = air_quality_vars or aq_df.columns.tolist()
    weather_vars = weather_vars or weather_df.columns.tolist()

    # only keep the timestamps that match the weather data
    weather_df = weather_df.loc[weather_df.index.isin(aq_df.index.get_level_values(-1))].sort_index()
    weather_df = weather_df.loc[weather_df.index.isin(traffic_data.index.get_level_values(-1))]

    make_features_from_weather = partial(
        feature_extraction.make_features,
        (traffic_data, buoys_gdf),
        (aq_df, aq_gdf),
        k_nearest_traffic=k_nearest_traffic,
        k_nearest_aq=k_nearest_air_quality,
        traffic_vars=traffic_vars,
        aq_vars=air_quality_vars,
        weather_vars=weather_vars,
        station_ids=aq_eval_stations,
        **make_features_args,
    )

    if n_jobs>1:
        # make time splits of weather data index for parallel processing
        weather_start = weather_df.index.get_level_values(-1).min()
        weather_end = weather_df.index.get_level_values(-1).max()
        start, end = utils.parse_start_end_times(weather_start, weather_end)
        weather_splits = utils.make_time_splits(start, end, num_splits=n_jobs, group_by_pairs=True)
        # make a weather dataframe for each split
        weather_splits = [
            weather_df.loc[split_start:split_end] 
            for split_start,split_end in weather_splits
        ]
        make_features_from_weather = delayed(make_features_from_weather)
    else:
        # make features for each job
        weather_splits = [weather_df]
    
    eval_features = (
        make_features_from_weather(split)
        for split in weather_splits
    )
    if n_jobs>1:
        # Make the features in parallel
        with utils.tqdm_joblib(tqdm(desc="Computing eval features...",total=len(weather_splits))) as pbar:
            eval_features = Parallel(n_jobs=n_jobs)(eval_features)
    else:
        # Make the features in serial
        eval_features = list(eval_features)
    
    # concatenate the features and remove missing values
    eval_features = pd.concat(eval_features, axis=0)
    eval_features = eval_features.filter(
        regex="^((?!traffic_\d$|aq_\d$).)*$",
    ).dropna(
        how="any",
        subset=weather_vars+air_quality_vars
    ).sort_index()
    logger.info(f"Evaluation features shape: {eval_features.shape}")
    logger.info(f"Evaluation features columns: {eval_features.columns}")

    X_eval, y_eval = eval_features.drop(target, axis=1), eval_features[target]
    
    return X_eval, pd.DataFrame(y_eval,columns=[target])





       