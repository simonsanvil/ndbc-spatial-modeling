from typing import Dict, List
from functools import partial, update_wrapper, wraps

import pandas as pd
from kedro.pipeline import Pipeline, node

from spatial_interpolation.pipelines.feature_interpolation.validation import splits
from spatial_interpolation.utils import modeling
from spatial_interpolation import features, utils
from spatial_interpolation.pipelines.feature_interpolation import (
    validation,
    train,
    feature_extraction
)

extractor = feature_extraction.NDBCFeatureExtractor()

def preprocess_data(extractor_params:dict=None,dataloader_params:dict=None,*datasets):
    extractor.update_params(**extractor_params)
    dataloader = extractor.Dataloader.from_datasets(*datasets,**(dataloader_params or {}))
    extractor.dataloader = dataloader
    return extractor.preprocessed
    
def update_partial(func, **kwargs):
    """Update a partial function with new keyword arguments"""
    return update_wrapper(partial(func, **kwargs), func)

def make_pipelines(**kwargs):
    """
    Make the feature extraction pipelines.    
    """
    return {
        "feature_extraction": Pipeline(
            [   
                node(
                    func=preprocess_data,
                    inputs=[
                        "params:feature_extraction.params",
                        "params:load_data.params",
                        "buoys_data",
                        "buoys_geo"
                    ],
                    outputs=["buoys_df","buoys_gdf"],
                    name="preprocess_data",
                    tags=["preprocessing","preprocess"],
                ),
                node(
                    func=splits.make_validation_splits,
                    inputs={
                        "buoys_df": "buoys_df",
                        "buoys_gdf": "buoys_gdf",
                        "validation_strategy": "params:validation_strategy",
                    },
                    outputs=["train_splits","eval_splits"],
                    name="make_validation_splits",
                    tags=["make_splits","validation_splits","split"],
                ),
                node(
                    func=update_partial(train.make_features_of_split,is_location_split=False),
                    inputs={
                        "validation_splits": "train_splits",
                        "buoys_df": "buoys_df",
                        "buoys_gdf": "buoys_gdf",
                        "target_var": "params:target",
                        "num_jobs": "params:num_jobs",
                        "make_features_args": "params:make_features",             
                    },
                    outputs=["X_train","y_train"],
                    name="make_train_features",
                    tags=["feature_extraction","extract_features","train_features"],
                ),
                node(
                    func=update_partial(train.make_features_of_split,is_location_split=True),
                    inputs={
                        "validation_splits": "eval_splits",
                        "buoys_df": "buoys_df",
                        "buoys_gdf": "buoys_gdf",
                        "target_var": "params:target",
                        "make_features_args": "params:make_features",
                        "num_jobs": "params:num_jobs",             
                    },
                    outputs=["X_eval","y_eval"],
                    name="make_eval_features",
                    tags=["feature_extraction","extract_features","eval_features"],
                ),
            ]
        ),
        'model_training': Pipeline(
            [
                node(
                    func=modeling.tweak_features,
                    inputs=[
                        "params:pretrain_funcs",
                        "X_train",
                        "X_eval",
                    ],
                    outputs=["X_train_post","X_eval_post"],
                    name="tweak_features_pretrain",    
                    tags=["tweak_features","pretrain","train","training","preeval","evaluate","evaluation","modeling"],
                ),
                node( 
                    func=train.train_model,
                    inputs={
                        "model": "params:model",
                        "X_train": "X_train_post",
                        "y_train": "y_train",
                        "model_params": "params:model_params",
                        'fit_params': 'params:model_fit_params',
                        "log_mlflow": "params:log_mlflow",
                        'gridsearch_params': 'params:gridsearch_params',
                        "log_model_params": "params:mlflow_logging.at_training",
                    },
                    outputs="model",
                    name="train_model",
                    tags=["train","training","modeling"],
                ),
                node(
                    func=validation.evaluate_model,
                    inputs={
                        "model": "model",
                        "X_eval": "X_eval_post",
                        "y_eval": "y_eval",
                        "X_train": "X_train_post",
                        "y_train": "y_train",
                        "evaluation_params":"params:mlflow_logging.at_evaluation",
                    },
                    outputs="eval_results",
                    name="evaluate_model",
                    tags=["evaluate","evaluation","modeling"],
                ),
                # node(
                #     func=validation.log_map_of_locations,
                #     inputs={
                #         "X_train": "X_train",
                #         "X_eval": "X_eval",
                #         "buoys_gdf": "traffic_geo",
                #         "map_params": "params:mlflow_logging.map_of_locations",
                #     },
                #     outputs=None,
                #     name="log_map_of_locations",
                #     tags=["evaluate","evaluation"],
                # )
            ]
        ),
        'move_features': Pipeline(
            [
                node(
                    func=move_features_to_dir,
                    inputs={
                        "features_dir": "params:move_features.features_dir",
                        "target_dir": "params:move_features.target_dir",
                        "verbose": "params:move_features.verbose",
                    },
                    outputs=None,
                    name="move_features_to_dir",
                )
            ]
        ),
    }

def move_features_to_dir(features_dir,target_dir, from_wd=True, verbose=True):
    import shutil, glob, logging, os
    if from_wd:
        wdir = os.getcwd()
        features_dir = os.path.join(wdir,features_dir)
        target_dir = os.path.join(wdir,target_dir)

    feature_files = list(glob.glob(features_dir+"/*.parquet"))
    feature_files = feature_files + list(glob.glob(features_dir+"/*.pkl"))
    for source in feature_files:
        if verbose:
            logging.info(f"Moving {source} to {target_dir}")
        destination = os.path.join(target_dir,os.path.basename(source))
        shutil.move(source,destination)