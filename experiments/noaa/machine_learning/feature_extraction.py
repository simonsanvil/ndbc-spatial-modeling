"""
An experiment to extract features from the data to use for machine learning models.
"""
import os
import warnings, logging
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
from joblib import Parallel, delayed

from spatial_interpolation.data.dataloaders import NDBCData
from spatial_interpolation.utils.experiments import Experiment
from spatial_interpolation.pipelines.feature_interpolation import feature_extraction
from spatial_interpolation import features, data, utils

from experiments.configs import feature_extraction_conf

class FeatureExtractionExperiment(Experiment):
    """
    An experiment to extract features from the training and test sets of the data 
    to use to train and evaluate machine learning models on the NDBC buoy data.
    """
    config = feature_extraction_conf
    experiment_name = "feature extraction"
    
    def run(self, save_features=True, **kwargs):
        """
        Run the experiment.
        """
        config = self.get_config()
        if config.get("seed"):
            np.random.seed(config.seed)
            
        self.logger.info("Loading data...")
        dataset = self.get_data()

        train, test = dataset.split_eval(**config.split_strategy)
        train_pipe = feature_extraction.NDBCFeatureExtractor(**config.feature_extraction)
        eval_pipe = feature_extraction.NDBCFeatureExtractor(**config.feature_extraction)
        eval_pipe.fit(dataset); train_pipe.fit(train)
        
        num_jobs = config.get("num_jobs") or config.get("n_jobs") or config.get("n_cores") or 1
        self.logger.info(f"Extracting features with {num_jobs=}...")
        compute_features_params = dict(target=config.get("target","wave_height"), n_jobs=num_jobs)
        train_features = self.compute_features(train_pipe, train, **compute_features_params)
        eval_features = self.compute_features(eval_pipe, test, **compute_features_params)

        if len(train_features)==0:
            self.logger.warning("No features were extracted from training set at the specified time period and area")
        if len(eval_features)==0:
            self.logger.warning("No features were extracted from test set at the specified time period and area")
        
        if not save_features:
            return train_features, eval_features
        
        if config.get("year"):
            train_fpath = os.path.join(config.output.train_dir, f"{config.year}.parquet")
            test_fpath = os.path.join(config.output.eval_dir, f"{config.year}.parquet")
        else:
            train_fpath = os.path.join(config.output.train_dir, "train.parquet")
            test_fpath = os.path.join(config.output.eval_dir, "eval.parquet")

        # Save the features
        self.logger.info("Saving features...")
        self.save_features(train_features, train_fpath)
        self.save_features(eval_features, test_fpath)

        return train_features, eval_features
    
    @classmethod
    def compute_features(
        clss,
        extractor:feature_extraction.NDBCFeatureExtractor,
        dataset:NDBCData, 
        target:str="wave_height", 
        n_jobs=1):
        """
        Get the features for the data in parallel.
        """
        prep = extractor.preprocess_data(dataset, **extractor.preprocess_params)
        df = prep.buoys_data.copy()
        gdf = prep.buoys_geo.copy()

        if n_jobs == 1:
            points_gdf = gdf.join(df)[[target,"geometry"]].dropna()
            return extractor.transform(points_gdf)

        index_chunks = np.array_split(df.index, n_jobs)
        
        def get_features_of_chunk(ind_chunk):
            points_gdf_chunk = gdf.join(df.loc[ind_chunk])[[target,"geometry"]].dropna()
            return extractor.transform(points_gdf_chunk)
        
        feature_dfs = Parallel(n_jobs=n_jobs)(delayed(get_features_of_chunk)(ind_chunk) for ind_chunk in index_chunks)
        return pd.concat(feature_dfs, axis=0)
    
    def set_data(self,dataset:data.NDBCData):
        """
        Set the data.
        """
        if isinstance(dataset,tuple):
            dataset = data.NDBCData(*dataset)
        elif isinstance(dataset,data.NDBCDataLoader):
            dataset = data.data
        elif not isinstance(dataset,data.NDBCData):
            raise TypeError("data must be an instance of NDBCData")
        self._loaded_data = dataset

    @lru_cache(maxsize=1)
    def get_data(self):
        """
        Get the data.
        """
        if hasattr(self, "_loaded_data"):
            return self._loaded_data
        config = self.get_config()
        dataloader = data.NDBCDataLoader(**config.data_loading)
        dataset = dataloader.load_data()
        if config.get("area"):
            self.logger.info(f"Filtering data on area {config.area}")
            df, gdf = dataset.buoys_data, dataset.buoys_geo
            locations_within_area = gdf.loc[gdf.within(config.area)].index.get_level_values("buoy_id").unique()
            df = df.loc[df.index.get_level_values("buoy_id").isin(locations_within_area)]
            gdf = gdf.loc[idx[:, locations_within_area],:]
            self.logger.info(f"Filtered data has shapes {df.shape} and {gdf.shape}")
            dataset = NDBCData(df, gdf)
        if config.get("year"):
            self.logger.info(f"Filtering data on year {config.year}")
            df, gdf = dataset.buoys_data, dataset.buoys_geo
            time_index_year = df.index.get_level_values("time").year
            df, gdf = df.loc[time_index_year == config.year], gdf.loc[[config.year]]
            self.logger.info(f"Filtered data has shapes {df.shape} and {gdf.shape}")
            dataset = NDBCData(df, gdf)
        return dataset
    
    def save_features(self, features_df, fpath):
        """
        Save the features to a file.
        """
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        features_df.to_parquet(fpath, index=True)