"""
A transformer mixin that extracts features from the  dataset.
"""

from typing import Dict, Any, List, Tuple
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from pandas import IndexSlice as idx
from sklearn.base import BaseEstimator, TransformerMixin

from spatial_interpolation import features
from spatial_interpolation.data import NDBCDataLoader
from spatial_interpolation.pipelines.feature_interpolation import feature_extraction

from .base_extractor import BaseFeatureExtractor

class NDBCFeatureExtractor(BaseFeatureExtractor):

    coord_names = ["x","y"]
    Dataloader = NDBCDataLoader

    def __init__(self,
        inference:bool=True,
        preprocess_params:Dict[str,Any]=None,
        make_features_params:Dict[str,Any]=None,
        postprocess_params:Dict[str,Any]=None,
        dataloader:NDBCDataLoader=None,
        return_coords:bool=False,
        from_azure:bool=False,
        verbose:bool=True,
    ):
        if dataloader is not None:
            if not isinstance(dataloader,NDBCDataLoader):
                raise TypeError(f"dataloader must be an instance of {self.DataLoader}")

        super().__init__(
            dataloader=dataloader,
            preprocess_params=preprocess_params,
            make_features_params=make_features_params,
            postprocess_params=postprocess_params,
            verbose=verbose
        )
        self.inference = inference
        self.return_coords = return_coords
        self.from_azure = from_azure
        self._preprocessed_data = None
        if from_azure:
            self.dataloader.aq_from_azure = True
    
    @property
    def preprocessed(self):
        if self._preprocessed_data is None:
            return self.preprocess()
        return self._preprocessed_data

    @property
    def is_fitted(self):
        return self.dataloader.is_fitted
    
    @property
    def params(self):
        return {
            "preprocess_params":self.preprocess_params,
            "make_features_params":self.make_features_params,
            "post_features_params":self.post_features_params,
        }

    def fit(self,*args,**kwargs):
        if len(args) == 1 and isinstance(args[0],self.Dataloader.Data):
            self.dataloader.data = args[0]
        elif isinstance(args[0],pd.DataFrame) and isinstance(args[1],gpd.GeoDataFrame):
            print("Fitting with dataframes")
            self.dataloader = self.Dataloader.from_datasets(args[0],args[1])
        else:
            self.dataloader.fit(*args,**kwargs)
        return self
    
    def update_params(self,**kwargs):
        self.preprocess_params = self.preprocess_params.update(kwargs.get("preprocess_params",{}))
        self.make_features_params = self.make_features_params.update(kwargs.get("make_features_params",{}))
        self.post_features_params = self.post_features_params.update(kwargs.get("post_features_params",{}))
        return self
    
    def make_features(self, points_gdf, data=None, preprocess=True):
        if data is None:
            data = self.dataloader.load()
            if preprocess:
                data = self.preprocess()
        return feature_extraction.make_features(
            buoys_df=data.buoys_data,
            buoys_gdf=data.buoys_geo,
            points_gdf=points_gdf,
            **self.make_features_params
        )
        
    def transform(self,X:gpd.GeoDataFrame,y=None):
        """
        Extract features from the  dataset based on the parameters passed to the constructor.

        Parameters
        ----------
        X : geopandas.GeoDataFrame
            The input geopandas dataframe of geographical point locations to 
            extract nearest-based features from.
        """
        if isinstance(X,zip) or isinstance(X,tuple):
            X = self.make_points_from_grid(X)
        if not isinstance(X,gpd.GeoDataFrame) and "geometry" in X.columns:
            X = gpd.GeoDataFrame(X)
            X.crs = "epsg:4326"

        self.logger.info("Attempting to load data from  dataloader...")
        _ = self.dataloader.load()
        self.logger.info("Data loaded. Preprocessing data...")
        data_frames = self.preprocess()
        self.logger.info("Attempting to make features of the data...")
        features_df = feature_extraction.make_features(
            *data_frames,
            points_gdf=X,
            inference=self.inference,
            **self.make_features_params
        )
        self.logger.info("Features made. Applying post-processing functions...")
        self.features_coordinates = features_df[self.coord_names]
        features_df = self.postprocess(features_df)

        if self.return_coords:
            return features_df,self.features_coordinates

        return features_df
    
    def fit_transform(self,points_gdf=None, *args,**kwargs):
        self.dataloader.fit(*args,**kwargs)
        return self.transform(points_gdf)
    
    def transform_grid(self, X:Any, Y=None):
        """
        Transform a grid of points into features.

        Parameters
        ----------
        X,Y : np.ndarray
            The input meshgrid of points to extract nearest-based features from.
            If Y is None, X is assumed to be a list (x,y) tuples of points.
        """
        X = self.make_points_from_grid(X,Y)
        return self.transform(X)

    @staticmethod
    def make_points_from_grid(X, Y=None):
        """
        Build a grid of points from the given X,Y meshgrid of coordinates.
        """
        if Y is None:
            if isinstance(X,zip) or isinstance(X,list):
                grid = np.array(list(X))
                X, Y = grid[:,0], grid[:,1]
            elif isinstance(X,tuple):
                X,Y = X                
        points_gdf = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_xy(
                X.ravel(),Y.ravel(),
                crs="epsg:4326",
            ))
        points_gdf.index.name = "station_id"
        points_gdf.shape
        return points_gdf
    
    def postprocess(self,features_df:pd.DataFrame):
        features_df = features.apply_functions_to_df(features_df,functions=self.postprocess_params)
        return features_df
    
    @classmethod
    def preprocess_data(cls, data, buoy_data_funcs, buoy_geo_funcs):
        buoy_df = data.buoys_data
        buoy_gdf = data.buoys_geo

        if buoy_data_funcs:
            buoy_df = features.apply_functions_to_df(buoy_df, buoy_data_funcs)
        if buoy_geo_funcs:
            buoy_gdf = features.apply_functions_to_df(buoy_gdf, buoy_geo_funcs)

        buoy_gdf.crs = "epsg:4326"
        buoy_df = buoy_df.loc[buoy_df.index.get_level_values("time").year.isin(buoy_gdf.index.get_level_values("year"))]
        buoy_gdf = buoy_gdf.loc[buoy_gdf.index.get_level_values("year").max()]
        preprocessed_data = cls.Dataloader.Data(buoy_df, buoy_gdf)
        return preprocessed_data
    
    # Data processing
    def preprocess(
        self,
        buoy_data_funcs:Dict[str,Any]=None,
        buoy_geo_funcs:Dict[str,Any]=None,
        data=None,
        **func_args,
    ):
        data = self.dataloader.load()
        data_functions = self.preprocess_params.get("buoy_data_funcs",buoy_data_funcs)
        geo_functions = self.preprocess_params.get("buoy_geo_funcs",buoy_geo_funcs)

        self._preprocessed_data = self.preprocess_data(data, data_functions, geo_functions)
        return self._preprocessed_data