"""
An spatial interpolator that uses a regression model and a 
krigging algorithm to interpolate the data using regression-kriging.
https://en.wikipedia.org/wiki/Regression-kriging
"""

import logging
from typing import Any, Callable, Dict, List, Tuple
from collections import OrderedDict
import numpy as np
import pandas as pd
import geopandas as gpd

class RegressionKriging:
    """
    An spatial interpolator that uses a regression model and a 
    kriging algorithm to interpolate the data using regression-kriging.
    https://en.wikipedia.org/wiki/Regression-kriging

    Regression kriging is a method for interpolating irregularly spaced data
    using a regression model. The method consists of the following steps:
    1. The regression model is used to predict the value of known points.
    2. After the prediction, the kriging algorithm is used to interpolate the
       residuals between the known and predicted values.


    Parameters
    ----------
    regressor: Any
        A fitted regression model to use for interpolation.
        Must have a predict method that takes a numpy array of shape (n,d)
        where n is the number of points to interpolate and d the number of dimensions.
    krige: Any
        The kriging algorithm to use for interpolation.
        Must implement `fit` and `predict` methods.
    min_points: int
        The minimum number of points to use for interpolation.
        If there are less than this number of points, the interpolation is not performed.
        Default is 0.
    """

    def __init__(
        self, 
        regressor, 
        krige, 
        min_points=0,
        coord_names=None,
    ):
        if not hasattr(regressor, "predict"):
            raise TypeError("regressor must have a predict method")
        if not hasattr(krige, "predict"):
            raise TypeError("krige must have a predict method")

        self.regressor = regressor
        self.krige = krige
        self.min_points = min_points
        self.coord_names = coord_names
        self.is_fitted = False
    
    def __repr__(self) -> str:
        return f"RegressionKriging(regressor={self.regressor}, krige={self.krige}, min_points={self.min_points})"
    
    def predict(self, *args, **kwargs) -> np.ndarray:
        return self.interpolate(*args, **kwargs)
        
    
    def fit(self, coords:gpd.GeoDataFrame, y:pd.DataFrame) -> None:
        """
        Fit the krigging model to interpolate the residuals between the known and predicted values.

        Parameters
        ----------
        coords: gpd.GeoDataFrame, pd.DataFrame
            The coordinates of the observed points to interpolate.
            - If a GeoDataFrame is passed, the coordinates are extracted from the geometry column.
            - If a DataFrame is passed, the coordinates are extracted from the columns specified in the `coord_names` parameter.
            - If an ndarray it must be a numpy array of shape (n,2) where n is the number of points.
        y: pd.Series or np.array
            The target observed values of shape (n,) to fit the kriging model to.
        """
        if isinstance(y,(pd.Series)):
            y = y.values
        coords_array = self.coords_to_array(coords)
        if y.shape[0] != coords_array.shape[0]:
            raise ValueError("The number of coordinates in X must match the number of observed values in y")
        if coords_array.shape[0] < self.min_points:
            logging.error(f"Not enough points to interpolate. {coords_array.shape[0]} < {self.min_points}")
            return None
        
        y_reg = self.regressor.predict(coords)
        self.krige.fit(coords_array,y-y_reg)
        self.is_fitted = True
    
    def interpolate(self, coords: Any) -> np.ndarray:
        """
        Interpolate the points using the features extracted from the nearest available covariates

        Parameters
        ----------
        X: pd.DataFrame
            Covariates/Features to estimate each point with the regression model.
        coords: gpd.GeoDataFrame, pd.DataFrame, ndarray
            The coordinates of the points to interpolate which will be passed to the regression model
            and the kriging algorithm to interpolate the values and residuals.
            - If a GeoDataFrame is passed, the coordinates are extracted from the geometry column.
            - If a DataFrame is passed, the coordinates are extracted from the columns specified in the `coord_names` parameter.
            - If an ndarray it must be a numpy array of shape (n,2) where n is the number of points.
        
        Returns
        -------
        y: np.ndarray
            The interpolated values with shape (n,)
            The values are the result of the prediction of the regression model
            plus the residuals interpolated by the krige.
        """
        if not self.is_fitted:
            raise ValueError("RegressionKriging has not been fitted yet. Call fit() first.")
        reg_pred = self.regressor.predict(coords)
        coords_array = self.coords_to_array(coords)
        krig_pred = self.krige.predict(coords_array)
        
        return reg_pred + krig_pred
    
    def coords_to_array(self, coords:Any):
        """
        Convert the coordinates to a numpy array of shape (n,2) where n is the number of points and d the number of dimension of the coordinates.

        Parameters
        ----------
        coords: gpd.GeoDataFrame, pd.DataFrame, ndarray
            The coordinates of the points to interpolate.
            - If a GeoDataFrame is passed, the coordinates are extracted from the geometry column.
            - If a DataFrame is passed, the coordinates are extracted from the columns specified in the `coord_names` parameter.
            - If an ndarray it must be a numpy array of shape (n,d) where n is the number of points 
        
        Returns
        -------
        coords_array: np.ndarray
            The coordinates as a numpy array of shape (n,d) where n is the number of points and d is the dimension of the coordinates.
        """
        if isinstance(coords,np.ndarray):
            if self.coord_names is not None:
                assert coords.shape[1] == len(self.coord_names), "The number of coordinates in coords must match the number of coordinates in coord_names"
            pass
        elif isinstance(coords, gpd.GeoDataFrame):
            coords = coords.geometry.apply(lambda x: x.coords[0]).apply(pd.Series).values
        elif isinstance(coords,pd.DataFrame):
            if self.coord_names is None:
                raise ValueError("Must specify the names of the coordinates")
            coords = coords[self.coord_names].values
        elif isinstance(coords,zip) or isinstance(coords,list):
            grid = np.array(list(coords))
            coords = grid[:,[0,1]], 
        elif isinstance(coords,tuple):
            coords = np.array(coords)
        else:
            raise ValueError("coords must be a GeoDataFrame, a DataFrame, a list of tuples, a list of lists, or a numpy array of coordinates")
        
        return coords

    

