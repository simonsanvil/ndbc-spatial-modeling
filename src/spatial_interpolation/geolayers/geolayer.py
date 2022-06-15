"""
A GeoStack is an abstraction of a collection of geopandas.GeoDataFrames or pandas.DataFrames
that represent multiple spatial variables that share dimensions with each other.
"""

import geopandas as gpd
import pandas as pd
import numpy as np

from typing import List, Tuple, Dict, Any, Union, Optional

class GeoStack:
    """
    A GeoStack is an abstraction of a collection of geopandas.GeoDataFrames or pandas.DataFrames with spatial data
    that is used to represent multiple variables or data sources that share a spatial dimensions with each other.

    You can perform operations and transformations on a GeoLayer as if it were a single GeoDataFrame.

    When they share the same dimensions, multiple GeoLayers can be combined together into one GeoStack.

    Attributes
    ----------
    layer_data :  geopandas.GeoDataFrame, List[Union[geopandas.GeoDataFrame, pd.DataFrame]]
        A GeoDataFrames or list of GeoDataFrames that represent the data in this GeoLayer.
    dimensions : List[str]
        A list of strings with the name of the columns on `layer_data` that represent the dimensions of the layer.
        Every dataframe in this GeoLayer must contain at least one of these dimensions in their columns.
        An example of a dimension is "geometry" for a GeoDataFrame, or "time", "lat", "lon" for a pandas.DataFrame.
    index : str = None
        The name of the column(s) on `layer_data` that represents the index of the layer.
        If there are multiple dataframes in this GeoLayer, they must all share the same index.
    """
    def __init__(
        self, 
        layer_data: Union[gpd.GeoDataFrame, List[Union[gpd.GeoDataFrame, pd.DataFrame]]],
        dimensions: List[str],
        index: Optional[str] = None
    ):
        """
        Initialize a new GeoLayer.

        Parameters
        ----------
        layer_data : Union[geopandas.GeoDataFrame, List[Union[geopandas.GeoDataFrame, pd.DataFrame]]]
            A GeoDataFrames or list of GeoDataFrames that represent the data in this GeoLayer.
        dimensions : List[str]
            A list of strings with the name of the columns on `layer_data` that represent the dimensions of the layer.
            Every dataframe in this GeoLayer must contain at least one of these dimensions in their columns.
            An example of a dimension is "geometry" for a GeoDataFrame, or "time", "lat", "lon" for a pandas.DataFrame.
        index : str = None
            The name of the column(s) on `layer_data` that represents the index of the layer.
            If there are multiple dataframes in this GeoLayer, they must all share the same index.
        """
        self.layer_data = layer_data
        self.dimensions = dimensions
        self.index = index

        if isinstance(layer_data, gpd.GeoDataFrame):
            self.geodataframes = [layer_data]
        else:
            self.geodataframes = layer_data

        self._check_geodataframes()
    
    def _check_geodataframes(self):
        """
        Check that all GeoDataFrames in this GeoLayer have the same dimensions and index.
        """
        if len(self.geodataframes) == 0:
            raise ValueError("GeoLayer must contain at least one GeoDataFrame.")

        if self.index is not None:
            if not all(df.index.name == self.index for df in self.geodataframes):
                raise ValueError("All GeoDataFrames in this GeoLayer must share the same index.")

        if not all(self.dimensions.issubset(df.columns) for df in self.geodataframes):
            raise ValueError("All GeoDataFrames in this GeoLayer must share the same dimensions.")

    def __repr__(self):
        """
        Return a string representation of this GeoLayer.
        """
        return "GeoLayer(geodataframes={})".format(self.geodataframes)

    def __str__(self):
        """
        Return a string representation of this GeoLayer.
        """
        return "GeoLayer(geodataframes={})".format(self.geodataframes)

    def __len__(self):
        """
        Return the number of GeoDataFrames in this GeoLayer.
        """
        return len(self.geodataframes)

    def __getitem__(self, index: int) -> gpd.GeoDataFrame:
        """
        Return the GeoDataFrame at the given index.

        Parameters
        ----------
        index : int
            The index of the GeoDataFrame to return.

        Returns
        -------
        geopandas.GeoDataFrame
            The GeoDataFrame at the given index.
        """
        return self.geodataframes[index]

    def __iter__(self):
        """
        Return an iterator over the GeoDataFrames in this GeoLayer.
        """
        return iter(self.geodataframes)

    def __eq__(self, other):
        """
        Return whether this GeoLayer is equal to another GeoLayer.
        """
        return all(df in other.geodataframes for df in self.geodataframes)

    def __ne__(self, other):
        """
        Return whether this GeoLayer is not equal to another GeoLayer.
        """
        return not self == other

    # def __init__(self,
    #              layer_name: str,
    #              layer_data: Union[gpd.GeoDataFrame,List[gpd.GeoDataFrame,pd.DataFrame]],
    #              layer_metadata: Dict[str, Any] = None,


class GeoMask:
    """
    A GeoMask is an abstraction of a collection of geopandas.GeoDataFrames
    that are used to represent a single layer of spatial data (for one source of spatial data).
    """

    pass