"""
A data loader for NOAA's buoy data
"""
import logging
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Tuple

import pandas as pd
import geopandas as gpd
from pandas import IndexSlice as idx
from shapely import geometry

from spatial_interpolation import utils, data
from .base_dataset import BaseData

@dataclass
class NDBCData(BaseData):
    buoys_data: pd.DataFrame
    buoys_geo: gpd.GeoDataFrame

    def join(self, resample=None, **kwargs):
        if resample:
            df = self.buoys_data .pipe(utils.resample_multiindex, interval=resample, **kwargs)
        else:
            df = self.buoys_data
        joined_gdf = gpd.GeoDataFrame(
            df
            .assign(year=lambda x: x.index.get_level_values("time").year)
            .set_index("year",append=True)
            .join(self.buoys_geo)
            .reset_index(level="year",drop=True)
            .sort_index()
        )
        return joined_gdf

class NDBCDataLoader:

    buoy_dataset_name = "noaa_data"
    buoy_geo_datase_name = "noaa_data_geo"

    Data = NDBCData

    def __init__(
        self,
        start:str=None,
        end:str="now",
        num_batches:int=None,
        post_loading_func:Callable=None,
        aq_from_azure:bool=False,
        verbose:bool=False,
        **load_kwargs
    ):
        self.start = start
        self.end = end
        self.n_batches = num_batches
        self.post_loading_func = post_loading_func
        self.data = None
        self.aq_from_azure = aq_from_azure
        self.load_kwargs = load_kwargs
        if self.start and self.end:
            self.is_fitted = True
        else:
            self.is_fitted = False
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
    
    def __repr__(self):
        return f"InterluftDataLoader(start={self.start},end={self.end},n_batches={self.n_batches})"
    
    @classmethod
    def from_datasets(cls,*datasets,**kwargs):
        new = cls(**kwargs)
        new.data = new.Data(*datasets)
        return new

    def fit(self,*args,**kwargs):
        """
        Change the start and end dates of the data loader
        or set the Data to a new Data object.
        """
        if len(args) == 1 and isinstance(args[0],self.Data):
            self.data = args[0]
        elif len(args)!=2:
            raise ValueError("fit takes exactly two arguments")
        if isinstance(args[0],pd.DataFrame):
            self.data = self.Data(args[0],args[1])
        else:
            self.start, self.end = args
            self.data = None

        self.is_fitted = True
        return self
    
    def transform(self, X=None, y=None):
        return self.load()
    
    def fit_transform(self,*args,**kwargs):
        return self.fit(*args,**kwargs).transform()
    
    def load(self,start=None,end=None,**kwargs):
        start = start or self.start
        end = end or self.end
        load_kwargs = self.load_kwargs.copy()
        load_kwargs.update(kwargs)
        data = self._load_data(start,end, **load_kwargs)
        return data
    
    def load_data(self,*args,**kwargs):
        return self.load(*args,**kwargs)
    
    @lru_cache(maxsize=1)
    def _load_data(self,start=None,end=None, **kwargs):
        if self.data is not None:
            return self.data
        if start is not None and end is not None:
            start, end = utils.parse_start_end(start,end)
            period = (start.isoformat(),end.isoformat())
        else:
            period = None
        self.logger.info("Loading data...")
        # loading noaa data
        buoy_df, buoy_gdf = data.load_buoy_data(period=period, **kwargs)
        world_countries = data.load_world_borders()
        # Remove the "buoys" that appear to be in land
        world_countries = world_countries[world_countries.overlaps(geometry.box(*buoy_gdf.total_bounds.tolist()))]
        land_buoys = buoy_gdf.geometry.apply(lambda x: any(world_countries.intersects(x,align=False)))
        buoy_gdf = buoy_gdf[~land_buoys]
        buoy_df = buoy_df[buoy_df.index.get_level_values("buoy_id").isin(buoy_gdf.index.get_level_values("buoy_id").unique())]
        self.data = self.Data(buoy_df,buoy_gdf)
        return self.data

    def load_dict(self,start=None,end=None, ordered=True):
        data = self.data if self.data is not None else self.load(start,end)
        odict = data._asdict()
        if not ordered:
            odict = dict(odict)
        return odict


        
