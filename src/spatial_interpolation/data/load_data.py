"""
Load data from azureml
"""

import logging
import os
from typing import Tuple, List
from functools import lru_cache

import pandas as pd
import geopandas as gpd
from pandas import IndexSlice as idx

from spatial_interpolation.utils import parse_start_end

def load_buoy_data(
    period: Tuple[str,str]=None,
    buoy_dataset:str=None,
    buoy_geo_dataset:str=None,
    load_geo:bool=True,
    from_azure:bool=False,
    **kwargs) -> Tuple[pd.DataFrame,gpd.GeoDataFrame]:
    """
    Load the NDBC's meteorological and oceanographic buoy data from AzureML or from local storage.

    Parameters
    ----------
    buoy_dataset: str (default: None)
        Name or path to the buoy (timeseries records) dataset.
        - If `from_azure=True` the name of the AzureML dataset of the buoy data.
        - If `from_azure=False` the glob path to the local buoy data (parquet) file(s).
        By default, the path is obtained from the `BUOY_DATASET` environment variable.
    buoy_geo_dataset: str (default: None)
        Name or path to the buoy locations dataset. If `load_geo=False`this is ignored.
        - If `from_azure=True` the name of the AzureML geo-dataset of the buoy locations.
        - If `from_azure=False` the glob path to the local (parquet) file(s) of the buoy locations.
        By default, the path is obtained from the `BUOY_GEO_DATASET` environment variable.
    load_geo: bool (default: True)
        Whether to load the buoy locations as well. By default, the buoy locations are loaded.
    from_azure: bool (default: False)
        Whether to load the data from AzureML. If False, the data is loaded from local storage.

    Returns
    -------
    pd.DataFrame, geopandas.GeoDataFrame
        The loaded buoy timeseries data records as a pandas.DataFrame and the buoy locations as a geopandas.GeoDataFrame.
    """
    buoy_dataset = buoy_dataset if buoy_dataset else os.environ.get("BUOY_DATASET")
    buoy_geo_dataset = buoy_geo_dataset if buoy_geo_dataset else os.environ.get("BUOY_GEO_DATASET")
    if from_azure:
        buoy_df = load_ml_dataset(buoy_dataset, time_period=period, as_pandas=True,**kwargs)
        if load_geo:
            buoy_locations_gdf = load_ml_dataset(buoy_geo_dataset, as_pandas=False, is_geo=True,**kwargs)
    else:
        import glob
        buoy_data_files = glob.glob(buoy_dataset)
        buoy_df = pd.concat(
            [pd.read_parquet(f) for f in buoy_data_files],
            axis="index",
        )
        if load_geo:
            buoy_locations_gdf = load_buoys_geo(buoy_geo_dataset, from_azure=False)
    
    buoy_df = buoy_df.set_index(["buoy_id","time"]).sort_index()
    if period:
        start, end = parse_start_end(*period)
        buoy_df = buoy_df.loc[idx[:,str(start):str(end)],:]
        df_year_inds = pd.MultiIndex.from_frame(
            buoy_df.index
            .to_frame()
            .assign(year=lambda df: df.time.dt.year)[["year","buoy_id"]]
            .drop_duplicates()
        ).sort_values()
        buoy_locations_gdf = buoy_locations_gdf.loc[buoy_locations_gdf.index.isin(df_year_inds)].sort_index()
        
    if load_geo:
        return buoy_df, buoy_locations_gdf
    
    return buoy_df

def load_buoys_geo(dataset_name:str=None, from_azure:bool=False, **kwargs) -> gpd.GeoDataFrame:
    """
    Load the buoy locations from AzureML or from local storage.

    Parameters
    ----------
    dataset_name: str (default: None)
        The name of the dataset to load.
        If None, the dataset name is obtained from the `BUOY_GEO_DATASET` environment variable.
    from_azure: bool (default: False)
        Whether to load the data from AzureML. If False, the data is loaded from local storage.

    Returns
    -------
    geopandas.GeoDataFrame
        The loaded buoy locations.
    """
    dataset_name = dataset_name if dataset_name else os.environ.get("BUOY_GEO_DATASET")
    if from_azure:
        buoys_geo = load_ml_dataset(dataset_name, is_geo=True,**kwargs)
    else:
        import glob
        buoy_geo_files = glob.glob(dataset_name)
        buoys_geo = pd.concat(
            [gpd.read_parquet(f) for f in buoy_geo_files],
            axis="index",
        )
    
    buoys_geo = buoys_geo.drop_duplicates(subset=["buoy_id","year"]).set_index(["year","buoy_id"]).sort_index()
    buoys_geo.crs = "epsg:4326"
    return buoys_geo
    

def load_world_borders(dataset_name:str=None,preprocess:bool=True,from_azure:bool=False,**kwargs) -> gpd.GeoDataFrame:
    """
    Load the [World Borders Dataset](https://thematicmapping.org/downloads/world_borders.php) as sourced from Bjorn Sandvik's [thematicmapping.org](https://thematicmapping.org)
    
    The dataset can be loaded from an AzureML workspace where the dataset is expected to be registered with `from_azure=True`. 
    Otherwise, the dataset is either downloaded from the source website or loaded from local storage 
    based on the `WORLD_BORDERS_DATASET` environment variable is set.

    Parameters
    ----------
    dataset_name: str (default: None)
        - If `from_azure=False` (the default) the path to the local country borders (.shp) file.
        - If `from_azure=True` the name of the AzureML dataset of the country borders.
        
        If `dataset_name` is not passed, this is attempted to be obtained from the `WORLD_BORDERS_DATASET` environment variable.
        If the environment variable is also not set, the dataset is downloaded from the thematicmapping.org website.
    preprocess: bool (default: True)
        Whether to preprocess the data by changing the column names and setting the index after loading.
    from_azure: bool (default: False)
        Whether to load the data from an AzureML dataset (based on `dataset_name`).

    Returns
    -------
    geopandas.GeoDataFrame
        The loaded world borders dataset as a geopandas.GeoDataFrame with a `geometry` column of multipolygons.

    Notes
    -----
    You can set `dataset_name` to `False` if you want to bypass the environmental variables and automatically download the dataset from the website.

    References
    ------
    - Bjorn Sandvik's World Borders Dataset: [thematicmapping.org](https://thematicmapping.org/downloads/world_borders.php)

    """
    dataset_name = dataset_name if dataset_name or dataset_name is False else os.environ.get("WORLD_BORDERS_DATASET")
    if dataset_name:
        if from_azure:
            world_borders_gdf = load_ml_dataset(dataset_name, as_pandas=False, is_geo=True,**kwargs)
        else:
            world_borders_gdf = gpd.read_file(dataset_name)
    else:
        import requests, io
        # Download the country borders dataset zip file
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        url = "https://thematicmapping.org/downloads/TM_WORLD_BORDERS-0.3.zip"
        print(f"Downloading country borders dataset from {url}...")
        r = requests.get(url, headers=headers)
        # read the zip file to geopandas
        world_borders_gdf = gpd.read_file(io.BytesIO(r.content))
    
    if preprocess:
        world_borders_gdf = (
            world_borders_gdf
            .rename(columns={"NAME": "country_name", "ISO2": "code"})
            .set_index(["code"])
        )
    return world_borders_gdf

def load_ml_dataset(dataset_name:str, as_pandas:bool=True, is_geo=False, time_period:Tuple[str,str]=None, workspace=None, auth_method:str=None)->pd.DataFrame:
    """
    Load a dataset from AzureML.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to load.
    as_pandas: bool (default: True)
        Whether to return the data as a pandas.DataFrame. Otherwise, return an azureml.core.Dataset object.
    is_geo: bool
        Whether to load the dataset as a geopandas.GeoDataFrame. Assumes the dataset contains a
        `geometry` column with the wkb-encoded geometries of the observations.
    time_period: Tuple[str,str]
        The start and end time of the data to load. Assumes the data has a `time` column.
    workspace: azureml.core.Workspace
        The workspace to load the dataset from. 
        If None, the workspace is attempted to be obtained based on environmental variables config.
    auth_method: str
        The authentication method of the AzureML workspace if `workspace` is None.
    
    Returns
    -------
    pd.DataFrame or geopandas.GeoDataFrame
        The loaded data as a pandas.DataFrame or geopandas.GeoDataFrame (if `is_geo=True`).
    """
    from azureml.core import Dataset
    ws = get_ml_workspace(auth=auth_method) if workspace is None else workspace
    dataset = Dataset.get_by_name(ws, name=dataset_name)
    if time_period:
        time_period = [str(s) for s in parse_start_end(*time_period)]
        dataset = dataset.filter((dataset.time >= time_period[0])&(dataset.time <= time_period[1]))
    if not as_pandas:
        return dataset
    df = dataset.to_pandas_dataframe()
    if is_geo:
        df = gpd.GeoDataFrame(df)
        df["geometry"] = gpd.GeoSeries.from_wkb(df["geometry"])
        df.crs = 'epsg:4326'

    return df


@lru_cache(maxsize=1)
def get_ml_workspace(auth=None):
    """
    Get the Azure ML workspace. If auth_method is specified, use that method to authenticate.
    auth_method can be one of "cli", "service_principal", "msi", or "interactive". 
    Default is None to use the default authentication method.

    Returns:
    --------
    azureml.core.Workspace
        The Azure ML workspace.
    """
    from azureml.core import Workspace
    # read environment variables
    subscription_id = os.environ.get("AZUREML_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZUREML_RESOURCE_GROUP")
    workspace_name = os.environ.get("AZUREML_WORKSPACE_NAME")
    service_principal_id = os.environ.get("AZUREML_SERVICE_PRINCIPAL_ID")
    service_principal_password = os.environ.get("AZUREML_SERVICE_PRINCIPAL_PASSWORD")
    tenant_id = os.environ.get("AZUREML_TENANT_ID")
    config_file = os.environ.get("AZUREML_CONFIG_FILE")
    auth_method = auth or os.environ.get("AZUREML_AUTH_METHOD")

    if auth_method is not None and isinstance(auth_method,str):
        from azureml.core.authentication import (
            MsiAuthentication,
            AzureCliAuthentication,
            InteractiveLoginAuthentication,
            ServicePrincipalAuthentication
        )
        try:
            if auth_method=="msi":
                 auth = MsiAuthentication()
            elif auth_method=="cli":
                auth = AzureCliAuthentication()
            elif auth_method=="interactive":
                auth = InteractiveLoginAuthentication()
            elif auth_method=="service_principal":
                auth = ServicePrincipalAuthentication(
                    tenant_id=tenant_id,
                    service_principal_id=service_principal_id,
                    service_principal_password=service_principal_password
                )
            else:
                auth = None
        except Exception as e:
            print(f"Authentication failed with {auth_method}")
            raise e
        
    if config_file:
        ws = Workspace.from_config(path=config_file, auth=auth)
    else:
        ws = Workspace(subscription_id, resource_group, workspace_name, auth=auth)
    return ws

