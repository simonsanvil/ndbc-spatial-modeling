"""
Extract features from geospatial and temporal-spatial data.
"""
from typing import Any, Tuple, List, Union
import logging, warnings

import tqdm
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import geometry
from pandas import IndexSlice as idx

from spatial_interpolation import data

warnings.simplefilter("ignore", UserWarning) 

def make_features(
    buoys_df: pd.DataFrame,
    buoys_gdf: gpd.GeoDataFrame,
    k_nearest: int,
    points_gdf: gpd.GeoDataFrame,
    ground_truth: Union[pd.DataFrame,str] = None,
    feature_vars: List[str] = None,
    add_directions: bool = True,
    crs:Union[int,str,dict]=4326,
    add_distance_to_shore: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Extract features based on the k-nearest buoy locations to the points to be interpolated.
    These features can be used for training a spatial-interpolation model or for inference.

    When making features for training (passing a `ground_truth`). The features are made based on the 
    location of each moore location at each available timestamp and the ground-truth
    values are returned. Otherwise, the features are made based on the location of each point in 
    `points_gdf`.

    Parameters
    ----------
    buoys_df: pd.DataFrame
        DataFrame of the recorded measurements of each buoy locations in time.
    buoys_gdf: gpd.GeoDataFrame
        GeoDataFrame of the buoy locations.    
    k_nearest: int
        The number of nearest buoys to extract features of.
    points_gdf: gpd.GeoDataFrame=None
        A GeoDataFrame of the target points to extract features from.
        The location of each point is obtained from the `geometry` column of the GeoDataFrame.
    ground_truth: pd.Series or str
        A pandas.Series of the ground-truth values of each of the target points to be interpolated.
        Index and length of the series must be the same as `points_gdf`.
        If a string, it is assumed to be the name of a column in `points_gdf`.
    feature_vars: List[str]
        The variables to extract features from.
    crs:Union[int,str,dict]=32633
        The projection of the geospatial data.
    add_directions: bool = True
        Whether to add the relative directions to the nearest locations from each target point.
    """
    
    if isinstance(crs,int):
        crs = f"epsg:{crs}"

    # get the data
    buoy_df, buoys_gdf = buoys_df.copy(), buoys_gdf.copy()
    # filter by the target variables
    feature_vars = feature_vars or buoy_df.columns.tolist()
    buoy_df = buoy_df.loc[:,feature_vars]
    # get the index of each point to interpolate
    point_ids = points_gdf.index.get_level_values(0).unique()
    # project the geometries to the same crs
    buoys_gdf_proj = buoys_gdf.to_crs(crs)
    points_gdf_proj = points_gdf[["geometry"]].to_crs(crs)
    if "time" in points_gdf.index.names:
        points_gdf_proj = points_gdf_proj.reset_index(level="time",drop=True)
        points_gdf_proj = points_gdf_proj[~points_gdf_proj.index.duplicated(keep='first')]
    if add_distance_to_shore:
        df_countries = data.load_world_borders()
        bounds = kwargs.get("map_bounds", points_gdf.total_bounds) 
        countries_geom = df_countries[df_countries.overlaps(geometry.box(*bounds))].unary_union
        points_gdf =  points_gdf.assign(distance_to_shore = lambda df: df.geometry.distance(countries_geom).values)
    # pre-compute the distances to get the k nearest neighbors
    point_dists_df = points_gdf_proj.geometry.apply(buoys_gdf_proj.distance)
    # Create the features at each available timestamp
    available_ts = points_gdf.index.get_level_values("time").unique()
    point_features = []
    for timestamp in tqdm.tqdm(available_ts,desc="making features at each point.."):
        if len(available_ts)==0:
            break
        # get available data at timestamp
        buoy_df_at_ts = buoy_df.loc[idx[:,timestamp],:].dropna()
        available_buoys = buoy_df_at_ts.index.get_level_values(0).unique()
        # Check if theres enough data       
        if len(buoy_df_at_ts)<=k_nearest or len(buoy_df_at_ts)==0:
            logging.info(f"Not enough data at timestamp {timestamp}")
            continue
        # Make features for this timestamp at each available station location
        for point_id in point_ids:
            if (point_id not in available_buoys):
                # print(f"No data for point location {point_id} at timestamp {timestamp}")
                continue
            try:
                available_points = available_buoys.drop(point_id) if point_id in available_buoys else available_buoys
                # Traffic features based on the k nearest roads
                nearest_dists = point_dists_df\
                    .loc[point_id,available_points]\
                        .sort_values()\
                            .iloc[:k_nearest]
                nearest_vals = buoy_df_at_ts.loc[nearest_dists.index,:].values.ravel()
                nearest_features_at_point = np.concatenate([
                    nearest_dists.index,
                    nearest_dists.values,
                    nearest_vals
                ])
                # concatenate features
                features_at_point = np.concatenate([np.array([point_id,timestamp]),nearest_features_at_point])
            except Exception as err:
                logging.info(f"Error at {timestamp} for point location {point_id}:",err)
                continue
            point_features.append(features_at_point)

    # Make the names of the features that have been created
    feature_names = (
        # name of index
        ["location_id","time"] +
        # name of the nearest locations
        [f"location_{i}" for i in range(k_nearest)] + 
        # distances
        [f"location_dist_{i}" for i in range(k_nearest)] + 
         # traffic values
        [f"{var}_{i%k_nearest}" for i in range(k_nearest) for var in feature_vars]
    )
    # create the dataframe
    gdf_cols = ["geometry", "distance_to_shore"] if add_distance_to_shore else ["geometry"]
    features_df = (
        points_gdf[gdf_cols]
        .assign(
            x=lambda df: df.geometry.apply(lambda g: g.x).astype("float"),
            y=lambda df: df.geometry.apply(lambda g: g.y).astype("float"),
        ).drop(columns=["geometry"])
        .astype("float")
        .join(
            pd.DataFrame(point_features,columns=feature_names)\
                .set_index(["location_id","time"])
        )
        .sort_index()
        .dropna(how="all",subset=[f"location_dist_{i}" for i in range(k_nearest)])
    )
    if ground_truth is not None:
        if isinstance(ground_truth,str):
            ground_truth = points_gdf[ground_truth]
        # Add the ground truth
        features_df = features_df.join(ground_truth)
    # compute degrees to nearest road and nearest aq station
    if add_directions and k_nearest>0:
        logging.info("Computing degrees to nearest locations")
        features_df = add_directions_of_nearest(features_df,buoys_gdf,proj_crs=crs)
    
    print(f"All features made. Total number of features: {len(features_df.columns)}. Number of points: {len(features_df)}")

    features_df = features_df[~features_df.index.duplicated(keep='first')]
    return features_df

def add_directions_of_nearest(
    features_df: pd.DataFrame, 
    buoys_gdf: gpd.GeoDataFrame,
    proj_crs: str = "epsg:4269",
) -> pd.DataFrame:
    """
    Add degrees to the nearest features in a GeoDataFrame of traffic road segments located in the same projection as the features.
    """
    ind_cols = features_df.filter(regex="^(location_\d)$").columns
    newcols = [col+'_degrees_nearest' for col in ind_cols]
    buoys_gdf["geometry"] = buoys_gdf["geometry"].apply(lambda x: x.centroid)
    nearest_degrees_df = extract_degrees_of_nearest(
        features_df, buoys_gdf, 
        ind_cols = ind_cols,
        proj_crs=proj_crs,

    )
    if len(nearest_degrees_df)==0:
        nearest_degrees_df = pd.DataFrame(columns=newcols)
    else:
        nearest_degrees_df.columns = newcols
    return pd.concat([features_df,nearest_degrees_df],axis=1)

def extract_degrees_of_nearest(df,gdf,ind_cols, proj_crs="epsg:4269"):
    coords_of_nearest = get_coords_of_nearest(df[ind_cols], gdf,proj_crs=proj_crs)
    nearest_xy_df = pd.DataFrame(coords_of_nearest.explode().tolist(),columns=['x','y'])
    degrees_nearest = np.degrees(
        np.arctan2(
            np.repeat(df.y.values,len(ind_cols))-nearest_xy_df.y,
            np.repeat(df.x.values,len(ind_cols))-nearest_xy_df.x)
    )
    nearest_degrees_df = pd.DataFrame(
        degrees_nearest.values,
        index=coords_of_nearest.explode().index
    ).assign(
        idv=np.tile(np.arange(len(ind_cols)),len(df))
    ).pivot(
        columns='idv'
    )
    return nearest_degrees_df

def get_coords_of_nearest(df,gdf,proj_crs=None):
    """
    Get the coordinates of the nearest features.
    """
    if proj_crs:
        gdf = gdf.to_crs(proj_crs)
    coords_of_nearest = df\
        .apply(
            lambda inds: 
                gdf.loc[inds.values].geometry\
                    .apply(lambda x: x.centroid)\
                        .apply(lambda c: [c.x,c.y])\
                            .values,
            axis=1
    )
    return coords_of_nearest

if __name__ == "__main__":
    # from breeze_interluft.utils import geo_utils as utils

    pass



