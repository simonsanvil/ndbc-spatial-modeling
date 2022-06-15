import folium
from folium import plugins

from shapely import geometry

from typing import Callable, Iterable, List, Tuple, Dict, Any, Union

from functools import partial

import geopandas as gpd
import pandas as pd
# import osmnx as ox

from .markers import IconMarker

def make_map_of_buoys(
    location: Tuple[float, float], 
    buoy_locations_df,
    highlight_buoy_ids: List[str] = None,
    marker: object = None,
    icon_params: Dict[str, Any] = None,
    **kwargs
) -> folium.Map:
    """
    Create a map  showing the locations of the buoys from the dataframe.

    Parameters
    ----------
    location: Tuple[float, float]
        The location coordinates of the center of the map.
    buoy_locations_df: pd.DataFrame
        A dataframe with columns "latitude", "longitude", and "buoy_name"
        of the buoys to be displayed on the map.
    highlight_buoy_ids: List[str] = None
        A list of buoys to highlight on the map.
    marker: object
        The marker class to use for the buoys (e.g: `folium.CircleMarker`).
        default: `folium.Marker`
    icon_params: Dict[str, Any]
        A dictionary of parameters to be passed to the folium.Icon() constructor.
    kwargs:
        Additional keyword arguments to pass to the folium.Map constructor.

    Returns
    -------
    folium.Map
        A folium map showing the locations of the buoys.
    """

    map_args = dict(
        tiles="cartodbpositron",
        zoom_start=5,
        # min_zoom=3,
        # max_zoom=11,
        scrollWheelZoom=False,
    )

    map_args.update(kwargs)
    icon_args = dict(
        icon="life-buoy", prefix="fa"
    )
    icon_args.update(icon_params or {})

    map = folium.Map(
        location=location,
        **map_args,
    )
    layer = folium.FeatureGroup(name="buoy locations",overlay=True)
    for row in buoy_locations_df.drop_duplicates(
        subset=["buoy_name"], keep="last"
    ).itertuples():
        if marker is None or row.buoy_id in (highlight_buoy_ids or []):
            loc_marker = folium.Marker(
                location=[row.latitude, row.longitude],
                popup=f"{row.buoy_name} ({row.buoy_id})",
                icon=folium.Icon(**icon_args),
            )
        else:
            loc_marker = marker(
                location=[row.latitude, row.longitude],
                popup=f"{row.buoy_name} ({row.buoy_id})",
            )
        layer.add_child(loc_marker)

    map.add_child(layer)
    plugins.Draw().add_to(map)
    heatmap = plugins.HeatMap(
        zip(buoy_locations_df.latitude, buoy_locations_df.longitude),
        radius=10,
        name="Buoy data availability",
    )
    heatmap.add_to(map)
    return map

def make_folium_map(
    coords: Union[Tuple[float, float], str],
    zoom: int = 12,
    tiles: str = "cartodbpositron",
    **kwargs,
) -> folium.Map:
    """
    Make a map from a list or tuple of coordinates.
    """
    if isinstance(coords,geometry.Point):
        coords = (coords.y,coords.x)
    map_args = {
        "zoom_start": zoom,
        "tiles": tiles,
        # "min_zoom": 11,
        # "max_zoom": 18,
        "scrollWheelZoom": False,
    }
    map_args.update(kwargs)
    return folium.Map(coords, **map_args)
    
def add_geodf_to_map(
    geodf: gpd.GeoDataFrame,
    map: folium.Map=None,
    color: Union[str,Callable] = "blue",
    fill: bool = False,
    popup:Union[str,Callable]=None,
    marker:Union[folium.Marker,str]=None,
    layer_name:str=None,
    overlay:bool=True,
    **kwargs,
) -> None:
    """
    Add a GeoDataFrame to a folium map.
    """
    if not isinstance(geodf, gpd.GeoDataFrame):
        if "geometry" in geodf.columns:
            geodf = gpd.GeoDataFrame(geodf)
        else:
            raise ValueError("geodf must be a GeoDataFrame or a dataframe with a geometry column")
        
    if map is None:
        map = make_folium_map(geodf.dissolve().centroid.iloc[0], **kwargs.pop("map_args", {}))

    if isinstance(popup, str):
        popup = lambda _: popup
    
    if marker is None:
        marker = partial(
            folium.CircleMarker,
            radius=kwargs.pop("radius", 3),
            weight=kwargs.pop("weight", 5),
        )
    elif isinstance(marker, str):
        marker = partial(IconMarker, icon=marker)

    layer = folium.FeatureGroup(name=layer_name,overlay=overlay)
    for idx,row in geodf.iterrows():
        if isinstance(row.geometry, geometry.LineString):
            poly = folium.PolyLine(
                row.geometry.coords,
                color=color if isinstance(color, str) else color(row),
                popup=idx if popup is None else popup(row),
                **kwargs
            )
        elif isinstance(row.geometry, geometry.Point):
            poly = marker(
                (row.geometry.y, row.geometry.x), 
                color=color if isinstance(color, str) else color(row),
                popup=idx if popup is None else popup(row),
                **kwargs
            )
        else:
            poly = folium.Polygon(
                row.geometry.exterior.coords, 
                color=color if isinstance(color, str) else color(row),
                fill=fill,
                popup=idx if popup is None else popup(row),
                **kwargs
            )
        layer.add_child(poly)
    map.add_child(layer)    
    # folium.LayerControl().add_to(map)
    return map

def heatmap_with_time(gdf, time_col:str="time", weight_col:str=None, mp:folium.Map=None, format=None, **kwargs):
    """
    Add a HeatMapWithTime
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        if "geometry" in gdf.columns:
            gdf = gpd.GeoDataFrame(gdf)
        else:
            raise ValueError("gdf must be a GeoDataFrame or a dataframe with a geometry column")
    if mp is None:
        map_args = {} or kwargs.pop("folium_map", {})
        if "zoom_start" in kwargs:
            map_args = {"zoom_start": kwargs.pop("zoom_start"),**map_args}
        mp = make_folium_map(gdf.dissolve().centroid.iloc[0],**map_args)
    heat_data, times = [], []
    gdf["centroids"] = gdf.geometry.centroid
    for day, d in gdf.groupby(time_col):
        heat_data.append([
            [row.centroids.y, row.centroids.x] + [row[c] for c in [weight_col] if c]
            for _, row in d.iterrows()
        ])
        times.append(str(day) if format is None else day.strftime(format))
    hm = plugins.HeatMapWithTime(heat_data, index=times, **kwargs)
    hm.add_to(mp)
    folium.LayerControl().add_to(mp)
    return mp


def add_point(
    point: Union[Tuple[float, float], geometry.Point],
    map: folium.Map,
    **kwargs,
) -> None:
    """
    Add a point to a folium map.
    """
    if kwargs.get("icon"):
        marker = IconMarker
    else:
        marker = folium.CircleMarker

    if isinstance(point, geometry.Point):
        point = (point.y, point.x)
    
    marker(point, **kwargs).add_to(map)