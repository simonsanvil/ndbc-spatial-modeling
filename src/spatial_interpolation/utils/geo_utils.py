import geopandas as gpd
from shapely import geometry
from shapely.geometry.base import BaseGeometry

from typing import Tuple, Union


def make_polygon_of_radius(
    center: Tuple[float, float],
    radius: float = None,
    radius_meters: float = None,
    n_points: int = 100,
) -> geometry.LineString:
    """
    Make a polygon of a circle with a given radius.

    Parameters
    ----------
    center: Tuple[float,float]
        The x and y coordinates of the center of the circle.
    radius: float
        The radius of the circle.
        If not given, the radius will be calculated from the radius_meters parameter.
    radius_meters: float
        The radius of the circle in meters.
        If not given, the radius will be calculated from the radius parameter.
    n_points: int
        The number of points to use to make the polygon.
        Default is 100.

    Note: Either radius or radius_meters must be given. If both are given, radius will be used.
    """
    center_point = geometry.Point(center)
    if radius is None and radius_meters is None:
        raise ValueError("Either radius or radius_meters must be provided.")
    if radius_meters is not None and radius is None:
        radius = (
            radius_meters / 111000
        )  # meters to degrees based on 111,000 meters per degree

    return center_point.buffer(radius, resolution=n_points)

def flip_coords(
    coords: Union[Tuple[float,float],BaseGeometry, gpd.GeoDataFrame]
) -> Union[Tuple[float,float],BaseGeometry, gpd.GeoDataFrame]:
    """
    Flip the coordinates of a point or a polygon.
    """
    def extract_coords_from_poly(poly: BaseGeometry) -> Tuple[float,float]:
        try:
            coords = poly.coords
        except NotImplementedError:
            coords = poly.exterior.coords
        return coords

    if isinstance(coords, tuple):
        return coords[1], coords[0]
    elif isinstance(coords, BaseGeometry):
        return coords.__class__([(y,x) for x,y in coords.coords])
    elif isinstance(coords, gpd.GeoDataFrame):
        return coords.assign(
                geometry = coords.geometry.apply(
                    lambda geom: geom.__class__(
                        [(y,x) for x,y in extract_coords_from_poly(geom)]
                    )
                )
            )
    else:
        raise ValueError("coords must be a tuple, a shapely geometry, or a geopandas GeoDataFrame.")
