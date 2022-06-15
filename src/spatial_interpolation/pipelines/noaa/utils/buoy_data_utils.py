import pandas as pd
import numpy as np

from typing import List


def get_buoy_distances_to(
    buoy_id: str, buoy_locations_df: pd.DataFrame, year: int
) -> pd.Series:
    """
    Get the distances between a given buoy and all other buoys.

    Returns a pandas Series with the euclidean distance between
    the given buoy and all other available buoys locations on the given year.
    """
    if not isinstance(year, int):
        year = pd.Timestamp(year).year

    buoy_locations_df = buoy_locations_df[buoy_locations_df.year == year]

    if buoy_id not in buoy_locations_df.buoy_id.values:
        raise ValueError(f"Buoy {buoy_id} not found in the dataframe.")

    buoy_lat, buoy_lon = buoy_locations_df.loc[buoy_locations_df.buoy_id==buoy_id, ["latitude", "longitude"]].values[0]

    buoy_locations_df = buoy_locations_df.drop_duplicates(
        subset=["buoy_id"]
    ).set_index(
        "buoy_id"
    )
    distances = buoy_locations_df[buoy_locations_df.index != buoy_id].apply(
        lambda row: np.sqrt(
            (row.latitude - buoy_lat) ** 2 + (row.longitude - buoy_lon) ** 2
        ),
        axis=1,
    )
    distances.name = "distance"
    return distances.sort_values()


def get_active_buoys_at_time(
    t: pd.Timestamp, buoy_df: pd.DataFrame, tolerance: pd.Timedelta = None
) -> List[str]:
    """
    Get the active buoys at a given time.

    Parameters
    ----------
    t: pd.Timestamp
        Time at which to get the active buoys.
    buoy_df: pd.DataFrame
        Dataframe with the buoys locations.
    tolerance: pd.Timedelta
        Timedelta of tolerance to consider a buoy active.
        If provided, the active buoys are those that are active at the time
        `t` plus or minus `tolerance`. Default is no tolerance.

    Returns
    -------
    List[str]
        List with the unique ids of the active buoys at the time given.

    """

    t = pd.Timestamp(t)
    tolerance = tolerance if tolerance is not None else pd.Timedelta(0)

    tstart, tend = t - tolerance, t + tolerance
    df = buoy_df[buoy_df.time.between(tstart, tend)]
    return df.buoy_id.unique().tolist()

def get_buoy_relative_directions_to(
    buoy_id: str, buoy_locations_df: pd.DataFrame, year: int
) -> pd.Series:
    """
    Get the directions of the buoys relative to a given buoy.

    Returns a pandas Series with the direction of the buoys relative to the
    given buoy.
    """
    if not isinstance(year, int):
        year = pd.Timestamp(year).year

    buoy_locations_df = buoy_locations_df[buoy_locations_df.year == year].copy()
    buoy_locations_df.loc[
        buoy_locations_df["longitude"]<0,"longitude"
    ] += 360

    if buoy_id not in buoy_locations_df.buoy_id.values:
        raise ValueError(f"Buoy {buoy_id} not found in the dataframe.")

    buoy_lat, buoy_lon = buoy_locations_df.loc[buoy_locations_df.buoy_id==buoy_id, ["latitude", "longitude"]].values[0]

    buoy_locations_df = buoy_locations_df.drop_duplicates(
        subset=["buoy_id"]
    ).set_index(
        "buoy_id"
    )
    directions = buoy_locations_df[buoy_locations_df.index != buoy_id].apply(
        lambda row: np.arctan2(
            row.latitude - buoy_lat, row.longitude - buoy_lon
        ),
        axis=1,
    )+np.pi/2
    directions = (directions/np.pi)*180
    directions[directions<0] += 360
    directions.name = "direction"
    return directions