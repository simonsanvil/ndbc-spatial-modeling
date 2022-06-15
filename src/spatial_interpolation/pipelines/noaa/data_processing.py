import os
import pandas as pd
import numpy as np

from typing import List, Union

import tqdm
import re, io, logging, time
import multiprocessing

logger = logging.getLogger("noaa.data_processing")


def process_raw_buoy_data(
    raw_buoy_files: List[str],
    num_cores: int = 2,
) -> pd.DataFrame:
    """
    Make the processed dataset of buoy data obtained from
    NOAA's Marine Environmental Buoy Database

    Parameters
    ----------
    raw_buoy_files: List[str] or dict
        List of raw dataframes as obtained from `data_processing.parse_raw_buoy_data`
        or a dictionary obtained from a `kedro.io.PartitionedDataSet` `load` method.

    Returns
    -------
    pd.DataFrame
        Processed dataframe of the raw data obtained from NOAA's NDBC buoy stdmet historical data.
    """

    start = time.time()

    raw_buoy_dfs = parse_raw_buoy_files(raw_buoy_files, num_cores)
    processed_buoy_dfs = process_raw_buoy_dfs(raw_buoy_dfs, num_cores)

    end = time.time()

    logger.info(f"Total time taken: {end - start} seconds")

    return pd.concat(processed_buoy_dfs, axis="index")


def parse_raw_buoy_files(
    raw_buoy_files: List[str],
    num_cores: int = 2,
) -> List[pd.DataFrame]:
    """
    Parse raw txt files of buoy data obtained from
    NOAA's Marine Environmental Buoy Database

    Parameters
    ----------
    raw_buoy_files: List[str] or dict
        List of raw txt.gz filepaths as obtained from NOAA's Marine Environmental Buoy Database
        or a dictionary obtained from a `kedro.io.PartitionedDataSet` `load` method.
    num_cores: int
        Number of cores to use for parallel processing.

    Returns
    -------
    List[pd.DataFrame]
        Dataframes with the parsed raw data obtained from NOAA's NDBC buoy stdmet historical data.
    """

    if num_cores == -1 or num_cores is None:
        multiprocessing.cpu_count()
    else:
        num_cores = min(num_cores, multiprocessing.cpu_count())

    start_1 = time.time()
    raw_buoy_dfs = []

    with multiprocessing.Pool(num_cores) as pool:
        logger.info(f"Parsing {len(raw_buoy_files)} files with {num_cores} cores...")
        if isinstance(raw_buoy_files, dict):
            pool_func = load_dataset_partition
            file_vals = raw_buoy_files.values()
        else:
            pool_func = parse_raw_buoy_data
            file_vals = raw_buoy_files

        for df in tqdm.tqdm(
            pool.imap_unordered(pool_func, file_vals),
            total=len(file_vals),
        ):
            raw_buoy_dfs.append(df)

    end_1 = time.time()
    logger.info(f"Raw txt data parsing completed in {end_1 - start_1} seconds")

    return raw_buoy_dfs

def process_raw_buoy_dfs(
    raw_buoy_dfs: List[pd.DataFrame],
    num_cores: int = 2,
) -> List[pd.DataFrame]:
    """
    Process raw dataframes of buoy data as obtained `from data_processing.parse_raw_buoy_data`

    Parameters
    ----------
    raw_buoy_dfs: List[pd.DataFrame]
        List of raw dataframes as obtained from `data_processing.parse_raw_buoy_data`
    num_cores: int
        Number of cores to use for parallel processing.
    
    Returns
    -------
    List[pd.DataFrame]
        List of processed dataframes.
    """    

    if isinstance(raw_buoy_dfs, dict):
        logger.info("Loading raw dataframes from kedro.io.PartitionedDataSet...")
        raw_buoy_dfs = [load_df() for load_df in raw_buoy_dfs.values()]
        logger.info(f"{len(raw_buoy_dfs)} raw dataframes loaded as: {type(raw_buoy_dfs[0])}")

    logger.info("Columns of first: " + str(raw_buoy_dfs[0].columns))

    start_2 = time.time()
    processed_buoy_dfs = []
    with multiprocessing.Pool(num_cores) as pool:
        logger.info(
            f"Processing raw data from {len(raw_buoy_dfs)} dfs with {num_cores} cores"
        )
        for df in tqdm.tqdm(
            pool.imap_unordered(process_raw_stdmet_df, raw_buoy_dfs),
            total=len(raw_buoy_dfs),
        ):
            if len(df) > 0:
                processed_buoy_dfs.append(df)

    end_2 = time.time()
    logger.info(f"Raw data processing completed in {end_2 - start_2} seconds")
    logger.info(f"Number of final dfs: {len(processed_buoy_dfs)}")

    return processed_buoy_dfs
    # pd.concat(processed_buoy_dfs, axis="index")


def process_raw_stdmet_df(raw_stdmet_df: pd.DataFrame) -> pd.DataFrame:

    # logger.info(f"Processing dataframe with shape {raw_stdmet_df.shape}...")

    if len(raw_stdmet_df) == 0:
        return pd.DataFrame()

    df = (
        raw_stdmet_df.sort_values(by="time")  # pd.concat(raw_stdmet_dfs, axis="index")
        .reset_index(drop=True)
        .assign(buoy_id=lambda df: df.filename.str.split("h").str[0])
        .drop(columns="filename")
    )
        
    # imput as nan the values from string columns that dont have a numeric pattern
    # to later cast them as float:
    numeric_cols = df.columns[(df.dtypes == float) | (df.dtypes == int)].tolist()
    for col in df.columns.difference(["time","buoy_id"]+numeric_cols):
        df.loc[~df[col].str.match(r"^[0-9]+[.]*[0-9]*$"),col] = np.nan
        df[col] = df[col].astype(float, errors="ignore")

    # Replace 999 and 99 missing values with NaNs
    numeric_cols = df.columns[(df.dtypes == float) | (df.dtypes == int)].tolist()
    largest_val_per_col = df.loc[:, numeric_cols].max()
    nan_vals_per_col = largest_val_per_col[largest_val_per_col % 9 == 0]
    df.loc[:, nan_vals_per_col.index] = df.loc[:, nan_vals_per_col.index].apply(
        lambda col: col.replace(nan_vals_per_col[col.name], np.nan)
    ).dropna(
        how="all", 
        axis="rows"
    )

    # Some attributes like wind direction and air pressure
    # have changed name over time. We merge the columns respectively
    # df["WDIR"] = np.where(pd.isnull(df["WDIR"]), df["WD"], df["WDIR"])
    # df["PRES"] = np.where(pd.isnull(df["PRES"]), df["BAR"], df["PRES"])

    column_names = {
        "WDIR": "wind_direction",
        "WD": "wind_direction",
        "WSPD": "wind_speed",
        "GST": "wind_gust",
        "WVHT": "wave_height",
        "DPD": "dominant_wave_period",
        "APD": "average_wave_period",
        "MWD": "mean_wave_direction",
        "PRES": "sea_level_pressure",
        "BAR": "sea_level_pressure",
        "ATMP": "air_temperature",
        "WTMP": "water_temperature",
        "DEWP": "dew_point",
        "VIS": "visibility",
        "TIDE": "tide_level",
        "BAR": "barometer",
    }
    column_names = {k: v for k, v in column_names.items() if k in df.columns}
    df = df.rename(
        columns=column_names
    ).astype(
        {col:float for col in column_names.values()},
        errors='ignore'
    )
    df = df[["time", "buoy_id"] + df.columns.difference(["time", "buoy_id"]).tolist()]

    return df


def load_dataset_partition(partitioned):
    return partitioned()


def parse_raw_buoy_data(txt_data: str) -> pd.DataFrame:
    """
    Parse the raw buoy data from a .txt file obtained from the NOAA NDBC website
    and return as pandas dataframe.
    """
    # logger.info("Parsing raw Buoy data...")

    if os.path.isfile(txt_data):
        with open(txt_data, "r") as f:
            txt_data = f.read()

    # Parse the raw text to be able to differentiate the columns
    table_str = re.sub(r"[ \t]{2,}", " ", txt_data)

    # Read as pandas dataframe
    df = pd.read_csv(
        io.StringIO(table_str), sep=" ", skiprows=[0, 1], header=None, index_col=None
    ).dropna(how="all", axis=1)

    df.columns = table_str.split("\n")[0].strip("#").strip().split(" ")

    # Parse date and time columns
    date_cols = df.columns[df.columns.str.match("YY+|MM+|DD+")].tolist()
    time_cols = df.columns[df.columns.str.match("hh+|mm+")].tolist()
    df[date_cols + time_cols] = df[date_cols + time_cols].astype(str)
    df[date_cols + time_cols] = (
        df[date_cols + time_cols].astype(str).apply(lambda col: col.str.zfill(2))
    )

    df["time"] = (
        df[date_cols].agg("-".join, axis="columns")
        + " "
        + df[time_cols].agg(":".join, axis="columns")
    )
    df = df.astype({"time": "datetime64"})#.drop(columns=date_cols + time_cols)
    df = df[["time"] + list(df.columns[:-1])]

    return df
