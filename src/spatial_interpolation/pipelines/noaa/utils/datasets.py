from kedro.io import AbstractDataSet, PartitionedDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path
from typing import Any, Dict, List
from pathlib import PurePosixPath

import os, gzip, logging
import pandas as pd

from spatial_interpolation.pipelines.noaa import data_processing as noaa_proceesing

class RawBuoyDataSet(AbstractDataSet):
    """
    A dataset that loads and parses raw data obtained from 
    [NOAA's NDBC buoy stdmet historical data](https://www.ndbc.noaa.gov/data/stdmet/).
    """
    
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ):
        self._filepath = PurePosixPath(filepath)
        self._load_args = load_args or {}
        self._save_args = save_args or {}

    def _load(self) -> pd.DataFrame:
        filename = os.path.basename(self._filepath)
        file_content_str = gzip.open(self._filepath, 'rb').read().decode('utf-8')
        try:
            df = noaa_proceesing.parse_raw_buoy_data(file_content_str)
        except Exception as err:
            logging.error(f"Error parsing raw buoy data file {filename}: {err}")
            return pd.DataFrame()

        df["filename"] = filename

        return df

    def _save(self, data: pd.DataFrame) -> None:
        data.to_feather(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

class RawBuoyPartitionedDataSet(PartitionedDataSet):
    """
    A dataset that loads a folder of raw txt data obtained from 
    [NOAA's NDBC buoy stdmet historical data](https://www.ndbc.noaa.gov/data/stdmet/).
    and parses it into a single dataframe.

    Replaces the PartitionedDataSet.load() method to return a DataFrame.
    """

    def load(self) -> pd.DataFrame:
        dict_of_data = super().load()

        df = pd.concat(
            [delayed() for delayed in dict_of_data.values()]
        )
         
        return df

