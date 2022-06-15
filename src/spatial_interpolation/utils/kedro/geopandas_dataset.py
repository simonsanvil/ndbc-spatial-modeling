"""
A kedro dataset that loads or registers a Dataset from a parquet file containing a GeoDataFrame.
"""

import os, dotenv
from typing import Any, Dict, List
from pathlib import PurePosixPath

import fsspec
import pandas as pd
import geopandas as gpd
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path, Version


class GeopandasDataset(AbstractDataSet):

    def __init__(
        self,
        filepath: str,
        file_type: str = "parquet",
        index: bool = False,
    ) -> None:
        protocol, path = get_protocol_and_path(filepath)
        self._filepath = PurePosixPath(path)
        self._protocol = protocol
        self._fs = fsspec.filesystem(protocol)
        self._file_type = file_type
        self._keep_index = index
    
    def _load(self) -> gpd.GeoDataFrame:
        load_path = get_filepath_str(self._filepath, self._protocol)
        if self._file_type == "parquet":
            return gpd.read_parquet(load_path)
        return gpd.read_file(load_path, file_type=self._file_type)
    
    def _save(self, data: gpd.GeoDataFrame) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        if self._file_type == "parquet":
            data.to_parquet(save_path)
        else:
            data.to_file(save_path, file_type=self._file_type)
    
    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)
