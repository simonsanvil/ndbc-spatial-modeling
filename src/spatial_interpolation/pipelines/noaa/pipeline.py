"""
Pipelines to extract and process the data
as well as training, and evaluating the models
to interpolate NOAA's NDBC south-atlantic wave data
"""

from kedro.pipeline import Pipeline, node
from typing import List, Dict

from spatial_interpolation.pipelines.noaa import (
    metadata_extraction as meta_extraction,
    data_extraction as extraction,
    data_processing as processing
)

import pandas as pd

def create_processing_pipeline(**kwargs):

    def make_dict_of_dfs(dfs:List[pd.DataFrame]) -> Dict[str,pd.DataFrame]:
        return {f"{i}":df for i, df in enumerate(dfs)}

    return Pipeline(
        [
            node(
                processing.parse_raw_buoy_files,
                ["ndbc_raw_buoy_data","params:num_processes"],
                "raw_buoy_dfs",
                name="parse_raw_buoy_files",
            ),
            node(
                make_dict_of_dfs,
                ["raw_buoy_dfs"],
                "raw_buoy_data_parquets",
                name="make_dict_of_raw_dfs",
            ),
            node(
                processing.process_raw_buoy_dfs,
                ["raw_buoy_data_parquets","params:num_processes"],
                "processed_buoy_dfs",
                name="process_raw_buoy_dfs",
            ),
            node(
                make_dict_of_dfs,
                ["processed_buoy_dfs"],
                "buoy_data_parquets",
                name="make_dict_of_processed_dfs",
            ),
        ]
    )

def create_metadata_pipeline(**kwargs):
    def get_buoy_ids_from_json(buoys_dict:Dict[str,List[str]]) -> List[str]:
        return [id_ for l in buoys_dict.values() for id_ in l]

    return Pipeline(
        [
            node(
                get_buoy_ids_from_json,
                ["buoy_ids_dict"],
                "buoy_ids",
                name="get_buoy_ids_from_json",
            ),
            node(
                meta_extraction.extract_buoy_locations,
                ["buoy_ids"],
                "buoy_locations_df",
                name="extract_buoy_locations",
            ),
            node(
                meta_extraction.get_buoy_stdmet_index_df,
                ["stdmet_index_html"],
                "stdmet_index_df",
                name="extract_stdmet_index",
            ),
            node(
                meta_extraction.make_ndbc_metadata_df,
                ["buoy_locations_df", "stdmet_index_df"],
                "ndbc_metadata_df",
                name="make_ndbc_metadata_df",
            )
        ]
    )