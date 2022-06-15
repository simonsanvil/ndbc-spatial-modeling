"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from spatial_interpolation.pipelines.noaa import pipeline as noaa_pipes
from spatial_interpolation.pipelines.feature_interpolation import pipeline as feature_interpolation_pipes

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    noaa_processing_pipe = noaa_pipes.create_processing_pipeline()
    noaa_metadata_pipe = noaa_pipes.create_metadata_pipeline()
    feature_interpolation_pipe = feature_interpolation_pipes.make_pipelines()

    return {
        "feature_interpolation": sum(v for k,v in feature_interpolation_pipe.items() if k!="move_features"),
        "noaa-metadata" : noaa_metadata_pipe,
        "__default__": noaa_metadata_pipe, # + noaa_processing_pipe
        "ndbc-data-processing": noaa_processing_pipe,
        **feature_interpolation_pipe
    }
