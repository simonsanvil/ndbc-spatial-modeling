"""
A kedro dataset that loads or registers a Dataset from an AzureML workspace.
"""

import os, dotenv
from typing import Any, Dict, List


from pathlib import PurePosixPath
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path, Version

from azureml.core import Workspace, Datastore, Dataset
from azureml.core import authentication

import pandas as pd

class AzureMLDataset(AbstractDataSet):
    """
    A kedro dataset that loads (or registers) a Dataset from an AzureML workspace as (or from) a pandas DataFrame.

    This class is useful for loading datasets that are registered in AzureML Workspaces directly from a kedro pipeline.
    Different datasets can be registered in a Kedro catalog and loaded from there.

    E.g:
    ```python
    # (catalog.yml)
    my_azureml_dataset:
        type: AzureMLDataset
        name: "my_dataset"
        index: ["id"]
        is_geodataframe: True
        auth: "cli"
    
    my_new_dataset
        type: AzureMLDataset
        name: "transformed_dataset"
        save_args:
            datastore: "my_datastore_name"
            description: "Result of my_dataset after applying a data transformation"
    ...
    
    # (pipeline.py)
    from kedro.Pipeline import Pipeline, node

    def create_pipeline():
        return Pipeline(
            [
                node(
                    func=apply_transformation_to_data,
                    inputs=["my_azureml_dataset"],
                    outputs=["my_new_dataset"],
                ),
            ]
        )
    
    def apply_transformation_to_data(df:pd.DataFrame) -> pd.DataFrame:
        # ...
        return new_df
    ```
    """

    def __init__(
        self,
        name: str,
        index: List = None,
        is_geodataframe: bool = False,
        subscription_id = os.environ.get("AZUREML_SUBSCRIPTION_ID"),
        resource_group = os.environ.get("AZUREML_RESOURCE_GROUP"),
        workspace_name = os.environ.get("AZUREML_WORKSPACE_NAME"),
        auth:str = None,
        load_args: Dict = None,
        save_args: Dict = None,
        auth_args: Dict = None,
    ):
        """
        Create new instance of BreezeDataset to load data from Breeze sensors.

        Parameters
        ----------
        name : str
            Name of the dataset to load.
        index : List, optional
            List of columns to use as index.
        is_geodataframe : bool, optional
            Whether to parse geometry column as a GeoPandas GeoSeries with `geopandas.GeoSeries.from_wkb`.
        subscription_id : str
            Azure subscription id. Defaults to the environment variable `AZUREML_SUBSCRIPTION_ID`.
        resource_group : str
            Azure ML Workspace resource group. Defaults to the environment variable `AZUREML_RESOURCE_GROUP`.
        workspace_name : str
            Name of the Azure ML Workspace where the dataset is registered or will be registered.
            Defaults to the environment variable `AZUREML_WORKSPACE_NAME`.
        auth : str
            Authentication method to use. Possible values are "cli", "service_principal" or "msi". 
            Any other value will result in the authentication method being the one 
            that azureml.core.Workspace uses by default. 
        load_args : dict, optional
            Additional arguments to pass to azureml.core.Dataset.get_by_name
        save_args : dict, optional
            Additional arguments to pass to azureml.core.Dataset.Tabular.register_pandas_dataframe
            If 'datastore' is specified as a key with the name of the datastore as its value,
            then the dataset will be registered to that datastore. Otherwise, 
            the default datastore of the AzureML workspace will be used.
            e.g: `save_args={'datastore': 'my_datastore','show_progress': True, 'description': 'my description'}`
        """
        self.workspace = Workspace(
            subscription_id, 
            resource_group, 
            workspace_name,
            auth=AzureMLDataset.get_authentication_method(auth, auth_args or {})
        )
        self._filepath = f"{subscription_id}/{resource_group}/{workspace_name}/{name}"
        self._load_args = load_args or {}
        self._save_args = save_args or {}
        self.dataset_name = name
        self._dataset_index = index
        self.is_geodataframe = is_geodataframe
        
    
    def get_authentication_method(auth_str,auth_args):
        if auth_str == "cli":
            return authentication.AzureCliAuthentication(**auth_args)
        elif auth_str == "service_principal":
            return authentication.ServicePrincipalAuthentication(**auth_args)
        elif auth_str == "msi":
            return authentication.MsiAuthentication(**auth_args)
        else:
            return None
    
    def load_azureml_dataset(self) -> Dataset:
        return Dataset.get_by_name(self.workspace, self.dataset_name, **self._load_args)
        # [
        #     f'{datastore["datastoreName"]}:{datastore["path"]}'
        #     for step in aq_geo_dataset._dataflow._get_steps() 
        #     if "datastores" in step.arguments
        #     for datastore in step.arguments["datastores"]
        # ]
    
    def _load(self) -> pd.DataFrame:
        """
        Loads the dataset from AzureML.
        """
        df = self.load_azureml_dataset().to_pandas_dataframe()
        if self._dataset_index:
            df = df.set_index(self._dataset_index)
        if self.is_geodataframe:
            import geopandas as gpd
            if "geometry" in df.columns:
                df = df.assign(geometry=lambda df: gpd.GeoSeries.from_wkb(df.geometry))
            df = gpd.GeoDataFrame(df)            
        
        return df
    
    def _save(self, data: pd.DataFrame) -> None:
        """
        Registers the dataset to the AzureML Workspace
        """
        datastore_name = self._save_args.pop("datastore", None)
        if datastore_name:
            datastore = Datastore.get(self.workspace, datastore_name)
        else:
            datastore = self.workspace.get_default_datastore()
        
        Dataset.Tabular.register_pandas_dataframe(
            data, 
            target=datastore, 
            name=self.dataset_name, 
            **self._save_args
        )
    
    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            workspace=self.workspace.name,
            dataset=self.dataset_name,
        )
