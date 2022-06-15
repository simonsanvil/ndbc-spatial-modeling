from abc import ABC
from dataclasses import dataclass
import logging
from typing import Dict, Tuple, Iterable, Union
from collections import OrderedDict

@dataclass
class BaseData(ABC):

    def __iter__(self):
        """Iterate through all the datasets that conform this data"""
        data_attrs = self.__dataclass_fields__
        for attr in data_attrs:
            yield getattr(self, attr)

    def _asdict(self):
        return OrderedDict([(attr, getattr(self, attr)) for attr in self.__dataclass_fields__])

    def split_eval(self, method:str, params=None, **kwargs) -> Tuple["BaseData", "BaseData"]:
        """
        Split the dataset into train and test sets using the specified method.
        """
        split_params = params or {}
        split_params.update(kwargs)
        if method == "random":
            return self.split_random(**split_params)
        elif method == "slice":
            return self.split_slice(**split_params)
        elif method == "temporal":
            return self.split_temporal(**split_params)
        elif method == "stratified":
            return self.split_stratified(**split_params)
        else:
            if hasattr(self, method):
                return getattr(self, method)(**split_params)
            raise ValueError(f"Invalid split method: {method}")

    def split_random(self, test_size=0.3) -> Tuple["BaseData", "BaseData"]:
        data_attrs = self.__dataclass_fields__
        datasets = [getattr(self, attr) for attr in data_attrs]
        train_dfs = [df.sample(frac=1 - test_size) for df in datasets]
        test_dfs = [df.loc[~df.index.isin(train_df.index)] for train_df, df in zip(train_dfs, datasets)]
        train_dataset = self.__class__(*train_dfs)
        test_dataset = self.__class__(*test_dfs)
        return train_dataset, test_dataset
    
    def split_slice(
        self, 
        train:Union[slice,Tuple[slice],Dict[str,slice]]=None, 
        test:Union[slice,Tuple[slice],Dict[str,slice]]=None, 
        dataset=None, 
        level=None,
        **kwargs
    ) -> Tuple["BaseData", "BaseData"]:
        """
        Splits the given dataset into training and test by providing slices of the index of each dataset.

        Parameters
        ----------
        train : Union[slice,Tuple[slice],Dict[str,slice]]
            Slice of the index of the data to be used for training. If not given, the complement of the test slice is used.
            Can be a single slice or a tuple of slices, or a dictionary of slices where the keys are the dataset names.
        test : Union[slice,Tuple[slice],Dict[str,slice]]
            Slice of the index of the data to be used for testing. If not given, the complement of the train slice is used.
            Can be a single slice or a tuple of slices, or a dictionary of slices where the keys are the dataset names.
        dataset : str, optional
            Name of the dataset to split when train and test are not dictionaries specifying the slices of each dataset.
            If None, the first dataset is used.
        level : int, optional
            Index of the level to split when train and test are just the index of a MultiIndex.
            If None, the first level is used.

        Returns
        -------        
        Tuple[BaseData, BaseData]
            Tuple of the training and test datasets.

        Usage
        -----
        >>> import pandas as pd
        >>> import geopandas as gpd
        >>> from pandas import IndexSlice as idx
        >>> from spatial_interpolation.data.base_dataset import BaseData
        
        >>> df_A = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6})
        >>> df_B = pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})
        >>> df_A.index = pd.date_range("2020-01-01", "2020-01-03")
        >>> df_B.index = pd.date_range("2020-01-06", "2020-01-08")
        >>> data = BaseData(df_A, df_B)
        >>> train, test = data.split_slice(train=idx[:"2020-01-03"])
        >>> train
        <BaseData(train)>
                    a  b
        2020-01-01  1  4
        2020-01-02  2  5
        >>> test
        <BaseData(test)>
                    a  b
        2020-01-03  3  6
        """
        if "eval" in kwargs and test is None:
            test = kwargs["eval"]
        if train is None and test is None:
            raise ValueError("Either a train or a test index slice must be specified.")
        # print(f"Splitting dataset with train={train} and test={test}")
        if isinstance(dataset,list):
            dataset_names = dataset
        elif isinstance(dataset,str):
            dataset_names = [dataset]
        elif dataset in [all,...,None]:
            dataset_names = [attr for attr in self.__dataclass_fields__]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        datasets = {attr:getattr(self, attr) for attr in self.__dataclass_fields__}
        default_train_dict, default_test_dict = datasets.copy(), datasets.copy()
        single_split_inds = [0] if dataset is None else [i for i,s in enumerate(datasets.keys()) if s in dataset_names]

        def make_dfs_dict_from_split(split, split_name=""):
            if split is None:
                return None
            if isinstance(split, dict):
                split_dfs_dict = {dataset:getattr(self, dataset).loc[split[dataset]] for dataset in split}
            elif isinstance(split, (slice,tuple)):
                split_dfs_dict = {dataset_names[i] : datasets[dataset_names[i]].loc[split] for i in single_split_inds}
            elif isinstance(split, list):
                split_dfs_dict = {dataset_names[i] : datasets[dataset_names[i]].loc[datasets[dataset_names[i]].index.isin(split)] for i in single_split_inds}
            elif level is not None:
                level_vals = [getattr(self, dataset_name).index.get_level_values(level) for dataset_name in dataset_names]
                split_dfs_dict = {dataset_names[i] : datasets[dataset_names[i]].loc[level_vals[i].isin(split)] for i in single_split_inds}
            else:
                raise ValueError("Invalid split.")
            print(f"Split {split_name} obtained with dataset(s)={list(split_dfs_dict.keys())} and shape(s)={[df.shape for df in split_dfs_dict.values()]}")
            return split_dfs_dict
        
        train_dfs_dict = make_dfs_dict_from_split(train, split_name="train")
        test_dfs_dict = make_dfs_dict_from_split(test, split_name="test") 
        if train_dfs_dict is None:
            train_dfs_dict = {
                dataset:default_train_dict[dataset].loc[~default_train_dict[dataset].index.isin(test_dfs_dict[dataset].index)] 
                for dataset in test_dfs_dict
            }
        if test_dfs_dict is None:
            test_dfs_dict = {
                dataset:default_test_dict[dataset].loc[~default_test_dict[dataset].index.isin(train_dfs_dict[dataset].index)] 
                for dataset in train_dfs_dict
            }
        
        default_train_dict.update(train_dfs_dict)
        default_test_dict.update(test_dfs_dict)
        train_dataset = self.__class__(**default_train_dict)
        test_dataset = self.__class__(**default_test_dict)
        return train_dataset, test_dataset
        
            
    def split_temporal(self, test_size) -> Tuple["BaseData", "BaseData"]:
        raise NotImplementedError()
    
    def split_stratified(self, test_size) -> Tuple["BaseData", "BaseData"]:
        raise NotImplementedError()
    
    




            
