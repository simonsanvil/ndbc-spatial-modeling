import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Union
from collections import namedtuple

import pandas as pd
import numpy as np
import geopandas as gpd

from pandas import IndexSlice as idx
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from spatial_interpolation import utils

@dataclass
class TrainTestSplit:
    train:pd.Series
    test:pd.Series

    def __repr__(self) -> str:
        return f"TrainTestSplit(train shape:{self.train.shape}, test shape:{self.test.shape})"
    
    def __deepcopy__(self, memo):
        return TrainTestSplit(self.train.copy(), self.test.copy())
    
    def copy(self):
        return TrainTestSplit(self.train.copy(), self.test.copy())
    
    @classmethod
    def split_index(
        cls,
        index:pd.Series,
        strategy:str,
        *,
        test_size:Union[int,float]=0.2,
        priorize_test_set:bool=False,
        split_on_unique:bool=False,
        index_level:int=0,
        seed:int=None,
        **kwargs
    ) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """
        Splits the given index into training and test.
        """
        if seed is not None:
            np.random.seed(seed)
        if len(index) == 0:
            return None
        if split_on_unique:
            if isinstance(index,pd.MultiIndex):
                index = index.get_level_values(index_level)
            index = index.unique()
        if isinstance(test_size, float):
            test_size = int(len(index)*test_size)
        else:
            test_size = test_size    
        if test_size >= len(index):
            logging.warning(f"Test size is larger or equal to the number of index points.")
            if priorize_test_set:
                logging.warning(f"Train set will be empty since {priorize_test_set=}. "\
                f"Test size will be the length of the index ({len(index)}). ")
                test_size = len(index)
            else:
                logging.warning(f"Prioritizing train set since {priorize_test_set=}. "\
                    f"Test size will be the length of index minus 1 ({len(index)-1}).")
                test_size = len(index)-1
        logging.info(f"Making splits of index of size {len(index)} with strategy: {strategy} and test size {test_size}.")
        # split the traffic data into training and validation
        if strategy is None:
            # return the same data for training and an empty array for validation
            train_index, val_index = index, np.array([])
        elif strategy == "random":
            # split the indices randomly into training and validation by test_size
            val_sample = np.zeros(len(index), dtype=int)
            val_sample[:test_size] = 1
            np.random.shuffle(val_sample)
            val_index = index[val_sample.astype(bool)]
            train_index = index[~val_sample.astype(bool)]
        elif strategy == "time":
            # split leaving the last test_size% of the data for validation    
            train_index = index[:-test_size]
            val_index = index[-test_size:]
        else:
            raise NotImplementedError(f"Unknown strategy {strategy=}")
            
        # return the training and validation indices
        return cls(train_index, val_index)
    
@dataclass
class ValidationSplit:
    start:str
    end:str
    values: pd.Series
    name: str
    validation_strategy: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ValidationSplit(name: {self.name.capitalize()}, start: {self.start}, end: {self.end}, length: {len(self.values)})"
    
    def __str__(self) -> str:
        return f"ValidationSplit(name: {self.name.capitalize()}, start: {self.start}, end: {self.end}, length: {len(self.values)})"

    def splitter(X,split_on=None):
        if isinstance(split_on, pd.DataFrame):
            split_on = X.index.isin(split_on.index)
        elif isinstance(split_on, (list,np.array,pd.Series,pd.Index,pd.MultiIndex)):
            split_on = X.index.isin(split_on)   
        
        yield ~split_on, split_on
    

def make_validation_splits(
    buoys_df:pd.DataFrame,
    buoys_gdf:gpd.GeoDataFrame,
    validation_strategy:Union[str,Dict]=None,
    start:str=None,
    end:str=None,
    ) -> List[ValidationSplit]:
    """
    Trains a spatial interpolation model for the interluft project.
    """
    logging.info("Making validation splits...")

    # project the geographic data to epsg 4326
    buoys_gdf.geometry.crs = 'epsg:4326'
    eval_locations = validation_strategy.pop("eval_locations",None)
    start = start or validation_strategy.pop("start",None)
    end = end or validation_strategy.pop("end",None)

    if eval_locations:
        # filter out locations that are in the evaluation set (not training or test)
        if isinstance(eval_locations, (float,int)):
            locations = buoys_df.index.get_level_values("station_id").unique().tolist()
            size_eval = int(eval_locations) if isinstance(eval_locations,int) or eval_locations>1 else int(len(locations)*eval_locations)
            eval_locations = np.random.choice(locations, size=size_eval, replace=False)
        buoys_gdf = buoys_gdf.loc[~buoys_gdf.index.get_level_values(0).isin(eval_locations)]
        buoys_df = buoys_df.loc[~buoys_df.index.get_level_values(0).isin(eval_locations)]
        logging.info(f"Evaluation set locations: {eval_locations}")
    
   # make the validation splits
    if isinstance(validation_strategy, str):
        validation_strategy = {"strategy":validation_strategy}
    num_splits = validation_strategy.pop("num_splits", 5)
    validation_strategy = validation_strategy or {}
    # make time splits of data index 
    logging.info(f"Dividing the available data into {num_splits} splits for training and testing.")
    logging.info(f"{len(eval_locations)} station(s) will be used for evaluation.")
    start = buoys_df.index.get_level_values(-1).min() if start is None else start
    end = buoys_df.index.get_level_values(-1).max() if end is None else end
    start, end = utils.parse_start_end_times(start, end)
    logging.info("Available data goes from {} to {}.".format(start, end))
    time_splits = utils.make_time_splits(start, end, num_splits=num_splits, group_by_pairs=True)
    # create a list of Validation splits for each timeframe
    logging.info(f"Dividing into train and test splits w/ time splits: {time_splits}...")
    train_test_splits:List[TrainTestSplit] = [
        TrainTestSplit.split_index(
            buoys_df.loc[idx[:,split_start:split_end],:].index,
            **validation_strategy,
        )
        for split_start, split_end in time_splits
    ]
    train_test_splits = [
        (split_start,split_end,split) 
        for (split_start,split_end),split 
        in zip(time_splits, train_test_splits) 
        if split is not None
    ]
    training_splits = [
        ValidationSplit(
            start=split_start,
            end=split_end,
            values=split.train,
            name=f"Train split #{i+1}/{len(train_test_splits)}",
            validation_strategy=validation_strategy,
        )
        for i, (split_start, split_end, split) in enumerate(train_test_splits)
    ]
    test_splits = [
        ValidationSplit(
            start=split_start,
            end=split_end,
            values=split.test.get_level_values(0).unique(),
            name=f"Test split #{i+1}/{len(train_test_splits)}",
            validation_strategy=validation_strategy,
        )
        for i, (split_start, split_end, split) in enumerate(train_test_splits)
    ]
    eval_splits = [
        ValidationSplit(
            start=start,
            end=end,
            values=eval_locations,
            name="Evaluation split",
            validation_strategy={},
        )
    ]
    return training_splits, test_splits, eval_splits

