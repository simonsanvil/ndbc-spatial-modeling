# import logging
# from dataclasses import dataclass, field
# from typing import Dict, Iterable, List, Tuple, Union
# from collections import namedtuple

# import pandas as pd
# import numpy as np
# import geopandas as gpd

# from pandas import IndexSlice as idx
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# from spatial_interpolation import utils

# from .base_dataset import BaseDataset

# def eval_split(data, strategy:str, **kwargs):
#     if strategy == "random":
#         return eval_split_random(data, **kwargs)
#     elif strategy == "stratified":
#         return eval_split_stratified(data, **kwargs)
#     elif strategy == "temporal":
#         return eval_split_temporal(data, **kwargs)
#     elif strategy == "index":
#         return eval_split_index(data, **kwargs)
#     else:
#         raise ValueError(f"Unknown strategy: {strategy}")

# def eval_split_random(data, test_size:float=0.2, seed:int=None, **kwargs):
#     if seed is not None:
#         np.random.seed(seed)
#     if isinstance(data, BaseDataset):
#         return data.split(test_size=test_size, **kwargs)
#     return train_test_split(data, test_size=test_size, **kwargs)