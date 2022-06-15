from typing import Tuple
import psutil, os
import contextlib, logging

import pandas as pd
import numpy as np
import joblib

def make_config_dict(clss=None,*,frozen=False):
    from ml_collections import config_dict
    import typing
    if clss is None:
        return lambda clss: make_config_dict(clss, frozen=frozen)
    elif not isinstance(clss, type):
        raise ValueError("clss argument must be a class")
    cls_fields = {k:getattr(clss,k) for k in dir(clss) if not k.startswith("_") and not callable(getattr(clss,k))}
    for name, annotation in typing.get_type_hints(clss).items():
        if name in cls_fields: continue
        cls_fields[name] = config_dict.placeholder(annotation)
    if frozen:
        return config_dict.FrozenConfigDict(cls_fields)
    return config_dict.ConfigDict(cls_fields)


def print_memory_usage() -> None:
    print(f'memory use: {get_memory_usage():.4f} GB')

def get_memory_usage() -> float:
    pid = os.getpid()
    python_process = psutil.Process(pid)
    memoryUse = python_process.memory_info()[0]/2.**30 
    return memoryUse

def get_object_from_str(s):
    """
    Get a object from its import name

    e.g `get_model_from_str("sklearn.ensemble.RandomForestRegressor")`
    """
    try:
        module_name, class_name = s.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as e:
        logging.warning(f'Could not retrieve "{s}": {e}')
        return None


def int_partition(n,s):
    a, i = np.repeat(np.ceil(s/n), n).astype(int), 0
    while a.sum() != s: i+=1; a[i%n] -= 1
    return a


def resample_multiindex(df, interval, time_level=-1, agg="mean", non_numeric_agg="first", with_apply=False):
    index_names = df.index.names
    assert time_level in index_names or isinstance(
        time_level, int
    ), f"time_level must be the name or position of one of the multiindex's level"
    target_index = time_level if time_level in index_names else index_names[time_level]
    groupers = [
        pd.Grouper(level=level)
        if level != target_index
        else pd.Grouper(level=level, freq=interval)
        for level in index_names
    ]
    if agg is None:
        return df.groupby(groupers)
    numeric_cols = df.select_dtypes('number').columns
    non_num_cols = df.columns.difference(numeric_cols)
    agg_dict = {**{x: agg for x in numeric_cols}, **{x: non_numeric_agg for x in non_num_cols}}
    if with_apply:
        new_df = df.groupby(groupers).apply(agg)
    else:
        new_df = df.groupby(groupers).agg(agg_dict)
    return new_df


def parse_start_end_times(start_end_times, time_format):
    if isinstance(start_end_times, str):
        start_end_times = [start_end_times]
    start_end_times = [
        pd.to_datetime(time, format=time_format) for time in start_end_times
    ]
    return start_end_times

def parse_time_str(timestamp: str, parse_deltas: bool = True) -> pd.Timestamp:
    """
    Parses a timestamp from a string to a pandas.Timestamp or pd.Timedelta.

    Parameters
    ----------
    timestamp : str
        The timestamp to be parsed. Can be either the string representation of a
        pandas.Timestamp or of a pandas.Timedelta.
    parse_deltas : bool
        If True and the timestamp contains a time delta, the time delta will be parsed and returned
        as a pd.Timedelta. If False, the time delta will be ignored.
    Returns
    -------
    pandas.Timestamp or pd.Timedelta
        The parsed timestamp or timedelta.
    """
    import re
    # Check if the timestamp is a timedelta
    if parse_deltas and re.match(r"^now[-|+][0-9]+[dhms]$", timestamp.lower()):
        delta = timestamp.split("now")[1]
        return pd.Timestamp("now") + pd.Timedelta(delta)
    if parse_deltas and re.match(r"^[+|-]?[0-9]+[dhms]$", timestamp.lower()):
        return pd.Timedelta(timestamp)
    return pd.Timestamp(timestamp.lower())

def parse_start_end(start: str, end: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Parses a start and end time from strings to timestamps

    Parameters
    ----------
    start : str
        The start time to be parsed. Can be either the string representation of a
        pandas.Timestamp or of a pandas.Timedelta.
    end : str
        The end time to be parsed. Can be either the string representation of a
        pandas.Timestamp or of a pandas.Timedelta.

    Returns
    -------
    tuple
        The parsed start and end time as a tuple of pandas.Timestamp.

    Usage
    -----
    ```python
    start, end = "2020-05-20T15:00:00", "2020-05-20T15:59:59"
    start, end = parse_start_end_times(start,end)
    >> start
    [Out] pd.Timestamp('2020-05-20T15:00:00')
    >> end
    [Out] pd.Timestamp('2020-05-20T15:59:59')

    start,end = "now-1h","now"
    start,end = parse_start_end_times(start,end)
    >> start
    [Out] pd.Timestamp('2022-02-08T15:00:00')
    >> end
    [Out] pd.Timestamp('2020-02-08T15:59:59')
    ```

    """
    if isinstance(start, str):
        start = parse_time_str(start)
    if isinstance(end, str):
        end = parse_time_str(end)
    if isinstance(start, pd.Timedelta) and isinstance(end, pd.Timestamp):
        start = end + start if start < pd.Timedelta(0) else end - start
    elif isinstance(end, pd.Timedelta) and isinstance(start, pd.Timestamp):
        end = start + end
    elif isinstance(start, pd.Timedelta) and isinstance(start, pd.Timedelta):
        raise ValueError(
            "Only one of start and end can be a Timedelta. The other one (or both) must be a timestamp."
        )
    if start > end:
        raise AttributeError(f"start ({start}) cant be greater than end ({end})")
    return start, end


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()