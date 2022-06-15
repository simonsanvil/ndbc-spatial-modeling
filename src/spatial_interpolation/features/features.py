"""
Data preprocessing functions that can be applied before or after the features are extracted.
These functions can be used to test different combinations of features before fitting them to a model
for experimentation.
"""
from functools import partial
import re
import numpy as np
import pandas as pd
import geopandas as gpd

from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
from spatial_interpolation import utils

__all__ = [
    "assign_from_columns",
    "apply_functions_to_df",
    "apply_operation_to_column",
    "add_time_shifts",
    "add_window_rollings",
]

def apply_functions_to_df(
    df: pd.DataFrame,
    functions: Union[Dict[str,Dict[str,Any]],List[Dict[str,Any]],List[Union[str,Callable]]],
    function_args: List[Union[Dict,Any]]=None,
    search_globals:bool=True,
) -> pd.DataFrame:
    """
    Apply a list or dictionary of functions to a dataframe.
    The functions given can be strings or callables.
    If strings, the functions are searched for in the `pandas.api.dataframe.DataFrame`
    and `pandas.api` namespaces. If not found and `search_globals=True`, the functions are searched for 
    in the global namespace .
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to apply the functions to.
    functions : Union[Dict[str,Dict[str,Any]],List[Union[str,Callable]]]
        A dictionary of functions with the keys being the actual 
        functions or the names of the functions to apply to the dataframe
        and the values being the arguments to pass to the function either 
        as a dictionary of keyword arguments or as a list of ordered arguments.
        If a list, the functions are applied in the order given.
        and the arguments must be passed on `function_args` in the same order.
    function_args : List[Union[Dict,Any]]
        The arguments to pass to the functions. 
        Only applies if `functions` is a list.
    
    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the functions applied.
    
    Examples
    --------
    >>> df = pd.DataFrame({'a':0,1],'b':[1,2]})
    >>> df
         a  b
    0  0  1
    1  1  2
    >>> add_n_to_df = lambda df,n : df + n
    >>> df = apply_functions_to_df(
                df,
                functions = {
                    'add_n_to_df':{'n':1},
                    'drop':{'columns':['a']}
                })
    >>> df
        b
    0   2
    1   3
    >>> df = apply_functions_to_df(df,'assign',{'a':0})
    >>> df
        a  b
    0  0  2
    1  0  3
    """
    if function_args is None and not isinstance(functions, (dict,list)):
        raise ValueError("Must specify function_args or functions must be a dictionary of function:arguments key-value pairs.")
    elif isinstance(functions, dict) and function_args is not None:
        raise ValueError("If functions is a dictionary of function arguments `function_args` must not be specified.")
    elif isinstance(functions,dict) and function_args is None:
        function_args = list(functions.values())
        functions = list(functions.keys())
    elif isinstance(functions,list) and function_args is None:
        function_args = [args for d in functions for args in list(d.values())]
        functions = [f for d in functions for f in list(d.keys())]
    elif not isinstance(functions, list):
        functions = [functions]
    if function_args is None:
        function_args = [{}] * len(functions)        
    elif not isinstance(function_args, list):
        function_args = [function_args]
   
    for func,args in zip(functions,function_args):
        if isinstance(func, str):
            if func in dir(df):
                f = getattr(df, func)
            elif func in dir(pd) and callable(getattr(pd, func)):
                f = partial(getattr(pd, func), df)
            elif search_globals and func in globals() and callable(globals()[func]):
                f = partial(globals()[func], df)
            elif utils.get_object_from_str(func) is not None and callable(utils.get_object_from_str(func)):
                f = partial(utils.get_object_from_str(func),df)
            else:
                raise ValueError(f"Function \"{func}\" not found.")                
        elif callable(func):
            f = partial(func, df)
        else:
            raise ValueError(f"Function {func} must be a string or a callable.")
        if isinstance(args, dict):
            df = f(**args)
        elif isinstance(args, (list,tuple)):
            df = f(*args)
        else:
            df = f(args)
    return df


def assign_from_columns(
    df: pd.DataFrame,
    var_name: str,
    columns_a: str,
    columns_b: str,
    operation: Union[str,Callable],
) -> pd.DataFrame:
    """
    Assign a new variable from performing an operation on two columns.
    """
    df[var_name] = df[columns_a].apply(operation, args=(df[columns_b],))
    return df


def apply_operation_to_column(
    df,
    colname,
    operation,
    other_colnames: List[str],
    **kwargs,
) -> pd.Series:
    """
    Apply an operation to a column.
    """
    if isinstance(other_colnames, str):
        other_colnames = [other_colnames]
    if other_colnames:
        args = [df[c] for c in other_colnames]
    if isinstance(operation, str):
        result = df[colname].apply(operation, args=args, **kwargs)
    else:
        result = operation(df[colname], *args, **kwargs)
    
    return result


def add_time_shifts(
    df,
    shift_columns:Union[List[str],str,Dict[str,int]],
    shift_by:Union[int,List[int]]=None,
    group_by:Union[str,List[str]]=None,
) -> pd.DataFrame:
    """
    Add time shift columns to a dataframe.
    """
    if shift_by is None and not isinstance(shift_columns, dict):
        raise ValueError("Must specify shift_by or shift_columns as a dictionary of col:int pairs.")

    if isinstance(shift_columns, dict):
        shift_by = list(shift_columns.values())
        shift_columns = list(shift_columns.keys())
    elif isinstance(shift_columns, str):
        shift_columns = [shift_columns]
    if isinstance(shift_by, int):
        shift_by = [shift_by]*len(shift_columns)
    elif not isinstance(shift_by, (int,Iterable)):
        raise ValueError("shift_by must be an integer or iterable.")
    if shift_by and len(shift_by)==1:
        shift_by = shift_by*len(shift_columns)
    elif len(shift_by)!=len(shift_columns):
        raise ValueError("iterable shift_by must be of the same length as shift_columns.")
    if not isinstance(group_by, (list, tuple)):
        group_by = [group_by]*len(shift_columns)
    elif len(group_by) != len(shift_columns):
        raise ValueError("iterable group_by must be of the same length as shift_columns.")

    for col,shift,group in zip(shift_columns,shift_by,group_by):
        x = df[col] if group is None else df.groupby(group)[col]
        if isinstance(shift,list):
            for bb in shift:
                df[col+'_shift_'+str(bb)] = x.shift(bb)
        else:
            df[col+'_shifted_'+str(shift)] = x.shift(shift)
    
    return df

def add_window_rollings(
    df,
    rolling_columns:Union[List[str],str,Dict[str,int]],
    rolling_windows:Union[int,List[int]]=None,
    operation:Union[str,Callable]="mean",
    group_by:str=None,
) -> pd.DataFrame:
    """
    Add time rolling columns to a dataframe.
    """
    if rolling_windows is None and not isinstance(rolling_columns, dict):
        raise ValueError("Must specify rolling_windows or rolling_columns as a dictionary of col:int pairs.")

    if isinstance(rolling_columns, dict):
        rolling_windows = list(rolling_columns.values())
        rolling_columns = list(rolling_columns.keys())
    elif isinstance(rolling_columns, str):
        rolling_columns = [rolling_columns]
    if isinstance(rolling_windows, int):
        rolling_windows = [rolling_windows]*len(rolling_columns)
    elif not isinstance(rolling_windows, (int,Iterable)):
        raise ValueError("rolling_windows must be an integer or iterable.")
    if rolling_windows and len(rolling_windows)==1:
        rolling_windows = rolling_windows*len(rolling_columns)
    elif len(rolling_windows)!=len(rolling_columns):
        raise ValueError("iterable rolling_windows must be of the same length as rolling_columns.")
    if not isinstance(group_by, (list, tuple)):
        group_by = [group_by]*len(rolling_windows)
    elif len(group_by) != len(rolling_columns):
        raise ValueError("iterable group_by must be of the same length as rolling_columns.")
    
    for col,w,group in zip(rolling_columns,rolling_windows,group_by):
        x = df[col] if group is None else df.groupby(group)[col]
        if isinstance(w,list):
            for ww in w:
                rolled = x.rolling(window=ww).agg(operation)
                if group:
                    rolled = rolled.reset_index(group,drop=True)
                df[col+'_rolling_'+str(ww)] = rolled
        else:
            rolled = x.rolling(window=w).agg(operation)
            if group:
                rolled = rolled.reset_index(group,drop=True)
            df[col+'_rolling_'+str(w)] = rolled
    
    return df