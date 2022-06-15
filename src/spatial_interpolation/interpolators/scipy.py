from typing import Any
import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd


class ScipyInterpolator(BaseEstimator):
    def __init__(self,interpolator, dimensions=None, dim_cols=None, **kwargs):
        self.interpolator = interpolator
        self.interpolator_args = kwargs
        self.f = None
        self.dimensions = dimensions
        self.dim_cols = dim_cols or np.array(dimensions).ravel().tolist()
        self._fitted = False
    
    @property
    def fitted(self):
        return self._fitted
    
    def fit(self,X,y):
        if isinstance(y,str):
            y = X[y].values
        if isinstance(X,pd.DataFrame):
            if self.dimensions:
                dims = [X[dim] for dim in self.dimensions]
            else:
                dims = [X.values]
        else:
            dims = [X]
        self.f = self.interpolator(*dims,y,**self.interpolator_args)
        self._fitted = True
        return self
    
    def predict(self,*args: Any, **kwds: Any) -> Any:
        if len(args)==1:
            X = args[0]
        else:
            if self.dim_cols:
                X = pd.DataFrame(dict(zip(self.dim_cols,[x.ravel() for x in args])))
            else:
                X = args
        if isinstance(X,pd.DataFrame):
            if self.dimensions:
                dims = [X[dim] for dim in self.dimensions]
            else:
                dims = [X.values]
        else:
            dims = [X]
        return self.f(*dims)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if not self.fitted:
            raise ValueError("Model not fitted")
        return self.predict(*args, **kwds)