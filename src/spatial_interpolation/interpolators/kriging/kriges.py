from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
import pykrige
from abc import ABC, abstractmethod

import skgstat as skg
import gstools as gs
from pykrige.rk import Krige
from skgstat import OrdinaryKriging

@dataclass
class GsKrige:
    """
    Fit and do kriging interpolation using [gstools](https://gstools.readthedocs.io)
    """
    model:gs.CovModel
    dimensions:list
    krige_params:dict = field(default_factory=dict)
    model_params:dict = field(default_factory=dict)
    latlon:bool = False
    gs_models = {
        "gaussian": gs.Gaussian,
        "linear": gs.Linear,
        "exponential": gs.Exponential,
        "matern": gs.Matern,
        # "stable": gs.Stable,
        # "rational": gs.Rational,
        # "circular": gs.Circular,
        "spherical": gs.Spherical,
    }

    def __post_init__(self):
        model = self.model if not isinstance(self.model, str) else self.gs_models[self.model.lower()]
        self.krige_params = self.krige_params or {}
        self.model_params = self.model_params or {}
        self._model = model
        self.model = model(latlon=self.latlon, **self.model_params)
        self.dim_cols = np.array(self.dimensions).ravel().tolist()

    def __repr__(self) -> str:
        params_str= ", ".join(f"{k}={v}" for k,v in self.krige_params.items())
        cls_name = self.__class__.__name__
        return f"{cls_name}({params_str})"
    
    def fit(self,X,y):
        if isinstance(y,str):
            y = X[y].values
        if isinstance(X,pd.DataFrame):
            if self.dimensions:
                dims = [X[dim].values for dim in self.dimensions]
            else:
                dims = [X.values]
        else:
            dims = [X]
        bin_center, gamma = gs.vario_estimate(dims,y, latlon=self.latlon)
        _, _, r2 = self.model.fit_variogram(bin_center, gamma, return_r2=True)
        self.fitted_variogram_r2 = r2
        self.krige = gs.krige.Krige(model=self.model, cond_pos=dims, cond_val=y, **self.krige_params)
        return self
    
    def predict(self, *args: Any, with_error:bool=False, **kwargs: Any) -> Any:
        if len(args)==1:
            X = args[0]
        else:
            if self.dim_cols:
                X = pd.DataFrame(dict(zip(self.dim_cols,[x.ravel() for x in args])))
            else:
                X = args
            print(X.columns)
        if isinstance(X,pd.DataFrame):
            if self.dimensions:
                dims = [X[dim].values for dim in self.dimensions]
            else:
                dims = [X.values]
        else:
            dims = [X]
        kwargs["return_var"] = True
        pred, err = self.krige(dims, **kwargs)
        self.error = err
        if with_error:
            return pred, err
        return pred
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.predict(*args, **kwds)


@dataclass
class SkGKrige:
    variogram:object = None
    krige_params:dict = None
    variogram_params:dict = None
    mode:str = "exact"

    def __post_init__(self):
        self.variogram_params = self.variogram_params or {}
        self.variogram = self.variogram or self.variogram_params.get("variogram")
            
    def __repr__(self) -> str:
        params_str= ", ".join(f"{k}={v}" for k,v in self.krige_params.items())
        cls_name = self.__class__.__name__
        return f"{cls_name}({params_str})"

    def fit(self, x, y):
        krige_params = self.krige_params or {}
        if self.variogram:
            V = self.variogram
        else:
            V = skg.Variogram(x, y)
            print("Variogram fitted:",V.model.__name__)
        self.krige = OrdinaryKriging(V, mode=self.mode, **krige_params)
    
    def predict(self, x):
        pred, _ = self.krige.transform(x.values)
        return pred

@dataclass
class pyKrige:
    krige_params:dict = field(default_factory=dict)

    def __repr__(self) -> str:
        params_str= ", ".join(f"{k}={v}" for k,v in self.krige_params.items())
        cls_name = self.__class__.__name__
        return f"{cls_name}({params_str})"

    def fit(self, x, y):
        krige_params = self.krige_params or {}
        self.krige = Krige(**krige_params)
        self.krige.fit(x=x.values, y=y.values)
        return self
    
    def predict(self, x):
        pred, ss = self.krige.model.execute(
            "points", 
            x.values[:,0],x.values[:,1]
        )
        return pred