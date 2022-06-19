from dataclasses import dataclass, field
import pykrige
from abc import ABC, abstractmethod

import skgstat as skg
import gstools as gs
from pykrige.rk import Krige
from skgstat import OrdinaryKriging

@dataclass
class SkGKrige:
    variogram:object = None
    krige_params:dict = None
    variogram_params:dict = None
    mode:str = "exact"

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
class gsKrige:
    model:gs.CovModel
    
    krige_params:dict = None

    def __repr__(self) -> str:
        params_str= ", ".join(f"{k}={v}" for k,v in self.krige_params.items())
        cls_name = self.__class__.__name__
        return f"{cls_name}({params_str})"


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