import numpy as np
import pandas as pd

class IDWInterpolator:

    def __init__(self,x, z):
        self.x = x.values if isinstance(x, (pd.DataFrame, pd.Series)) else x
        self.z = z.values if isinstance(z, (pd.DataFrame, pd.Series)) else z
    
    def __call__(self, x):
        x = x.values if isinstance(x, (pd.DataFrame, pd.Series)) else x
        return self.idw(self.x[:,0],self.x[:,1], self.z, x[:,0], x[:,1])
    
    def idw(self, x, y, z, xi, yi):
        dist = self.distance_matrix(x,y, xi,yi)
        # In IDW, weights are 1 / distance
        weights = 1.0 / dist
        # Make weights sum to one
        weights /= weights.sum(axis=0)
        # Multiply the weights for each interpolated point by all observed Z-values
        zi = np.dot(weights.T, z)
        return zi

    @classmethod
    def distance_matrix(cls, x0, y0, x1, y1):
        obs = np.vstack((x0, y0)).T
        interp = np.vstack((x1, y1)).T
        d0 = np.subtract.outer(obs[:,0], interp[:,0])
        d1 = np.subtract.outer(obs[:,1], interp[:,1])
        return np.hypot(d0, d1)
