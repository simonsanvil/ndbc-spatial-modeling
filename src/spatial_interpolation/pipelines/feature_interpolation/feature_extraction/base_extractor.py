from typing import Dict, Any, List, Tuple
import logging

from sklearn.base import TransformerMixin

class BaseFeatureExtractor(TransformerMixin):

    Dataloader = None

    def __init__(
        self,
        dataloader=None,
        preprocess_params:Dict[str,Any]=None,
        make_features_params:Dict[str,Any]=None,
        postprocess_params:Dict[str,Any]=None,
        verbose:bool=True,
        ):
        self.dataloader = dataloader if dataloader is not None else self.Dataloader()
        self.preprocess_params = preprocess_params or {}
        self.make_features_params = make_features_params or {}
        self.postprocess_params = postprocess_params or {}
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
        self.verbose = verbose

    @property
    def preprocessed(self):
        if self._preprocessed_data is None:
            return self.preprocess(self.dataloader.load())
        return self._preprocessed_data
    
    def fit(self,*args,**kwargs):
        self.dataloader.fit(*args,**kwargs)
        return self
    
    def update_params(self,**kwargs):
        params_dict = {}
        for k,v in kwargs.items():
            if k in self.__dict__:
                params_dict[k] = v
            elif k.startswith("preprocess__") or k.startswith("pre__"):
                params_dict["preprocess_params"][k.split("__")[1]] = v
            elif "features__" in k:
                params_dict["make_features_params"][k.split("__")[1]] = v
            elif k.startswith("postprocess__") or k.startswith("post__"):
                params_dict["postprocess_params"][k.split("__")[1]] = v
        self.__dict__.update(params_dict)
        return self
    
    def transform(self,*args,**kwargs):
        """
        Extract features from the  dataset based on the parameters passed to the constructor.
        """
        if kwargs:
            self.update_params(**kwargs)
        self.logger.info("Attempting to load data from  dataloader...")
        data = self.dataloader.load()
        self.logger.info("Data loaded. Preprocessing...")
        preprocessed_data = self.preprocess(data, **self.preprocess_params)
        self.logger.info("Preprocessing complete. Making features...")
        features = self.make_features(preprocessed_data, **self.make_features_params)
        self.logger.info("Features made. Postprocessing...")
        postprocessed_features = self.postprocess(features, **self.postprocess_params)
        self.logger.info("Postprocessing complete.")
        return postprocessed_features
    
    @classmethod
    def preprocess(cls, data, *args,**kwargs):
        """
        Preprocess data using the extractor.
        """
        if isinstance(data,cls):
            extractor = data
            return extractor.preprocess(*args,**kwargs)
        raise NotImplementedError("preprocess is an abstract method. Please use a subclass.")
    
    @classmethod
    def make_features(cls, data, *args,**kwargs):
        """
        Extract features from the preprocessed data.
        """
        if isinstance(data,cls):
            extractor = data
            return extractor.make_features(*args,**kwargs)
        raise NotImplementedError("make_features is an abstract method. Please use a subclass.")
    
    @classmethod
    def postprocess(cls, data, *args,**kwargs):
        """
        Postprocess extracted features.
        """
        if isinstance(data,cls):
            extractor = data
            return extractor.postprocess(*args,**kwargs)
        raise NotImplementedError("postprocess is an abstract method. Please use a subclass.")
    


        