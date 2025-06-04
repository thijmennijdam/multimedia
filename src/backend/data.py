import numpy as np
import numba
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import umap.umap_ as umap

class Datasets:
    def __init__(self):
        self.datasets = {
            'iris': sklearn.datasets.load_iris(),
            'wine': sklearn.datasets.load_wine(),
            'digits': sklearn.datasets.load_digits(),
            'breast_cancer': sklearn.datasets.load_breast_cancer(),
            'california_housing': sklearn.datasets.fetch_california_housing(),
        }

    def get_dataset(self, name):
        ds = self.datasets.get(name, None)
        if ds is None:
            return None, None, None
        if hasattr(ds, 'data') and hasattr(ds, 'target'):
            return ds.data, ds.target, ds.feature_names if hasattr(ds, 'feature_names') else None
        return ds, None, None
    def list_datasets(self):
        return list(self.datasets.keys())
    
    def get_umap_embedding(self, data, n_components=3, random_state=42):
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(data)
    
    
