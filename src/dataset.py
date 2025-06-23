from sklearn.datasets import load_digits, load_iris, load_wine
import numpy as np

def _load_dataset(name: str):
    if name == "iris":
        ds = load_iris()
    elif name == "wine":
        ds = load_wine()
    elif name == "digits":
        ds = load_digits()
    else:
        raise ValueError(name)

    print(
        f"Loaded dataset: {name} with {ds.data.shape[0]} samples and {ds.data.shape[1]} features."
    )

    feat_names = getattr(
        ds, "feature_names", [f"f{i}" for i in range(ds.data.shape[1])]
    )
    targ_names = getattr(ds, "target_names", None)
    images = getattr(ds, "images", None)  # (n, 8, 8) for digits, else None
    return (
        ds.data.astype(np.float32),
        ds.target.astype(int),
        list(feat_names),
        targ_names,
        images,
    )