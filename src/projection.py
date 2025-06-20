import numpy as np
import umap

# ... existing code for projection helpers ... 

def _hyperboloid_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xh, yh, zh) coordinates on the hyperboloid ℍ²."""
    emb2 = umap.UMAP(
        n_components=2,
        n_jobs=1,
        random_state=42,
        output_metric="hyperboloid",
    ).fit(x)
    xh, yh = emb2.embedding_[:, 0], emb2.embedding_[:, 1]
    zh = np.sqrt(1.0 + np.sum(emb2.embedding_**2, axis=1), dtype=np.float32)
    return xh, yh, zh


def _umap_hyperbolic(x: np.ndarray) -> np.ndarray:
    """3D hyperbolic (hyperboloid model) embedding."""
    xh, yh, zh = _hyperboloid_2d(x)
    return np.column_stack((xh, yh, zh))


PROJECTIONS: dict[str, callable] = {
    "UMAP": _umap_hyperbolic,
}


def _interpolate_hyperbolic(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    """Hyperbolic interpolation between two points on the hyperboloid."""
    inner_product = p1[2] * p2[2] - p1[0] * p2[0] - p1[1] * p2[1]
    inner_product = max(inner_product, 1.0 + 1e-10)
    d = np.arccosh(inner_product)

    if d < 1e-10:
        return p1.copy()

    sinh_d = np.sinh(d)
    coeff1 = np.sinh((1 - t) * d) / sinh_d
    coeff2 = np.sinh(t * d) / sinh_d
    return coeff1 * p1 + coeff2 * p2 