from __future__ import annotations

"""
Embedding Projector (Numba-free UMAP)
====================================
Dash app that lets you switch between PCA and UMAP (3-D) and shows nearest
neighbours.  **Numba JIT is disabled** up-front, so you avoid the annoying
threading crash.  For Iris/Wine-sized demos the pure-Python fallback is
fast enough.

Run::

    python embedding_dashboard.py --debug

Requirements
~~~~~~~~~~~~
• dash ≥2 • plotly • scikit-learn • umap-learn (optional, for UMAP)
"""

from dataclasses import dataclass, field
from typing import Callable, Mapping

import os
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# ---------------------------------------------------------------------------
#  Disable Numba before UMAP import
# ---------------------------------------------------------------------------

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")  # no JIT → no thread issues

import umap

# ---------------------------------------------------------------------------
#  Projection & distance registry
# ---------------------------------------------------------------------------

# ...existing code...

from tqdm import tqdm  # Add this import

if umap is not None:
    # --- UMAP variants ---------------------------------------------------

    def _umap_euclidean(x: np.ndarray) -> np.ndarray:
        for _ in tqdm(range(1), desc="UMAP (Euclidean)"):
            return umap.UMAP(n_components=2, random_state=0).fit_transform(x)

    def _umap_sphere(x: np.ndarray) -> np.ndarray:
        for _ in tqdm(range(1), desc="UMAP (Sphere)"):
            ll = umap.UMAP(
                n_components=2, output_metric="haversine", random_state=0
            ).fit_transform(x)
            lat, lon = ll.T  # radians
            xs = np.sin(lat) * np.cos(lon)
            ys = np.sin(lat) * np.sin(lon)
            zs = np.cos(lat)
            return np.column_stack([xs, ys, zs])

    def _umap_hyperbolic(x: np.ndarray) -> np.ndarray:
        for _ in tqdm(range(1), desc="UMAP (Hyperbolic)"):
            xy = umap.UMAP(
                n_components=2, output_metric="hyperboloid", random_state=0
            ).fit_transform(x)
            x_, y_ = xy.T
            z_ = np.sqrt(1 + np.sum(xy**2, axis=1))
            return np.column_stack([x_, y_, z_])

    PROJECTIONS = {
        "UMAP (Euclidean)": _umap_euclidean,
        # "UMAP (Sphere)": _umap_sphere,
        # "UMAP (Hyperbolic)": _umap_hyperbolic,
    }
# ...existing code...

# ---------------------------------------------------------------------------
#  UI helpers
# ---------------------------------------------------------------------------

def _config_panel(feat_names: list[str], distances: Mapping[str, str]) -> html.Div:
    return html.Div(
        [
            html.H4("Configuration"),
            html.Label("Projection"),
            dcc.Dropdown(
                id="proj",
                options=[{"label": k, "value": k} for k in PROJECTIONS],
                value=list(PROJECTIONS.keys())[0],  # Default to first UMAP projection
                clearable=False
            ),
            html.Br(),
            html.P(f"Features: {', '.join(feat_names)}"),
            html.P("Click a point for neighbours."),
        ],
        style={"width": 240, "padding": "1rem", "borderRight": "1px solid #e2e2e2"},
    )

def _centre_panel() -> html.Div:
    return html.Div(
        dcc.Graph(id="scatter", style={"height": "80vh"}, config={"displayModeBar": False}),
        style={"flex": 1, "padding": "1rem"},
    )


def _nbr_panel() -> html.Div:
    return html.Div(
        [html.H4("Nearest neighbours"), html.Ul(id="nbrs")],
        style={"width": 260, "padding": "1rem", "borderLeft": "1px solid #e2e2e2"},
    )


# ---------------------------------------------------------------------------
#  Dashboard class
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingDashboard:
    data: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    target_names: list[str] | None = None

    app: dash.Dash = field(init=False)
    distances: Mapping[str, str] = field(default_factory=lambda: {
        "Euclidean": "euclidean",
        "Cosine": "cosine",
    })

    def __post_init__(self):
        self.app = dash.Dash(__name__)
        self.app.layout = self._layout()
        self._callbacks()

    @classmethod
    def from_backend(cls, name: str = "iris") -> "EmbeddingDashboard":
        from sklearn.datasets import load_iris, load_wine

        ds = load_iris() if name == "iris" else load_wine()
        return cls(ds.data, ds.target, ds.feature_names, ds.target_names)

    # ---- layout -----------------------------------------------------------
    def _layout(self) -> html.Div:
        return html.Div(
            [
                html.Div(html.H2("Embedding Projector"), style={"padding": "0.5rem 1rem"}),
                dcc.Store(id="emb"),
                dcc.Store(id="metric_store", data="euclidean"),
                html.Div([_config_panel(self.feature_names, self.distances), _centre_panel(), _nbr_panel()], style={"display": "flex"}),
            ],
            style={"display": "flex", "flexDirection": "column", "height": "100vh"},
        )

    # ---- callbacks --------------------------------------------------------
    def _callbacks(self):
        @self.app.callback(Output("emb", "data"), Input("proj", "value"))
        def _compute(method):  # noqa: ANN001
            return PROJECTIONS[method](self.data).tolist()

        @self.app.callback(Output("scatter", "figure"), Input("emb", "data"), Input("scatter", "clickData"))
        def _scatter(edata, click):  # noqa: ANN001
            emb = np.asarray(edata)
            idx = _clicked(click)
            return self._fig(emb, idx)

        @self.app.callback(
            Output("nbrs", "children"), Input("emb", "data"), Input("metric_store", "data"), Input("scatter", "clickData")
        )
        def _nbrs(edata, metric, click):  # noqa: ANN001
            emb = np.asarray(edata)
            idx = _clicked(click)
            return self._nbr_list(emb, metric, idx)

    # ---- drawing helpers --------------------------------------------------
    def _fig(self, emb: np.ndarray, idx: int | None):
        labels_txt = [
            f"{i}: {self.target_names[self.labels[i]]}" if self.target_names is not None else f"{i}: {self.labels[i]}"
            for i in range(len(self.labels))
        ]
        base = go.Scatter3d(
            x=emb[:, 0],
            y=emb[:, 1],
            z=emb[:, 2],
            mode="markers",
            text=labels_txt,
            hoverinfo="text",
            marker=dict(size=6, opacity=0.8, color=self.labels, colorscale="Viridis"),
        )
        data = [base]
        if idx is not None:
            data.append(
                go.Scatter3d(x=[emb[idx, 0]], y=[emb[idx, 1]], z=[emb[idx, 2]], mode="markers", marker=dict(size=10))
            )
        return go.Figure(data=data).update_layout(margin=dict(l=0, r=0, b=0, t=0))

    def _nbr_list(self, emb: np.ndarray, metric: str, idx: int | None):
        if idx is None:
            return [html.Li("Select a point.")]
        d = pairwise_distances([emb[idx]], emb, metric=metric).ravel()
        nbrs = np.argsort(d)[1:11]
        name = (
            lambda i: self.target_names[self.labels[i]] if self.target_names is not None else str(self.labels[i])
        )
        return [html.Li(f"{i}: {name(i)} — d={d[i]:.3f}") for i in nbrs]

    # ---- run --------------------------------------------------------------
    def run(self, **kw):
        self.app.run(**kw)


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _clicked(click):  # noqa: ANN001
    try:
        pt = click["points"][0]
        if pt.get("curveNumber", 0) != 0:
            return None
        return int(pt.get("pointIndex", pt["pointNumber"]))
    except (TypeError, KeyError, IndexError):
        return None


# ---------------------------------------------------------------------------
#  Quick test stub
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["iris", "wine", "digits"], default="iris")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    EmbeddingDashboard.from_backend(args.dataset).run(debug=args.debug)
