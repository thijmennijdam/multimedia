from __future__ import annotations
"""
Embedding Projector (flat script, no classes)
===========================================
Dash app that switches between PCA and 3‑D UMAP and shows nearest neighbours.
Numba JIT is **not** disabled up‑front – if you still hit threading issues,
export `NUMBA_DISABLE_JIT=1` before running.

Run::

    python embedding_projector_flat.py --debug

Requirements
------------
• dash ≥2 • plotly • scikit‑learn • umap‑learn (optional, for UMAP)
"""

import warnings
from pathlib import Path
import argparse

import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris, load_wine, load_digits

# ---------------------------------------------------------------------------
#  Optional UMAP import (falls back to PCA only if missing)
# ---------------------------------------------------------------------------
import umap  # type: ignore

warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
)

# ---------------------------------------------------------------------------
#  Projection functions (3‑D)
# ---------------------------------------------------------------------------

def _umap_euclidean(x: np.ndarray) -> np.ndarray:  # noqa: ANN001
    return (
        umap.UMAP(
            n_components=3,
            n_neighbors=10,
            min_dist=0.3,
            n_epochs=100,
            n_jobs=1,  # force single‑thread to avoid warnings
            random_state=0,
        )
        .fit_transform(x)
        .astype(np.float32)
    )


PROJECTIONS: dict[str, callable[[np.ndarray], np.ndarray]] = {}
PROJECTIONS["UMAP (Euclidean)"] = _umap_euclidean

# ---------------------------------------------------------------------------
#  UI helper components
# ---------------------------------------------------------------------------

def _config_panel(feat_names: list[str]) -> html.Div:
    return html.Div(
        [
            html.H4("Configuration"),
            html.Label("Projection"),
            dcc.Dropdown(
                id="proj",
                options=[{"label": k, "value": k} for k in PROJECTIONS],
                value=next(iter(PROJECTIONS)),  # first key
                clearable=False,
            ),
            html.Br(),
            html.P(f"Features: {', '.join(feat_names)}"),
            html.P("Click a point for neighbours."),
        ],
        style={"width": 240, "padding": "1rem", "borderRight": "1px solid #e2e2e2"},
    )


def _centre_panel() -> html.Div:
    return html.Div(
        dcc.Loading(
            id="loading-scatter",
            type="circle",
            children=dcc.Graph(
                id="scatter",
                style={"height": "80vh"},
                config={"displayModeBar": False},
            ),
        ),
        style={"flex": 1, "padding": "1rem"},
    )


def _nbr_panel() -> html.Div:
    return html.Div(
        [html.H4("Nearest neighbours"), html.Ul(id="nbrs")],
        style={"width": 260, "padding": "1rem", "borderLeft": "1px solid #e2e2e2"},
    )

# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

def _clicked(click):  # noqa: ANN001
    """Return point index from `clickData` or *None* if background/none."""
    try:
        pt = click["points"][0]
        if pt.get("curveNumber", 0) != 0:  # ignore neighbour overlay
            return None
        return int(pt.get("pointIndex", pt["pointNumber"]))
    except (TypeError, KeyError, IndexError):
        return None


# ---------------------------------------------------------------------------
#  Parse CLI *before* building the app so globals are ready for callbacks
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["iris", "wine", "digits"], default="iris")
parser.add_argument("--debug", action="store_true")
ARGS = parser.parse_args()


def _load_dataset(name: str):
    if name == "iris":
        ds = load_iris()
    elif name == "wine":
        ds = load_wine()
    elif name == "digits":
        ds = load_digits()
    else:  # pragma: no cover – CLI ensures validity
        raise ValueError(name)
    # Some datasets (digits) lack `feature_names`/`target_names` attributes
    feat_names = getattr(ds, "feature_names", [f"f{i}" for i in range(ds.data.shape[1])])
    targ_names = getattr(ds, "target_names", None)
    return ds.data.astype(np.float32), ds.target.astype(int), list(feat_names), targ_names


DATA, LABELS, FEATURE_NAMES, TARGET_NAMES = _load_dataset(ARGS.dataset)

# Distance metrics (keys exposed to user, values passed to sklearn)
DISTANCES: dict[str, str] = {"Euclidean": "euclidean", "Cosine": "cosine"}

# ---------------------------------------------------------------------------
#  Build Dash app
# ---------------------------------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Div(html.H2("Embedding Projector"), style={"padding": "0.5rem 1rem"}),
        dcc.Store(id="emb"),
        dcc.Store(id="metric_store", data="euclidean"),
        html.Div(
            [_config_panel(FEATURE_NAMES), _centre_panel(), _nbr_panel()],
            style={"display": "flex"},
        ),
    ],
    style={"display": "flex", "flexDirection": "column", "height": "100vh"},
)

# ---------------------------------------------------------------------------
#  Callbacks
# ---------------------------------------------------------------------------


@app.callback(Output("emb", "data"), Input("proj", "value"))
def _compute(method):  # noqa: ANN001
    """Compute embedding and cache in dcc.Store (as list, not ndarray)."""
    emb = PROJECTIONS[method](DATA)
    return emb.tolist()


@app.callback(
    Output("scatter", "figure"),
    Input("emb", "data"),
    Input("scatter", "clickData"),
)
def _scatter(edata, click):  # noqa: ANN001
    if edata is None:
        return go.Figure()  # empty until embedding computed
    emb = np.asarray(edata, dtype=np.float32)
    idx = _clicked(click)
    return _fig(emb, idx)


@app.callback(
    Output("nbrs", "children"),
    Input("emb", "data"),
    Input("metric_store", "data"),
    Input("scatter", "clickData"),
)
def _nbrs(edata, metric, click):  # noqa: ANN001
    if edata is None:
        return [html.Li("Loading...")]
    emb = np.asarray(edata, dtype=np.float32)
    idx = _clicked(click)
    return _nbr_list(emb, metric, idx)

# ---------------------------------------------------------------------------
#  Drawing helpers (no state mutation)
# ---------------------------------------------------------------------------

def _fig(emb: np.ndarray, idx: int | None):
    labels_txt = [
        f"{i}: {TARGET_NAMES[LABELS[i]] if TARGET_NAMES is not None else LABELS[i]}"
        for i in range(len(LABELS))
    ]

    base = go.Scatter3d(
        x=emb[:, 0],
        y=emb[:, 1],
        z=emb[:, 2],
        mode="markers",
        text=labels_txt,
        hoverinfo="text",
        marker=dict(size=6, opacity=0.8, color=LABELS, colorscale="Viridis"),
    )

    traces: list[go.Scatter3d] = [base]

    if idx is not None:
        traces.append(
            go.Scatter3d(
                x=[emb[idx, 0]],
                y=[emb[idx, 1]],
                z=[emb[idx, 2]],
                mode="markers",
                marker=dict(size=10),
            )
        )

    return go.Figure(data=traces).update_layout(margin=dict(l=0, r=0, b=0, t=0))


def _nbr_list(emb: np.ndarray, metric: str, idx: int | None):
    if idx is None:
        return [html.Li("Select a point.")]

    dists = pairwise_distances([emb[idx]], emb, metric=metric).ravel()
    neighbours = np.argsort(dists)[1:11]

    def _name(i: int) -> str:  # local helper
        return (
            TARGET_NAMES[LABELS[i]] if TARGET_NAMES is not None else str(LABELS[i])
        )

    return [html.Li(f"{i}: {_name(i)} — d={dists[i]:.3f}") for i in neighbours]


# ---------------------------------------------------------------------------
#  Run – placed at bottom so `app` is defined when module imported
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=ARGS.debug)
