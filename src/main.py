from __future__ import annotations
"""Embedding projector with pair‑wise comparison.

* Two **projection** options (Euclidean / Hyperbolic).
* Two **view** modes toggled in the centre panel:
    • **3‑D scatter** (default)
    • **Poincaré disk** – a top‑down 2‑D view; for hyperbolic embeddings the
      hyperboloid → disk mapping is applied, for Euclidean we simply drop the
      *z* coordinate.

The file retains the *layout first, run last* structure.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from sklearn.datasets import load_digits, load_iris, load_wine
from PIL import Image


import umap

warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
)

# ---------------------------------------------------------------------------
#  Projection helpers
# ---------------------------------------------------------------------------


def _umap_euclidean(x: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """3‑D Euclidean UMAP embedding."""
    return (
        umap.UMAP(
            n_components=3,
            n_jobs=1,
            random_state=42,
        )
        .fit_transform(x)
        .astype(np.float32)
    )


def _hyperboloid_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xh, yh, zh) coordinates on the hyperboloid ℍ²."""
    emb2 = (
        umap.UMAP(
            n_components=2,
            n_jobs=1,
            random_state=42,
            output_metric="hyperboloid",
        )
        .fit(x)
    )
    xh, yh = emb2.embedding_[:, 0], emb2.embedding_[:, 1]
    zh = np.sqrt(1.0 + np.sum(emb2.embedding_**2, axis=1), dtype=np.float32)
    return xh, yh, zh


def _umap_hyperbolic(x: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """3‑D hyperbolic (hyperboloid model) embedding."""
    xh, yh, zh = _hyperboloid_2d(x)
    return np.column_stack((xh, yh, zh))


PROJECTIONS: dict[str, callable[[np.ndarray], np.ndarray]] = {
    "UMAP (Euclidean)": _umap_euclidean,
    "UMAP (Hyperbolic)": _umap_hyperbolic,
}

# ---------------------------------------------------------------------------
#  Dataset loading (global so callbacks can access)
# ---------------------------------------------------------------------------


def _load_dataset(name: str):
    if name == "iris":
        ds = load_iris()
    elif name == "wine":
        ds = load_wine()
    elif name == "digits":
        ds = load_digits()
    else:
        raise ValueError(name)
    
    print(f"Loaded dataset: {name} with {ds.data.shape[0]} samples and {ds.data.shape[1]} features.")

    feat_names = getattr(ds, "feature_names", [f"f{i}" for i in range(ds.data.shape[1])])
    targ_names = getattr(ds, "target_names", None)
    return ds.data.astype(np.float32), ds.target.astype(int), list(feat_names), targ_names


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["iris", "wine", "digits"], default="digits")
parser.add_argument("--debug", action="store_true")
ARGS = parser.parse_args()

DATA, LABELS, FEATURE_NAMES, TARGET_NAMES = _load_dataset(ARGS.dataset)

# ---------------------------------------------------------------------------
#  UI helper components (pure functions → no side‑effects)
# ---------------------------------------------------------------------------


def _config_panel() -> html.Div:
    return html.Div(
        [
            html.H4("Configuration"),
            html.Label("Projection"),
            dcc.Dropdown(
                id="proj",
                options=[{"label": k, "value": k} for k in PROJECTIONS],
                value=next(iter(PROJECTIONS)),
                clearable=False,
            ),
            html.Br(),
            html.P("Select two points for comparison."),
        ],
        style={"width": 240, "padding": "1rem", "borderRight": "1px solid #e2e2e2"},
    )


def _centre_panel() -> html.Div:
    view_toggle = dcc.RadioItems(
        id="view",
        options=[
            {"label": "3‑D", "value": "3d"},
            {"label": "Poincaré disk", "value": "disk"},
        ],
        value="3d",
        inline=True,
        style={"marginBottom": "0.5rem"},
    )

    graph = dcc.Graph(
        id="scatter",
        style={"height": "78vh"},
        config={"displayModeBar": False},
    )

    return html.Div(
        dcc.Loading(id="loading-scatter", type="circle", children=[view_toggle, graph]),
        style={"flex": 1, "padding": "1rem"},
    )


def _cmp_panel() -> html.Div:
    return html.Div(
        [html.H4("Point comparison"), html.Div(id="cmp")],
        style={"width": 320, "padding": "1rem", "borderLeft": "1px solid #e2e2e2"},
    )


# ---------------------------------------------------------------------------
#  Layout builder (readers see this first)
# ---------------------------------------------------------------------------


def make_layout() -> html.Div:
    return html.Div(
        [
            html.Div(html.H2("Embedding Projector"), style={"padding": "0.5rem 1rem"}),
            dcc.Store(id="emb"),
            dcc.Store(id="sel", data=[]),
            html.Div(
                [_config_panel(), _centre_panel(), _cmp_panel()],
                style={"display": "flex"},
            ),
        ],
        style={"display": "flex", "flexDirection": "column", "height": "100vh"},
    )


# ---------------------------------------------------------------------------
#  Utility helpers
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
#  Drawing helpers
# ---------------------------------------------------------------------------


def _fig3d(emb: np.ndarray, sel: list[int]) -> go.Figure:
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
    traces = [base]
    if sel:
        traces.append(
            go.Scatter3d(
                x=emb[sel, 0],
                y=emb[sel, 1],
                z=emb[sel, 2],
                mode="markers",
                marker=dict(size=10, color="red"),
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), uirevision="embedding")
    return fig


def _fig_disk(x: np.ndarray, y: np.ndarray, sel: list[int]) -> go.Figure:
    base = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        text=[
            f"{i}: {TARGET_NAMES[LABELS[i]] if TARGET_NAMES is not None else LABELS[i]}"
            for i in range(len(LABELS))
        ],
        hoverinfo="text",
        marker=dict(size=6, opacity=0.8, color=LABELS, colorscale="Viridis"),
    )
    traces = [base]
    if sel:
        traces.append(
            go.Scatter(
                x=x[sel],
                y=y[sel],
                mode="markers",
                marker=dict(size=10, color="red"),
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision="embedding",
    )
    return fig


# ---------------------------------------------------------------------------
#  Callback registration
# ---------------------------------------------------------------------------


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(Output("emb", "data"), Input("proj", "value"))
    def _compute(method):
        return PROJECTIONS[method](DATA).tolist()

    @app.callback(
        Output("sel", "data"),
        Input("scatter", "clickData"),
        State("sel", "data"),
        prevent_initial_call=True,
    )
    def _select(click, sel):
        idx = _clicked(click)
        if idx is None:
            return sel
        sel = sel or []
        if idx in sel:
            sel.remove(idx)
        else:
            sel.append(idx)
            if len(sel) > 2:
                sel = sel[-2:]
        return sel

    @app.callback(
        Output("scatter", "figure"),
        Input("emb", "data"),
        Input("sel", "data"),
        Input("view", "value"),
        Input("proj", "value"),
    )
    def _scatter(edata, sel, view, proj):
        if edata is None:
            return go.Figure()
        emb = np.asarray(edata, dtype=np.float32)
        if view == "3d":
            return _fig3d(emb, sel or [])
        if "Hyperbolic" in proj:
            xh, yh, zh = emb[:, 0], emb[:, 1], emb[:, 2]
            dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
        else:
            dx, dy = emb[:, 0], emb[:, 1]
        return _fig_disk(dx, dy, sel or [])

    @app.callback(Output("cmp", "children"), Input("sel", "data"))
    def _compare(sel):
        if sel is None or len(sel) < 2:
            return html.P("Select two distinct points.")
        i, j = sel[:2]
       
        li = TARGET_NAMES[LABELS[i]] if TARGET_NAMES is not None else LABELS[i]
        lj = TARGET_NAMES[LABELS[j]] if TARGET_NAMES is not None else LABELS[j]
        return [
            html.P(f"Point {i} label: {li}"),
            html.P(f"Point {j} label: {lj}"),
        ]


# ---------------------------------------------------------------------------
#  App construction
# ---------------------------------------------------------------------------


def build_app(debug: bool = False) -> dash.Dash:
    app = dash.Dash(__name__)
    app.layout = make_layout()
    register_callbacks(app)
    return app


if __name__ == "__main__":
    build_app(debug=ARGS.debug).run(debug=ARGS.debug)