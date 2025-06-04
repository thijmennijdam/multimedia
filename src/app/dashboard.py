# dashboard_fixed.py
from __future__ import annotations

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
#  UI helpers                                                                 #
# --------------------------------------------------------------------------- #
def left_panel(dataset: str, reduction: str) -> html.Div:
    return html.Div(
        [
            html.H4("Configuration"),
            html.P(f"Dataset: {dataset}"),
            html.P(f"Reduction: {reduction} (3 D)"),
            html.P("Click a point to see neighbours."),
        ],
        style={"width": 220, "padding": "1rem", "borderRight": "1px solid #e2e2e2"},
    )


def centre_panel() -> html.Div:
    return html.Div(
        dcc.Graph(id="scatter-3d", style={"height": "80vh"}, config={"displayModeBar": False}),
        style={"flex": 1, "padding": "1rem"},
    )


def right_panel() -> html.Div:
    return html.Div(
        [html.H4("Nearest neighbours"), html.Ul(id="neighbor-list")],
        style={"width": 260, "padding": "1rem", "borderLeft": "1px solid #e2e2e2"},
    )


# --------------------------------------------------------------------------- #
#  Utility                                                                    #
# --------------------------------------------------------------------------- #
def _clicked_idx(click: dict | None) -> int | None:
    """
    Return index of the clicked point or None.

    Plotly:
      * go.* traces   → key 'pointNumber'
      * px.* traces   → key 'pointIndex'
    Ignore clicks on the highlight trace (curveNumber > 0).
    """
    try:
        pt = click["points"][0]
        if pt.get("curveNumber", 0) != 0:
            return None
        return int(pt.get("pointIndex", pt["pointNumber"]))
    except (TypeError, KeyError, IndexError):
        return None


# --------------------------------------------------------------------------- #
#  Dashboard                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingDashboard:
    data: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    target_names: list[str] | None = None
    reduction: str = "PCA"

    app: dash.Dash = field(init=False)
    points_3d: np.ndarray = field(init=False)

    # ---- construction ----------------------------------------------------- #
    def __post_init__(self):
        self.points_3d = self._reduce()
        self.app = dash.Dash(__name__)
        self.app.layout = self._layout()
        self._callbacks()

    @classmethod
    def from_backend(cls, dataset_name: str = "iris", reduction: str = "PCA") -> "EmbeddingDashboard":
        from src.backend.data import Datasets

        ds = Datasets()
        data, labels, feat = ds.get_dataset(dataset_name)

        target = None
        if dataset_name == "iris":
            from sklearn.datasets import load_iris
            target = load_iris().target_names
        elif dataset_name == "wine":
            from sklearn.datasets import load_wine
            target = load_wine().target_names

        return cls(data, labels, feat, target, reduction)

    # ---- helpers ---------------------------------------------------------- #
    def _reduce(self) -> np.ndarray:
        if self.reduction.upper() == "PCA":
            return PCA(n_components=3).fit_transform(self.data)
        raise ValueError(f"unknown reduction: {self.reduction}")

    def _layout(self) -> html.Div:
        return html.Div(
            [
                html.Div(html.H2("Embedding Projector"), style={"padding": "0.5rem 1rem"}),
                html.Div(
                    [left_panel(",".join(self.feature_names), self.reduction), centre_panel(), right_panel()],
                    style={"display": "flex", "flex": 1},
                ),
            ],
            style={"display": "flex", "flexDirection": "column", "height": "100vh"},
        )

    # ---- callbacks -------------------------------------------------------- #
    def _callbacks(self):
        @self.app.callback(Output("scatter-3d", "figure"), Input("scatter-3d", "clickData"))
        def _scatter(click):
            return self._build_scatter(_clicked_idx(click))

        @self.app.callback(Output("neighbor-list", "children"), Input("scatter-3d", "clickData"))
        def _neigh(click):
            return self._build_neighbour_list(_clicked_idx(click))

    # ---- pure logic ------------------------------------------------------- #
    def _build_scatter(self, idx: int | None) -> go.Figure:
        label_text = [
            f"{i}: {self.target_names[self.labels[i]]}" if self.target_names is not None else f"{i}: {self.labels[i]}"
            for i in range(len(self.labels))
        ]

        base = go.Scatter3d(
            x=self.points_3d[:, 0],
            y=self.points_3d[:, 1],
            z=self.points_3d[:, 2],
            mode="markers",
            text=label_text,
            hoverinfo="text",
            marker=dict(size=6, opacity=0.8, color=self.labels, colorscale="Viridis"),
        )

        fig = go.Figure(data=[base])
        if idx is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[self.points_3d[idx, 0]],
                    y=[self.points_3d[idx, 1]],
                    z=[self.points_3d[idx, 2]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    hoverinfo="skip",
                )
            )

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        return fig


    def _build_neighbour_list(self, idx: int | None):
        if idx is None:
            return [html.Li("Select a point to see its neighbours.")]

        dists = euclidean_distances([self.points_3d[idx]], self.points_3d).ravel()
        nbrs = np.argsort(dists)[1:11]

        name = (
            lambda i: self.target_names[self.labels[i]] if self.target_names is not None else str(self.labels[i])
        )
        return [html.Li(f"{i}: {name(i)} — d={dists[i]:.3f}") for i in nbrs]

    # ---- public ----------------------------------------------------------- #
    def run(self, **kw):
        self.app.run(**kw)


# --------------------------------------------------------------------------- #
#  Quick test                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    dash = EmbeddingDashboard.from_backend("iris")
    dash.run(debug=True)
