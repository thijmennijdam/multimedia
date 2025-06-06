from __future__ import annotations
import argparse
import warnings
import base64
import io
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
    """3D Euclidean UMAP embedding."""
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
    """3D hyperbolic (hyperboloid model) embedding."""
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
    images = getattr(ds, "images", None)  # (n, 8, 8) for digits, else None
    return (
        ds.data.astype(np.float32),
        ds.target.astype(int),
        list(feat_names),
        targ_names,
        images,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["iris", "wine", "digits"], default="digits")
parser.add_argument("--debug", action="store_true")
ARGS = parser.parse_args()

DATA, LABELS, FEATURE_NAMES, TARGET_NAMES, IMAGES = _load_dataset(ARGS.dataset)

# ---------------------------------------------------------------------------
#  UI helper components (pure functions → no side-effects)
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
                style={
                    "backgroundColor": "white",
                    "border": "1px solid #ccc",
                    "borderRadius": "6px",
                    "color": "black",
                },
            ),

            html.Br(),
            html.P("Select two points for comparison."),
            
            html.Div([
                html.Button(
                    "Interpolate Points",
                    id="interpolate-btn",
                    disabled=True,
                    style={
                        "backgroundColor": "#007bff",
                        "color": "white",
                        "border": "none",
                        "padding": "0.5rem 1rem",
                        "borderRadius": "6px",
                        "cursor": "pointer",
                        "marginTop": "0.5rem",
                        "width": "100%"
                    }
                ),
                html.Div([
                    html.Label("Interpolation factor (t):"),
                    dcc.Slider(
                        id="interpolation-slider",
                        min=0,
                        max=1,
                        step=0.1,
                        value=0.5,
                        marks={i/10: str(i/10) for i in range(11)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={"marginTop": "1rem"})
            ])
        ],
        style={
                "width": 240,
                "padding": "1rem",
                "borderRight": "1px solid #ddd",
                "backgroundColor": "white",
                "borderRadius": "8px",
                "margin": "1rem"
            },
    )
    
def _empty_fig3d() -> go.Figure:
    fig = go.Figure(data=[go.Scatter3d(x=[], y=[], z=[], mode="markers")])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision="embedding",
        showlegend=False,
    )
    return fig

def _centre_panel() -> html.Div:
    view_toggle = dcc.RadioItems(
        id="view",
        options=[
            {"label": "3D", "value": "3d"},
            {"label": "Poincaré disk", "value": "disk"},
        ],
        value="3d",
        inline=True,
        style={
            "marginBottom": "0.5rem",
            "display": "flex",
            "gap": "1rem",
            "color": "rgb(33, 43, 181)",
            "fontWeight": "bold"
        },
    )

    return html.Div(
        dcc.Loading(
            id="loading-scatter",
            type="circle",
            children=[
                view_toggle,
                html.Div(
                    dcc.Graph(
                        id="scatter",
                        figure=_empty_fig3d(),
                        style={"height": "78vh"},
                        config={"displayModeBar": False},
                    ),
                    id="scatter-container"
                )
            ]
        ),
        style={
            "flex": 1,
            "padding": "1rem",
            "margin": "1rem",
            "backgroundColor": "white",
            "borderRadius": "8px",
        }
    )


def _cmp_panel() -> html.Div:
    return html.Div(
        [html.H4("Point comparison"), html.Div(id="cmp")],
        style={
    "width": 320,
    "padding": "1rem",
    "borderLeft": "1px solid #ddd",
    "backgroundColor": "white",
    "borderRadius": "8px",
    "margin": "1rem"
},
    )


# ---------------------------------------------------------------------------
#  Layout builder (readers see this first)
# ---------------------------------------------------------------------------


def make_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                html.H2("Embedding Projector", style={
                    "color": "white",
                    "margin": 0
                }),
                style={
                    "padding": "0.5rem 1rem",
                    "backgroundColor": "rgb(33, 43, 181)",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"
                }
            ),
            dcc.Store(id="emb"),
            dcc.Store(id="sel", data=[]),
            dcc.Store(id="interpolated-point"),
            html.Div(
                [_config_panel(), _centre_panel(), _cmp_panel()],
                style={"display": "flex"},
            ),
        ],
       style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
                "fontFamily": "Inter, sans-serif",
                "backgroundColor": "#f7f9fc"
            },
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


def _interpolate_points(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two points."""
    return (1 - t) * p1 + t * p2


def _interpolate_hyperbolic(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    """Hyperbolic interpolation between two points on the hyperboloid."""
    # p1 and p2 are points on the hyperboloid: (x, y, z) where z^2 - x^2 - y^2 = 1
    # For hyperbolic interpolation, we use the hyperbolic distance and geodesics
    
    # Compute the hyperbolic inner product: <p1, p2> = p1_z * p2_z - p1_x * p2_x - p1_y * p2_y
    inner_product = p1[2] * p2[2] - p1[0] * p2[0] - p1[1] * p2[1]
    
    # Clamp to avoid numerical issues with acosh
    inner_product = max(inner_product, 1.0 + 1e-10)
    
    # Hyperbolic distance
    d = np.arccosh(inner_product)
    
    # Handle the case where points are identical (d ≈ 0)
    if d < 1e-10:
        return p1.copy()
    
    # Geodesic interpolation on the hyperboloid
    # Formula: sinh((1-t)*d)/sinh(d) * p1 + sinh(t*d)/sinh(d) * p2
    sinh_d = np.sinh(d)
    coeff1 = np.sinh((1 - t) * d) / sinh_d
    coeff2 = np.sinh(t * d) / sinh_d
    
    interpolated = coeff1 * p1 + coeff2 * p2
    
    return interpolated


# ---------------------------------------------------------------------------
#  Drawing helpers
# ---------------------------------------------------------------------------


def _fig3d(emb: np.ndarray, sel: list[int], interpolated_point: np.ndarray = None) -> go.Figure:
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
        name="Data points"
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
                name="Selected points"
            )
        )
    
    if interpolated_point is not None:
        traces.append(
            go.Scatter3d(
                x=[interpolated_point[0]],
                y=[interpolated_point[1]],
                z=[interpolated_point[2]],
                mode="markers",
                marker=dict(size=12, color="orange", symbol="diamond"),
                name="Interpolated point",
                text=["Interpolated point"],
                hoverinfo="text"
            )
        )
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision="embedding",
        showlegend=False,
    )
    return fig


def _fig_disk(x: np.ndarray, y: np.ndarray, sel: list[int], interpolated_point: np.ndarray = None) -> go.Figure:
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
        name="Data points"
    )
    traces = [base]
    
    if sel:
        traces.append(
            go.Scatter(
                x=x[sel],
                y=y[sel],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Selected points"
            )
        )
    
    if interpolated_point is not None:
        traces.append(
            go.Scatter(
                x=[interpolated_point[0]],
                y=[interpolated_point[1]],
                mode="markers",
                marker=dict(size=12, color="orange", symbol="diamond"),
                name="Interpolated point",
                text=["Interpolated point"],
                hoverinfo="text"
            )
        )
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision="embedding",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
#  Callback registration
# ---------------------------------------------------------------------------


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("scatter-container", "children"),
        Input("emb", "data"),
        Input("sel", "data"),
        Input("view", "value"),
        Input("proj", "value"),
        Input("interpolated-point", "data"),
    )
    def _scatter(edata, sel, view, proj, interpolated_point):
        if edata is None:
            return None  # Don't render the graph at all
        emb = np.asarray(edata, dtype=np.float32)
        
        interp_point = None
        if interpolated_point is not None:
            interp_point = np.asarray(interpolated_point, dtype=np.float32)
        
        if view == "3d":
            fig = _fig3d(emb, sel or [], interp_point)
        elif "Hyperbolic" in proj:
            xh, yh, zh = emb[:, 0], emb[:, 1], emb[:, 2]
            dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
            
            # Transform interpolated point if it exists
            if interp_point is not None:
                interp_dx = interp_point[0] / (1.0 + interp_point[2])
                interp_dy = interp_point[1] / (1.0 + interp_point[2])
                interp_transformed = np.array([interp_dx, interp_dy])
            else:
                interp_transformed = None
                
            fig = _fig_disk(dx, dy, sel or [], interp_transformed)
        else:
            dx, dy = emb[:, 0], emb[:, 1]
            
            # Use only x,y coordinates for Euclidean case
            if interp_point is not None:
                interp_transformed = interp_point[:2]
            else:
                interp_transformed = None
                
            fig = _fig_disk(dx, dy, sel or [], interp_transformed)
        
        return dcc.Graph(
            id="scatter",
            figure=fig,
            style={"height": "78vh"},
            config={"displayModeBar": False},
        )
        
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
        Output("interpolate-btn", "disabled"),
        Input("sel", "data")
    )
    def _update_button_state(sel):
        return sel is None or len(sel) != 2

    @app.callback(
        Output("interpolated-point", "data"),
        Input("interpolate-btn", "n_clicks"),
        State("sel", "data"),
        State("interpolation-slider", "value"),
        State("emb", "data"),
        State("proj", "value"),
        prevent_initial_call=True
    )
    def _interpolate(n_clicks, sel, t, edata, proj):
        if n_clicks is None or sel is None or len(sel) != 2 or edata is None:
            return None
        
        emb = np.asarray(edata, dtype=np.float32)
        i, j = sel[:2]
        p1, p2 = emb[i], emb[j]
        
        # Choose interpolation method based on projection
        if "Hyperbolic" in proj:
            interpolated = _interpolate_hyperbolic(p1, p2, t)
        else:
            interpolated = _interpolate_points(p1, p2, t)
            
        return interpolated.tolist()

    @app.callback(
        Output("cmp", "children"), 
        Input("sel", "data"),
        Input("interpolated-point", "data"),
        Input("interpolation-slider", "value")
    )
    def _compare(sel, interpolated_point, t_value):
        if sel is None or len(sel) < 2:
            return html.P("Select two distinct points.")
        i, j = sel[:2]
        li = TARGET_NAMES[LABELS[i]] if TARGET_NAMES is not None else LABELS[i]
        lj = TARGET_NAMES[LABELS[j]] if TARGET_NAMES is not None else LABELS[j]

        def _img_tag(idx: int) -> html.Img | html.Span:
            if IMAGES is None:
                return html.Span()
            pil = Image.fromarray((IMAGES[idx] * 16).astype("uint8"), mode="L").resize((64, 64), Image.NEAREST)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            return html.Img(src=uri, style={"marginRight": "0.5rem", "border": "1px solid #bbb"})

        def _interpolated_img_tag() -> html.Img | html.Span:
            if IMAGES is None:
                return html.Span()
            # Interpolate the images in the original data space
            img1 = IMAGES[i].astype(np.float32)
            img2 = IMAGES[j].astype(np.float32)
            interpolated_img = (1 - t_value) * img1 + t_value * img2
            
            # Convert to uint8 and create PIL image
            pil = Image.fromarray((interpolated_img * 16).astype("uint8"), mode="L").resize((64, 64), Image.NEAREST)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            return html.Img(src=uri, style={"marginRight": "0.5rem", "border": "2px solid orange"})

        components = [
            html.Div([
                _img_tag(i), html.P(f"Point {i} label: {li}")
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                _img_tag(j), html.P(f"Point {j} label: {lj}")
            ], style={"display": "flex", "alignItems": "center"})
        ]
        
        # Add interpolated point if it exists
        if interpolated_point is not None:
            components.append(
                html.Div([
                    _interpolated_img_tag(),
                    html.P(f"Interpolated point (t={t_value:.1f})")
                ], style={"display": "flex", "alignItems": "center", "marginTop": "0.5rem"})
            )
        
        return html.Div(components)


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