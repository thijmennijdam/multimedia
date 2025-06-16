from __future__ import annotations
import argparse
import warnings
import base64
import io
from pathlib import Path

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_daq as daq
import plotly.graph_objs as go
from sklearn.datasets import load_digits, load_iris, load_wine
from PIL import Image


import umap

warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
)


# ---------------------------------------------------------------------------
# Projection helpers
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
    emb2 = umap.UMAP(
        n_components=2,
        n_jobs=1,
        random_state=42,
        output_metric="hyperboloid",
    ).fit(x)
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
# Dataset loading (global so callbacks can access)
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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["iris", "wine", "digits"], default="digits")
parser.add_argument("--debug", action="store_true")
ARGS = parser.parse_args()

DATA, LABELS, FEATURE_NAMES, TARGET_NAMES, IMAGES = _load_dataset(ARGS.dataset)

# ---------------------------------------------------------------------------
# UI helper components (pure functions → no side-effects)
# ---------------------------------------------------------------------------


def _config_panel() -> html.Div:
    """Displays configuration options and mode controls."""
    return html.Div(
        [
            html.H4("Configuration"),
            html.Label("Projection"),
            html.Div([
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
                dcc.Loading(
                    id="proj-loading",
                    type="circle",
                    style={"position": "absolute", "right": "10px", "top": "50%", "transform": "translateY(-50%)"},
                ),
            ], style={"position": "relative"}),
            html.Br(),
            html.Label("Mode"),
            html.Div(
                [
                    html.Button(
                        "Compare",
                        id="compare-btn",
                        style={  # Default selected style
                            "backgroundColor": "green",
                            "color": "white",
                            "border": "none",
                            "padding": "0.5rem 1rem",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "width": "33.3%",
                        },
                    ),
                    html.Button(
                        "Interpolate",
                        id="interpolate-mode-btn",
                        style={
                            "backgroundColor": "#007bff",
                            "color": "white",
                            "border": "none",
                            "padding": "0.5rem 1rem",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "width": "33.3%",
                        },
                    ),
                    html.Button(
                        "Tree",
                        id="tree-mode-btn",
                        style={
                            "backgroundColor": "#007bff",
                            "color": "white",
                            "border": "none",
                            "padding": "0.5rem 1rem",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "width": "33.3%",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "0.5rem", "marginBottom": "1rem"},
            ),
            html.P(id="mode-instructions", children="Select up to 5 points."),
            html.Div(
                id="interpolate-controls",
                style={"display": "none"},  # Hidden by default
                children=[
                    html.Button(
                        "Interpolate Points",
                        id="run-interpolate-btn",
                        disabled=True,
                        style={
                            "backgroundColor": "#007bff",
                            "color": "white",
                            "border": "none",
                            "padding": "0.5rem 1rem",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "marginTop": "0.5rem",
                            "width": "100%",
                        },
                    ),
                    html.Div(
                        [
                            html.Label("Interpolation factor (t):"),
                            dcc.Slider(
                                id="interpolation-slider",
                                min=0,
                                max=1,
                                step=0.1,
                                value=0.5,
                                marks={i / 10: str(i / 10) for i in range(11)},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        style={"marginTop": "1rem"},
                    ),
                ],
            ),
        ],
        style={
            "width": 240,
            "padding": "1rem",
            "borderRight": "1px solid #ddd",
            "backgroundColor": "white",
            "borderRadius": "8px",
            "margin": "1rem",
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
    return html.Div(
        [
            html.Div(
                [
                    # 3D Plot
                    html.Div(
                        dcc.Graph(
                            id="scatter-3d",
                            figure=_empty_fig3d(),
                            style={"width": "100%", "height": "100%", "aspectRatio": "1"},
                            config={"displayModeBar": False},
                        ),
                        style={
                            "flex": 1,
                            "minWidth": "0",
                            "aspectRatio": "1 / 1",  # Ensures square aspect
                            "maxWidth": "600px",     # Optional: limit max size
                            "marginRight": "1rem"
                        },
                    ),
                    # 2D Plot
                    html.Div(
                        dcc.Graph(
                            id="scatter-disk",
                            figure=_empty_fig3d(),
                            style={"width": "100%", "height": "100%", "aspectRatio": "1"},
                            config={"displayModeBar": False},
                        ),
                        style={
                            "flex": 1,
                            "minWidth": "0",
                            "aspectRatio": "1 / 1",
                            "maxWidth": "600px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "justifyContent": "center",
                    "alignItems": "flex-start",
                    "gap": "1rem",
                    "width": "100%",
                    "height": "600px",  # Fixed or calculated height for the plots
                },
            ),
            html.Div(style={"height": "2rem"}),  # Spacer
            # (Add any other widgets below, or leave blank for now)
        ],
        style={
            "flex": 1,
            "padding": "1rem",
            "margin": "1rem",
            "backgroundColor": "white",
            "borderRadius": "8px",
            "display": "flex",
            "flexDirection": "column",
        },
    )


def _decode_point(idx: int, show_features: bool = True) -> html.Div:
    """Create a detailed view of a data point showing its features."""
    if idx is None:
        return html.Div("No point selected", style={"color": "#666", "fontStyle": "italic"})
    
    # Get the point's data and label
    point_data = DATA[idx]
    label = TARGET_NAMES[LABELS[idx]] if TARGET_NAMES is not None else LABELS[idx]
    
    # Create feature list only if show_features is True
    features = None
    if show_features:
        features = html.Div([
            html.Div([
                html.Span(f"{name}: ", style={"fontWeight": "bold", "color": "#444"}),
                html.Span(f"{value:.3f}", style={"color": "#666"})
            ], style={"marginBottom": "0.25rem"})
            for name, value in zip(FEATURE_NAMES, point_data)
        ], style={"fontSize": "0.9rem", "marginTop": "0.5rem"})
    
    return html.Div([
        # Image (if available)
        _create_img_tag(idx),
        # Label
        html.P(f"Point {idx}", style={"fontWeight": "bold", "margin": "0.5rem 0 0.25rem 0"}),
        html.P(f"Label: {label}", style={"color": "#007bff", "margin": "0 0 0.5rem 0"}),
        # Features (only if show_features is True)
        features if features is not None else None
    ], style={"padding": "0.5rem", "border": "1px solid #eee", "borderRadius": "4px"})


def _tree_node(title: str, content: html.Div, is_current: bool = False) -> html.Div:
    """Create a styled tree node with title and content."""
    return html.Div([
        html.H6(
            title,
            style={
                "color": "#007bff" if is_current else "#666",
                "fontWeight": "bold" if is_current else "normal",
                "margin": "0 0 0.5rem 0",
                "padding": "0.5rem",
                "backgroundColor": "#f8f9fa" if is_current else "transparent",
                "borderRadius": "4px",
            }
        ),
        content
    ], style={
        "marginBottom": "1rem",
        "position": "relative",
        "padding": "0.5rem",
        "backgroundColor": "white",
        "borderRadius": "8px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
    })


def _cmp_panel() -> html.Div:
    return html.Div(
        [
            html.H4("Point comparison"),
            # Tree traversal section (hidden by default)
            html.Div(
                [
                    html.H5(
                        "Tree Traversal",
                        style={
                            "marginTop": "1rem",
                            "color": "#007bff",
                            "padding": "0.5rem",
                            "borderBottom": "2px solid #007bff",
                            "marginBottom": "1rem"
                        }
                    ),
                    html.Div(
                        [
                            # Visual tree container
                            html.Div(
                                [
                                    # Parent node
                                    _tree_node("Parent Node", html.Div(id="tree-parent")),
                                    # Connection line
                                    html.Div(
                                        style={
                                            "height": "2rem",
                                            "width": "2px",
                                            "backgroundColor": "#007bff",
                                            "margin": "0 auto",
                                            "position": "relative",
                                        }
                                    ),
                                    # Current node
                                    _tree_node("Current Node", html.Div(id="tree-current"), is_current=True),
                                    # Connection line
                                    html.Div(
                                        style={
                                            "height": "2rem",
                                            "width": "2px",
                                            "backgroundColor": "#007bff",
                                            "margin": "0 auto",
                                            "position": "relative",
                                        }
                                    ),
                                    # Child node
                                    _tree_node("Child Node", html.Div(id="tree-child")),
                                ],
                                id="tree-traversal",
                                style={"display": "none"},  # Hidden by default
                            ),
                        ],
                    ),
                ],
            ),
            # Regular comparison view
            html.Div(id="cmp"),
        ],
        style={
            "width": 320,
            "padding": "1rem",
            "borderLeft": "1px solid #ddd",
            "backgroundColor": "white",
            "borderRadius": "8px",
            "margin": "1rem",
        },
    )


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------


def make_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                html.H2(
                    "Embedding Projector",
                    style={"color": "white", "margin": 0},
                ),
                style={
                    "padding": "0.5rem 1rem",
                    "backgroundColor": "rgb(33, 43, 181)",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                },
            ),
            dcc.Store(id="emb"),
            dcc.Store(id="sel", data=[]),
            dcc.Store(id="mode", data="compare"),  # Default mode
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
            "backgroundColor": "#f7f9fc",
        },
    )


# ---------------------------------------------------------------------------
# Utility & Image helpers
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
    inner_product = p1[2] * p2[2] - p1[0] * p2[0] - p1[1] * p2[1]
    inner_product = max(inner_product, 1.0 + 1e-10)
    d = np.arccosh(inner_product)

    if d < 1e-10:
        return p1.copy()

    sinh_d = np.sinh(d)
    coeff1 = np.sinh((1 - t) * d) / sinh_d
    coeff2 = np.sinh(t * d) / sinh_d
    return coeff1 * p1 + coeff2 * p2

# FIX: Moved image creation logic to top-level helper functions to resolve NameError.
def _create_img_tag(idx: int) -> html.Img | html.Span:
    """Creates a base64 encoded image tag from the IMAGES dataset."""
    if IMAGES is None:
        return html.Span()
    pil = Image.fromarray((IMAGES[idx] * 16).astype("uint8"), mode="L").resize(
        (64, 64), Image.NEAREST
    )
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return html.Img(src=uri, style={"marginRight": "0.5rem", "border": "1px solid #bbb"})


def _create_interpolated_img_tag(i: int, j: int, t_value: float) -> html.Img | html.Span:
    """Creates an image tag for the interpolated point."""
    if IMAGES is None:
        return html.Span()
    img1 = IMAGES[i].astype(np.float32)
    img2 = IMAGES[j].astype(np.float32)
    interpolated_img_data = (1 - t_value) * img1 + t_value * img2
    pil = Image.fromarray(
        (interpolated_img_data * 16).astype("uint8"), mode="L"
    ).resize((64, 64), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return html.Img(
        src=uri,
        style={"marginRight": "0.5rem", "border": "2px solid orange"},
    )

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _fig3d(
    emb: np.ndarray, sel: list[int], interpolated_point: np.ndarray = None
) -> go.Figure:
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
        name="Data points",
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
                name="Selected points",
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
                hoverinfo="text",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision="embedding",
        showlegend=False,
    )
    return fig


def _fig_disk(
    x: np.ndarray, y: np.ndarray, sel: list[int], interpolated_point: np.ndarray = None
) -> go.Figure:
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
        name="Data points",
    )
    traces = [base]

    if sel:
        traces.append(
            go.Scatter(
                x=x[sel],
                y=y[sel],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Selected points",
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
                hoverinfo="text",
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
# Callback registration
# ---------------------------------------------------------------------------


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        [Output("scatter-3d", "figure"), Output("scatter-disk", "figure")],
        Input("emb", "data"),
        Input("sel", "data"),
        Input("proj", "value"),
        Input("interpolated-point", "data"),
    )
    def _scatter(edata, sel, proj, interpolated_point):
        if edata is None:
            return _empty_fig3d(), _empty_fig3d()
        
        emb = np.asarray(edata, dtype=np.float32)
        sel = sel or []
        interp_point = (
            np.asarray(interpolated_point, dtype=np.float32)
            if interpolated_point is not None
            else None
        )

        # Always create 3D view
        fig_3d = _fig3d(emb, sel, interp_point)

        # Create Poincaré disk view for hyperbolic projection, otherwise use 2D projection
        if "Hyperbolic" in proj:
            xh, yh, zh = emb[:, 0], emb[:, 1], emb[:, 2]
            dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
            interp_transformed = None
            if interp_point is not None:
                interp_dx = interp_point[0] / (1.0 + interp_point[2])
                interp_dy = interp_point[1] / (1.0 + interp_point[2])
                interp_transformed = np.array([interp_dx, interp_dy])
        else:
            dx, dy = emb[:, 0], emb[:, 1]
            interp_transformed = interp_point[:2] if interp_point is not None else None
        
        fig_disk = _fig_disk(dx, dy, sel, interp_transformed)
        
        # Add titles to distinguish the views
        fig_3d.update_layout(title="3D View")
        fig_disk.update_layout(title="Poincaré Disk View" if "Hyperbolic" in proj else "2D View")

        return fig_3d, fig_disk

    @app.callback(
        [Output("emb", "data"),
         Output("proj", "disabled"),
         Output("proj-loading", "parent_style")],
        Input("proj", "value"),
        prevent_initial_call=True,
    )
    def _compute(method):
        # Show loading state and disable dropdown
        loading_style = {"display": "block"}
        try:
            # Compute the embedding
            result = PROJECTIONS[method](DATA).tolist()
            # Hide loading state and enable dropdown
            loading_style = {"display": "none"}
            return result, False, loading_style
        except Exception as e:
            print(f"Error computing projection: {e}")
            # In case of error, re-enable the dropdown but keep loading state
            return dash.no_update, False, loading_style

    @app.callback(
        Output("sel", "data"),
        [Input("scatter-3d", "clickData"), Input("scatter-disk", "clickData")],
        State("sel", "data"),
        State("mode", "data"),
        prevent_initial_call=True,
    )
    def _select(click_3d, click_disk, sel, mode):
        ctx = callback_context
        if not ctx.triggered:
            return sel
        
        # Use whichever graph was clicked
        click = click_3d if ctx.triggered[0]["prop_id"] == "scatter-3d.clickData" else click_disk
        idx = _clicked(click)
        if idx is None:
            return sel
        
        sel = sel or []
        if idx in sel:
            sel.remove(idx)
        else:
            sel.append(idx)
            if mode == "compare":
                max_points = 5
            elif mode == "interpolate":
                max_points = 2
            else:  # tree mode
                max_points = 1
            if len(sel) > max_points:
                sel = sel[-max_points:]
        return sel

    @app.callback(
        Output("run-interpolate-btn", "disabled"),
        Input("sel", "data"),
        Input("mode", "data"),
    )
    def _update_button_state(sel, mode):
        if mode == "interpolate":
            return not (sel and len(sel) == 2)
        return True

    @app.callback(
        Output("interpolated-point", "data"),
        Input("run-interpolate-btn", "n_clicks"),
        State("sel", "data"),
        State("interpolation-slider", "value"),
        State("emb", "data"),
        State("proj", "value"),
        prevent_initial_call=True,
    )
    def _interpolate(n_clicks, sel, t, edata, proj):
        if not (n_clicks and sel and len(sel) == 2 and edata):
            return None

        emb = np.asarray(edata, dtype=np.float32)
        i, j = sel[:2]
        p1, p2 = emb[i], emb[j]

        if "Hyperbolic" in proj:
            interpolated = _interpolate_hyperbolic(p1, p2, t)
        else:
            interpolated = _interpolate_points(p1, p2, t)
        return interpolated.tolist()

    @app.callback(
        [Output("tree-parent", "children"),
         Output("tree-current", "children"),
         Output("tree-child", "children")],
        Input("sel", "data"),
        Input("mode", "data"),
    )
    def _update_tree_view(sel, mode):
        if mode != "tree" or not sel or len(sel) != 1:
            return _decode_point(None), _decode_point(None), _decode_point(None)
        
        # Get the selected point
        idx = sel[0]
        
        # For now, show the same point in all three positions
        # In the future, this is where we'd look up the actual parent and child
        # Don't show features in tree mode
        decoded_point = _decode_point(idx, show_features=False)
        return decoded_point, decoded_point, decoded_point

    @app.callback(
        Output("cmp", "children"),
        Input("sel", "data"),
        Input("interpolated-point", "data"),
        Input("interpolation-slider", "value"),
        Input("mode", "data"),
    )
    def _compare(sel, interpolated_point, t_value, mode):
        if mode == "tree":
            return None  # Hide regular comparison when tree view is enabled
            
        sel = sel or []

        if mode == "compare":
            if not sel:
                return html.P("Select up to 5 points to compare.")
            components = []
            for idx in sel:
                label = (
                    TARGET_NAMES[LABELS[idx]] if TARGET_NAMES is not None else LABELS[idx]
                )
                components.append(
                    html.Div(
                        [_create_img_tag(idx), html.P(f"Point {idx} label: {label}")],
                        style={"display": "flex", "alignItems": "center"},
                    )
                )
            return html.Div(components)

        if mode == "interpolate":
            if len(sel) < 2:
                return html.P("Select two distinct points to interpolate.")
            i, j = sel[:2]
            li = TARGET_NAMES[LABELS[i]] if TARGET_NAMES is not None else LABELS[i]
            lj = TARGET_NAMES[LABELS[j]] if TARGET_NAMES is not None else LABELS[j]

            components = [
                html.Div(
                    [_create_img_tag(i), html.P(f"Point {i} label: {li}")],
                    style={"display": "flex", "alignItems": "center"},
                ),
                html.Div(
                    [_create_img_tag(j), html.P(f"Point {j} label: {lj}")],
                    style={"display": "flex", "alignItems": "center"},
                ),
            ]

            if interpolated_point is not None:
                interpolated_img = _create_interpolated_img_tag(i, j, t_value)
                components.append(
                    html.Div(
                        [
                            interpolated_img,
                            html.P(f"Interpolated point (t={t_value:.1f})"),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginTop": "0.5rem",
                        },
                    )
                )
            return html.Div(components)
        
        return html.Div()

    @app.callback(
        Output("mode", "data"),
        Output("compare-btn", "style"),
        Output("interpolate-mode-btn", "style"),
        Output("tree-mode-btn", "style"),
        Output("interpolate-controls", "style"),
        Output("tree-traversal", "style"),
        Output("mode-instructions", "children"),
        # IMPORTANT: These two outputs clear the selections when the mode changes.
        Output("sel", "data", allow_duplicate=True),
        Output("interpolated-point", "data", allow_duplicate=True),
        Input("compare-btn", "n_clicks"),
        Input("interpolate-mode-btn", "n_clicks"),
        Input("tree-mode-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _update_mode(compare_clicks, interpolate_clicks, tree_clicks):
        """
        Handles mode switching. Crucially, it clears any existing point 
        selections to prevent carry-over between modes.
        """
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        base_style = {
            "color": "white",
            "border": "none",
            "padding": "0.5rem 1rem",
            "borderRadius": "6px",
            "cursor": "pointer",
            "width": "33.3%",
        }
        compare_style = {**base_style, "backgroundColor": "#007bff"}
        interpolate_style = {**base_style, "backgroundColor": "#007bff"}
        tree_style = {**base_style, "backgroundColor": "#007bff"}
        selected_style = {**base_style, "backgroundColor": "green"}

        mode = "compare"
        interpolate_controls_style = {"display": "none"}
        tree_traversal_style = {"display": "none"}
        instructions = "Select up to 5 points to compare."

        if triggered_id == "interpolate-mode-btn":
            mode = "interpolate"
            interpolate_style = selected_style
            interpolate_controls_style = {"display": "block"}
            instructions = "Select 2 points to interpolate."
        elif triggered_id == "tree-mode-btn":
            mode = "tree"
            tree_style = selected_style
            tree_traversal_style = {"display": "block"}
            instructions = "Select 1 point to view its lineage."
        else:  # Default to compare mode
            compare_style = selected_style

        # Return an empty list for 'sel' and None for 'interpolated-point'
        # to ensure a clean state every time the user switches modes.
        return (
            mode,
            compare_style,
            interpolate_style,
            tree_style,
            interpolate_controls_style,
            tree_traversal_style,
            instructions,
            [],  # Clear selected points
            None,  # Clear any interpolated point
        )


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------


def build_app(debug: bool = False) -> dash.Dash:
    # Use prevent_initial_callbacks to handle outputs with allow_duplicate=True
    app = dash.Dash(__name__, prevent_initial_callbacks="initial_duplicate")
    app.layout = make_layout()
    register_callbacks(app)
    return app


if __name__ == "__main__":
    build_app(debug=ARGS.debug).run(debug=ARGS.debug)