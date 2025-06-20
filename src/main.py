from __future__ import annotations
import argparse
import warnings
import base64
import io
import json
from pathlib import Path

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
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
# Dataset loading
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
parser.add_argument("--dataset", choices=["iris", "wine", "digits", "imagenet"], default="digits")
parser.add_argument("--debug", action="store_true")
ARGS = parser.parse_args()

# MODIFIED: Global data is no longer loaded here. It will be managed by Dash Stores.

# ---------------------------------------------------------------------------
# UI helper components (pure functions → no side-effects)
# ---------------------------------------------------------------------------


def _config_panel() -> html.Div:
    """Displays configuration options and mode controls."""
    return html.Div(
        [
            html.H4("Configuration"),
            # ADDED: Dataset Dropdown
            html.Label("Dataset"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[
                    {"label": "Digits", "value": "digits"},
                    {"label": "Wine", "value": "wine"},
                    {"label": "Iris", "value": "iris"},
                    {"label": "ImageNet subset", "value": "imagenet"},
                ],
                value=ARGS.dataset,  # Use parsed arg as default
                clearable=False,
                style={
                    "backgroundColor": "white",
                    "border": "1px solid #ccc",
                    "borderRadius": "6px",
                    "color": "black",
                    # "marginBottom": "1rem",
                },
            ),
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
            "width": "240px",
            "padding": "1rem",
            "backgroundColor": "white",
            "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            "flexShrink": 0,  # Prevent panel from shrinking
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
                            style={"width": "100%", "height": "100%"},
                            config={"displayModeBar": False},
                        ),
                        style={
                            "flex": 1,
                            "minWidth": 0,
                            "aspectRatio": "1 / 1",
                            "maxWidth": "600px",
                            "marginRight": "1rem",
                            "marginTop": "2rem",
                        },
                    ),
                    # 2D Plot
                    html.Div(
                        dcc.Graph(
                            id="scatter-disk",
                            figure=_empty_fig3d(),
                            style={"width": "100%", "height": "100%"},
                            config={"displayModeBar": False},
                        ),
                        style={
                            "flex": 1,
                            "minWidth": 0,
                            "aspectRatio": "1 / 1",
                            "maxWidth": "600px",
                            "marginTop": "2rem",
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
                    "height": "100%",
                },
            ),
        ],
        style={
            "flex": 1,
            "padding": "1rem",
            "backgroundColor": "white",
            "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            "display": "flex",
            "flexDirection": "column",
            "minHeight": 0,  # Important for flex child
            "overflow": "hidden",  # Prevent scrollbars
        },
    )


# MODIFIED: Helper function now takes all data as arguments instead of using globals.
def _decode_point(
    idx: int,
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    images: np.ndarray | None,
    show_features: bool = True,
) -> html.Div:
    """Create a detailed view of a data point showing its features."""
    if idx is None:
        return html.Div("No point selected", style={"color": "#666", "fontStyle": "italic"})

    # Get the point's data and label
    point_data = data[idx]
    label = target_names[labels[idx]] if target_names is not None else labels[idx]

    # Create feature list only if show_features is True
    features = None
    if show_features:
        features = html.Div([
            html.Div([
                html.Span(f"{name}: ", style={"fontWeight": "bold", "color": "#444"}),
                html.Span(f"{value:.3f}", style={"color": "#666"})
            ], style={"marginBottom": "0.25rem"})
            for name, value in zip(feature_names, point_data)
        ], style={"fontSize": "0.9rem", "marginTop": "0.5rem"})

    return html.Div([
        # Image (if available)
        _create_img_tag(idx, images),
        # Label
        html.P(f"Point {idx}", style={"fontWeight": "bold", "margin": "0.5rem 0 0.25rem 0"}),
        html.P(f"Label: {label}", style={"color": "#007bff", "margin": "0 0 0.5rem 0"}),
        # Features (only if show_features is True)
        features if features is not None else None
    ], style={
        "padding": "0.5rem",
        "border": "1px solid #eee",
        "borderRadius": "4px",
        "cursor": "pointer",
        "transition": "background-color 0.2s",
        "hover": {
            "backgroundColor": "#f8f9fa"
        }
    }, id={"type": "point-card", "index": idx})


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
            "width": "320px",
            "padding": "1rem",
            "backgroundColor": "white",
            "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            "flexShrink": 0,  # Prevent panel from shrinking
            "overflowY": "auto",  # Allow scrolling if content overflows
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
                    "HIVE: Hyperbolic Interactive Visualization Explorer",
                    style={"color": "white", "margin": 0, "padding": "0.5rem 0"},
                ),
                style={
                    "padding": "0 1rem",
                    "backgroundColor": "rgb(33, 43, 181)",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                    "height": "48px",
                    "display": "flex",
                    "alignItems": "center",
                },
            ),
            # ADDED: Data Stores to manage state dynamically
            dcc.Store(id="data-store"),
            dcc.Store(id="labels-store"),
            dcc.Store(id="feature-names-store"),
            dcc.Store(id="target-names-store"),
            dcc.Store(id="images-store"),
            dcc.Store(id="meta-store"),
            dcc.Store(id="points-store"),
            # Existing Stores
            dcc.Store(id="emb"),
            dcc.Store(id="sel", data=[]),
            dcc.Store(id="mode", data="compare"),  # Default mode
            dcc.Store(id="interpolated-point"),
            html.Div(
                [_config_panel(), _centre_panel(), _cmp_panel()],
                style={
                    "display": "flex",
                    "flex": 1,
                    "minHeight": 0,  # Important for flex child
                    "padding": "0.5rem",
                    "gap": "0.5rem",
                },
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
            "width": "100vw",
            "margin": 0,
            "padding": 0,
            "fontFamily": "Inter, sans-serif",
            "backgroundColor": "#f7f9fc",
            "overflow": "hidden",  # Prevent scrollbars
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


# ---------------------------------------------------------------------------
# Image encoding helper (for datasets that reference image files on disk)
# ---------------------------------------------------------------------------

_IMAGENET_ROOT = Path("imagenet-subset")


def _encode_image(rel_path: str) -> str:
    """Return base64 data URI for *rel_path* (relative to *_IMAGENET_ROOT*)."""
    img_path = _IMAGENET_ROOT / rel_path
    mime = {
        ".jpeg": "jpeg",
        ".jpg": "jpeg",
        ".png": "png",
    }.get(img_path.suffix.lower(), "jpeg")

    try:
        with open(img_path, "rb") as f_img:
            enc = base64.b64encode(f_img.read()).decode()
    except FileNotFoundError:
        # If image not found, return empty string to avoid broken UI
        return ""

    return f"data:image/{mime};base64,{enc}"


# MODIFIED: Helper function now takes `images` data as an argument.
def _create_img_tag(idx: int, images: np.ndarray | None) -> html.Img | html.Span:
    """Creates a base64 encoded image tag from the IMAGES dataset."""
    # Gracefully handle missing images array
    if images is None:
        return html.Span()

    # Case 1: images are in-memory numpy arrays (e.g. sklearn *digits* dataset)
    if isinstance(images, (list, np.ndarray)) and isinstance(images[idx], (list, np.ndarray)):
        images_np = np.asarray(images)
        pil = (
            Image.fromarray((images_np[idx] * 16).astype("uint8"), mode="L")
            .resize((64, 64), Image.NEAREST)
        )
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return html.Img(
            src=uri, style={"marginRight": "0.5rem", "border": "1px solid #bbb"}
        )

    # Case 2: images are file paths on disk (e.g. ImageNet subset)
    try:
        img_rel = images[idx]
        uri = _encode_image(str(img_rel))  # type: ignore[arg-type]
        return html.Img(
            src=uri, style={"marginRight": "0.5rem", "border": "1px solid #bbb"}
        )
    except Exception:
        # Fallback: empty span on error (e.g. file missing)
        return html.Span()


# MODIFIED: Helper function now takes `images` data as an argument.
def _create_interpolated_img_tag(i: int, j: int, t_value: float, images: np.ndarray | None) -> html.Img | html.Span:
    """Creates an image tag for the interpolated point."""
    if images is None:
        return html.Span()
    
    images_np = np.asarray(images)
    img1 = images_np[i].astype(np.float32)
    img2 = images_np[j].astype(np.float32)
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


def _create_close_button(idx: int) -> html.Button:
    """Create a styled close button for removing points from selection."""
    return html.Button(
        "×",  # × symbol
        id={"type": "close-button", "index": idx},
        style={
            "position": "absolute",
            "top": "0.25rem",
            "right": "0.25rem",
            "width": "1.5rem",
            "height": "1.5rem",
            "borderRadius": "50%",
            "border": "none",
            "backgroundColor": "#ff4444",
            "color": "white",
            "fontSize": "1rem",
            "lineHeight": "1",
            "cursor": "pointer",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "0",
            "transition": "background-color 0.2s",
            "hover": {
                "backgroundColor": "#cc0000"
            }
        }
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

# MODIFIED: Helper function now takes `labels` and `target_names` as arguments.
def _fig3d(
    emb: np.ndarray,
    sel: list[int],
    labels: np.ndarray,
    target_names: list[str] | None,
    interpolated_point: np.ndarray = None,
) -> go.Figure:
    labels_txt = [
        f"{i}: {target_names[labels[i]] if target_names is not None else labels[i]}"
        for i in range(len(labels))
    ]
    base = go.Scatter3d(
        x=emb[:, 0],
        y=emb[:, 1],
        z=emb[:, 2],
        mode="markers",
        text=labels_txt,
        hoverinfo="text",
        marker=dict(size=6, opacity=0.8, color=labels, colorscale="Viridis"),
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

# MODIFIED: Helper function now takes `labels` and `target_names` as arguments.
def _fig_disk(
    x: np.ndarray,
    y: np.ndarray,
    sel: list[int],
    labels: np.ndarray,
    target_names: list[str] | None,
    interpolated_point: np.ndarray = None,
) -> go.Figure:
    base = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        text=[
            f"{i}: {target_names[labels[i]] if target_names is not None else labels[i]}"
            for i in range(len(labels))
        ],
        hoverinfo="text",
        marker=dict(size=6, opacity=0.8, color=labels, colorscale="Viridis"),
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
    # ADDED: This callback loads dataset data into stores when the dropdown changes.
    @app.callback(
        Output("data-store", "data"),
        Output("labels-store", "data"),
        Output("feature-names-store", "data"),
        Output("target-names-store", "data"),
        Output("images-store", "data"),
        Output("meta-store", "data"),
        Output("points-store", "data"),
        Output("sel", "data", allow_duplicate=True),
        Output("interpolated-point", "data", allow_duplicate=True),
        Output("emb", "data", allow_duplicate=True),
        Input("dataset-dropdown", "value"),
        prevent_initial_call=False,  # IMPORTANT: Load default dataset on startup
    )
    def _update_dataset_stores(dataset_name):
        if not dataset_name:
            return dash.no_update

        # ------------------------------------------------------------------
        # Classic sklearn datasets (digits / iris / wine)
        # ------------------------------------------------------------------
        if dataset_name in {"digits", "iris", "wine"}:
            data, labels, feature_names, target_names, images = _load_dataset(dataset_name)

            data_list = data.tolist()
            labels_list = labels.tolist()
            target_names_list = target_names.tolist() if target_names is not None else None
            images_list = images.tolist() if images is not None else None

            return (
                data_list,
                labels_list,
                feature_names,
                target_names_list,
                images_list,
                None,  # meta-store
                None,  # points-store
                [],
                None,
                None,
            )

        # ------------------------------------------------------------------
        # ImageNet subset tree dataset (generated via create_trees.py)
        # ------------------------------------------------------------------
        if dataset_name == "imagenet":
            try:
                with open("dataset/meta.json", "r", encoding="utf-8") as f_meta:
                    meta = json.load(f_meta)
                with open("dataset/points.json", "r", encoding="utf-8") as f_pts:
                    points = json.load(f_pts)
            except FileNotFoundError:
                print("meta.json / points.json not found in ./dataset – did you run create_trees.py?")
                return dash.no_update

            # ------------------------------------------------------------------
            # Flatten embeddings into data matrix
            # ------------------------------------------------------------------
            embeddings = np.array([pt["embedding"] for pt in points], dtype=np.float32)

            # Map synset_id → integer label (for colouring)
            synset_ids = [pt["synset_id"] for pt in points]
            unique_synsets = sorted({sid for sid in synset_ids})
            syn_to_int = {sid: i for i, sid in enumerate(unique_synsets)}
            labels = np.array([syn_to_int[s] for s in synset_ids], dtype=int)

            feature_names = [f"dim{i}" for i in range(embeddings.shape[1])]
            target_names = unique_synsets  # optional mapping to synset ids

            # For images-list we keep relative paths where available else None
            images_list: list[str | None] = [pt.get("image_path") for pt in points]

            return (
                embeddings.tolist(),
                labels.tolist(),
                feature_names,
                target_names,
                images_list,
                meta,
                points,
                [],
                None,
                None,
            )

        # ------------------------------------------------------------------
        # Unknown dataset
        # ------------------------------------------------------------------
        return dash.no_update


    # MODIFIED: Callback now reads from stores instead of globals.
    @app.callback(
        [Output("scatter-3d", "figure"), Output("scatter-disk", "figure")],
        Input("emb", "data"),
        Input("sel", "data"),
        Input("proj", "value"),
        Input("interpolated-point", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
    )
    def _scatter(edata, sel, proj, interpolated_point, labels_data, target_names):
        if edata is None or labels_data is None:
            return _empty_fig3d(), _empty_fig3d()
        
        emb = np.asarray(edata, dtype=np.float32)
        labels = np.asarray(labels_data, dtype=int)
        sel = sel or []
        interp_point = (
            np.asarray(interpolated_point, dtype=np.float32)
            if interpolated_point is not None
            else None
        )

        fig_3d = _fig3d(emb, sel, labels, target_names, interp_point)

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
        
        fig_disk = _fig_disk(dx, dy, sel, labels, target_names, interp_transformed)
        
        return fig_3d, fig_disk

    # MODIFIED: Callback now triggers on data change and reads from store.
    @app.callback(
        Output("emb", "data"),
        Output("proj-loading", "parent_style"),
        Input("proj", "value"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def _compute(method, data):
        if data is None:
            return None, {"display": "none"}
            
        loading_style = {"display": "block"}
        try:
            data_np = np.asarray(data, dtype=np.float32)
            result = PROJECTIONS[method](data_np).tolist()
            loading_style = {"display": "none"}
            return result, loading_style
        except Exception as e:
            print(f"Error computing projection: {e}")
            return None, {"display": "none"}

    @app.callback(
        Output("sel", "data"),
        [
            Input("scatter-3d", "clickData"),
            Input("scatter-disk", "clickData"),
            Input({"type": "close-button", "index": dash.ALL}, "n_clicks")
        ],
        State("sel", "data"),
        State("mode", "data"),
        prevent_initial_call=True,
    )
    def _select(click_3d, click_disk, close_clicks, sel, mode):
        ctx = callback_context
        if not ctx.triggered or not ctx.triggered_id:
            return dash.no_update

        triggered_id = ctx.triggered_id
        current_sel = sel or []

        if triggered_id in ["scatter-3d", "scatter-disk"]:
            click_data = ctx.inputs[f"{triggered_id}.clickData"]
            idx = _clicked(click_data)
            if idx is None:
                return dash.no_update

            if idx in current_sel:
                new_sel = [i for i in current_sel if i != idx]
            else:
                new_sel = current_sel + [idx]
                
                max_points = 1
                if mode == "compare": max_points = 5
                elif mode == "interpolate": max_points = 2
                
                if len(new_sel) > max_points:
                    new_sel = new_sel[-max_points:]
            return new_sel

        elif isinstance(triggered_id, dict) and triggered_id.get("type") == "close-button":
            if not ctx.triggered[0]['value']:
                return dash.no_update
            
            idx_to_remove = triggered_id.get("index")
            if idx_to_remove in current_sel:
                new_sel = [i for i in current_sel if i != idx_to_remove]
                return new_sel

        return dash.no_update

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

    # MODIFIED: Callback now reads meta/points stores to build detailed tree view.
    @app.callback(
        [Output("tree-parent", "children"),
         Output("tree-current", "children"),
         Output("tree-child", "children")],
        Input("sel", "data"),
        Input("mode", "data"),
        State("meta-store", "data"),
        State("points-store", "data"),
    )
    def _update_tree_view(sel, mode, meta, points):
        """Render a three-level tree similar to the *create_trees* demo.

        The three nodes are:
          • Parent   → Word/Name
          • Current  → Description
          • Child    → Representative image (or the selected image if kind==image)
        """

        if mode != "tree" or not sel or len(sel) != 1 or meta is None or points is None:
            # Nothing to show
            return html.Span(), html.Span(), html.Span()

        idx = sel[0]

        try:
            pt = points[idx]
        except (IndexError, TypeError):
            return html.Span(), html.Span(), html.Span()

        synset_id: str = pt.get("synset_id", "?")
        pt_kind: str = pt.get("kind", "?")

        meta_row = meta.get(synset_id, {}) if isinstance(meta, dict) else {}

        # Parent node: Word/Name
        parent_div = html.Div(
            [
                html.H4(meta_row.get("name", synset_id), style={"margin": "0.25rem 0", "color": "#007bff"}),
            ],
            style={
                "padding": "0.5rem",
                "backgroundColor": "#eef",
                "borderRadius": "4px",
                "marginBottom": "0.5rem",
            },
        )

        # Current node: Description
        current_div = html.Div(
            [
                html.P(meta_row.get("description", "(no description)"), style={"margin": 0}),
            ],
            style={
                "padding": "0.5rem",
                "borderLeft": "3px solid #99c",
                "marginLeft": "1rem",
                "marginBottom": "0.5rem",
            },
        )

        # Choose image
        if pt_kind == "image" and pt.get("image_path"):
            img_rel = pt["image_path"]
        else:
            img_rel = meta_row.get("first_image_path")

        img_src = _encode_image(img_rel) if img_rel else ""

        child_div = html.Div(
            [
                html.Img(src=img_src, style={"maxWidth": "220px", "border": "1px solid #ccc"}),
                html.P(
                    f"Image source: {img_rel}" if img_rel else "",
                    style={"fontSize": "0.75rem", "color": "#666"},
                ),
            ],
            style={
                "padding": "0.5rem",
                "borderLeft": "3px solid #c9c",
                "marginLeft": "2rem",
            },
        )

        return parent_div, current_div, child_div

    # MODIFIED: Callback now reads from stores to get data for comparison.
    @app.callback(
        Output("cmp", "children"),
        Input("sel", "data"),
        Input("interpolated-point", "data"),
        Input("interpolation-slider", "value"),
        Input("mode", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        State("images-store", "data"),
    )
    def _compare(sel, interpolated_point, t_value, mode, labels_data, target_names, images):
        if labels_data is None:
            return html.P("Select a dataset to begin.")

        labels = np.asarray(labels_data)
        sel = sel or []

        if mode == "tree":
            return None
            
        if mode == "compare":
            if not sel: return html.P("Select up to 5 points to compare.")
            components = []
            for idx in sel:
                label = target_names[labels[idx]] if target_names is not None else labels[idx]
                components.append(
                    html.Div([
                        _create_close_button(idx),
                        _create_img_tag(idx, images), 
                        html.P(f"Point {idx} label: {label}")
                    ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"})
                )
            return html.Div(components)

        if mode == "interpolate":
            if len(sel) < 2: return html.P("Select two distinct points to interpolate.")
            i, j = sel[:2]
            li = target_names[labels[i]] if target_names is not None else labels[i]
            lj = target_names[labels[j]] if target_names is not None else labels[j]

            components = [
                html.Div([
                    _create_close_button(i),
                    _create_img_tag(i, images), 
                    html.P(f"Point {i} label: {li}")
                ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"}),
                html.Div([
                    _create_close_button(j),
                    _create_img_tag(j, images), 
                    html.P(f"Point {j} label: {lj}")
                ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"}),
            ]

            if interpolated_point is not None:
                interpolated_img = _create_interpolated_img_tag(i, j, t_value, images)
                components.append(
                    html.Div([
                        interpolated_img,
                        html.P(f"Interpolated point (t={t_value:.1f})"),
                    ], style={"display": "flex", "alignItems": "center", "marginTop": "0.5rem"})
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
        Output("sel", "data", allow_duplicate=True),
        Output("interpolated-point", "data", allow_duplicate=True),
        Input("compare-btn", "n_clicks"),
        Input("interpolate-mode-btn", "n_clicks"),
        Input("tree-mode-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _update_mode(compare_clicks, interpolate_clicks, tree_clicks):
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        base_style = {"color": "white", "border": "none", "padding": "0.5rem 1rem", "borderRadius": "6px", "cursor": "pointer", "width": "33.3%"}
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
        else:
            compare_style = selected_style

        return (mode, compare_style, interpolate_style, tree_style, interpolate_controls_style, tree_traversal_style, instructions, [], None)


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------


def build_app(debug: bool = False) -> dash.Dash:
    app = dash.Dash(__name__, prevent_initial_callbacks="initial_duplicate")
    app.layout = make_layout()
    register_callbacks(app)
    return app


if __name__ == "__main__":
    build_app(debug=ARGS.debug).run(debug=ARGS.debug)