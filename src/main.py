from __future__ import annotations
import argparse
import warnings
import base64
import io
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
                            "width": "50%",
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
                            "width": "50%",
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
            "fontWeight": "bold",
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
                    id="scatter-container",
                ),
            ],
        ),
        style={
            "flex": 1,
            "padding": "1rem",
            "margin": "1rem",
            "backgroundColor": "white",
            "borderRadius": "8px",
        },
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
        Output("scatter-container", "children"),
        Input("emb", "data"),
        Input("sel", "data"),
        Input("view", "value"),
        Input("proj", "value"),
        Input("interpolated-point", "data"),
    )
    def _scatter(edata, sel, view, proj, interpolated_point):
        if edata is None:
            return None
        emb = np.asarray(edata, dtype=np.float32)

        interp_point = (
            np.asarray(interpolated_point, dtype=np.float32)
            if interpolated_point is not None
            else None
        )

        if view == "3d":
            fig = _fig3d(emb, sel or [], interp_point)
        elif "Hyperbolic" in proj:
            xh, yh, zh = emb[:, 0], emb[:, 1], emb[:, 2]
            dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
            interp_transformed = None
            if interp_point is not None:
                interp_dx = interp_point[0] / (1.0 + interp_point[2])
                interp_dy = interp_point[1] / (1.0 + interp_point[2])
                interp_transformed = np.array([interp_dx, interp_dy])
            fig = _fig_disk(dx, dy, sel or [], interp_transformed)
        else:
            dx, dy = emb[:, 0], emb[:, 1]
            interp_transformed = interp_point[:2] if interp_point is not None else None
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
        State("mode", "data"),
        prevent_initial_call=True,
    )
    def _select(click, sel, mode):
        idx = _clicked(click)
        if idx is None:
            return sel
        sel = sel or []
        if idx in sel:
            sel.remove(idx)
        else:
            sel.append(idx)
            max_points = 5 if mode == "compare" else 2
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
        Output("cmp", "children"),
        Input("sel", "data"),
        Input("interpolated-point", "data"),
        Input("interpolation-slider", "value"),
        Input("mode", "data"),
    )
    def _compare(sel, interpolated_point, t_value, mode):
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
        Output("interpolate-controls", "style"),
        Output("mode-instructions", "children"),
        # IMPORTANT: These two outputs clear the selections when the mode changes.
        Output("sel", "data", allow_duplicate=True),
        Output("interpolated-point", "data", allow_duplicate=True),
        Input("compare-btn", "n_clicks"),
        Input("interpolate-mode-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _update_mode(compare_clicks, interpolate_clicks):
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
            "width": "50%",
        }
        compare_style = {**base_style, "backgroundColor": "#007bff"}
        interpolate_style = {**base_style, "backgroundColor": "#007bff"}
        selected_style = {**base_style, "backgroundColor": "green"}

        mode = "compare"
        interpolate_controls_style = {"display": "none"}
        instructions = "Select up to 5 points to compare."

        if triggered_id == "interpolate-mode-btn":
            mode = "interpolate"
            interpolate_style = selected_style
            interpolate_controls_style = {"display": "block"}
            instructions = "Select 2 points to interpolate."
        else:  # Default to compare mode
            compare_style = selected_style

        # Return an empty list for 'sel' and None for 'interpolated-point'
        # to ensure a clean state every time the user switches modes.
        return (
            mode,
            compare_style,
            interpolate_style,
            interpolate_controls_style,
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