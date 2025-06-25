import dash
from dash import dcc, html

def _config_panel() -> html.Div:
    return html.Div(
        [
            html.H4("Configuration"),
            html.Label("Dataset"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[
                    {"label": "ImageNet subset", "value": "imagenet"},
                    {"label": "GRIT", "value": "grit"},
                ],
                value="imagenet",  # Default, can be overridden
                clearable=False,
                style={
                    "backgroundColor": "white",
                    "border": "1px solid #ccc",
                    "borderRadius": "6px",
                    "color": "black",
                },
            ),
            html.Label("Projection"),
            dcc.Dropdown(
                id="proj",
                options=[
                    {"label": "HoroPCA", "value": "horopca"},
                    {"label": "CO-SNE", "value": "cosne"}
                ],
                value="horopca",
                clearable=False,
                style={
                    "backgroundColor": "white",
                    "border": "1px solid #ccc",
                    "borderRadius": "6px",
                    "color": "black",
                },
            ),
            # Hyperparameters display
            html.Div(
                id="hyperparams-display",
                style={
                    "marginTop": "0.5rem",
                    "marginBottom": "1rem",
                    "padding": "0.75rem",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "6px",
                    "border": "1px solid #e9ecef",
                },
                children=[
                    html.H6("Hyperparameters", style={"margin": "0 0 0.5rem 0", "color": "#495057", "fontSize": "0.9rem"}),
                    html.Div(id="hyperparams-table")
                ]
            ),
            # Projection comparison section
            html.Div([
                html.Button(
                    "Compare Projections",
                    id="compare-projections-btn",
                    style={
                        "backgroundColor": "#6c757d",
                        "color": "white",
                        "border": "none",
                        "padding": "0.5rem 1rem",
                        "borderRadius": "6px",
                        "cursor": "pointer",
                        "width": "100%",
                        "marginBottom": "0.5rem",
                        "transition": "background-color 0.2s",
                    },
                ),
            ]),
            html.Br(),
            html.Label("Mode"),
            html.Div(
                [
                    html.Div([
                        html.Button(
                            "Compare",
                            id="compare-btn",
                            style={
                                "backgroundColor": "green",
                                "color": "white",
                                "border": "none",
                                "padding": "0.5rem 1rem",
                                "borderRadius": "6px",
                                "cursor": "pointer",
                                "width": "100%",
                                "minWidth": "0",
                                "flex": "1 1 0",
                                "boxSizing": "border-box",
                                "transition": "background-color 0.2s",
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
                                "width": "100%",
                                "minWidth": "0",
                                "flex": "1 1 0",
                                "boxSizing": "border-box",
                                "transition": "background-color 0.2s",
                            },
                        ),
                    ], style={"display": "flex", "gap": "0.5rem", "marginBottom": "0.5rem"}),
                    html.Div([
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
                                "width": "100%",
                                "minWidth": "0",
                                "flex": "1 1 0",
                                "boxSizing": "border-box",
                                "transition": "background-color 0.2s",
                            },
                        ),
                        html.Button(
                            "Neighbors",
                            id="neighbors-mode-btn",
                            style={
                                "backgroundColor": "#007bff",
                                "color": "white",
                                "border": "none",
                                "padding": "0.5rem 1rem",
                                "borderRadius": "6px",
                                "cursor": "pointer",
                                "width": "100%",
                                "minWidth": "0",
                                "flex": "1 1 0",
                                "boxSizing": "border-box",
                                "transition": "background-color 0.2s",
                            },
                        ),
                    ], style={"display": "flex", "gap": "0.5rem"}),
                ],
                style={"display": "flex", "flexDirection": "column", "gap": "0.5rem", "marginBottom": "1rem"},
            ),
            html.P(id="mode-instructions", children="Select up to 5 points."),
            html.Div(
                id="interpolate-controls",
                style={"display": "none"},
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
                            html.Label("Interpolation steps (t):"),
                            dcc.Input(
                                id="interpolation-slider",
                                type="number",
                                min=1,
                                step=1,
                                value=5,
                                debounce=False,
                                style={"width": "100px"},
                            ),
                        ],
                        style={"marginTop": "1rem"},
                    ),
                    html.Button(
                        "Clear Path",
                        id="clear-path-btn",
                        style={
                            "backgroundColor": "#dc3545",
                            "color": "white",
                            "border": "none",
                            "padding": "0.5rem 1rem",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "marginTop": "0.5rem",
                            "width": "100%",
                        },
                    ),
                ],
            ),
            html.Div(
                id="neighbors-controls",
                style={"display": "none"},
                children=[
                    html.Label("Number of neighbors (k):"),
                    dcc.Slider(
                        id="neighbors-slider",
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in range(1, 11)},
                        tooltip={"placement": "bottom", "always_visible": True},
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
            "flexShrink": 0,
        },
    )

def _centre_panel() -> html.Div:
    return html.Div(
        [
            html.Div(
                id="single-plot-container",
                children=[
                    dcc.Graph(
                        id="scatter-disk",
                        figure=None,  # Will be set by callback
                        style={"width": "100%", "height": "100%"},
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            "modeBarButtonsToAdd": [],
                            "showTips": True,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "scatter_plot",
                                "height": 600,
                                "width": 800,
                                "scale": 2
                            },
                            "modeBarButtons": [
                                ["pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                                ["toImage"]
                            ]
                        },
                    ),
                ],
                style={
                    "width": "100%",
                    "height": "100%",
                    "maxWidth": "800px",
                    "maxHeight": "800px",
                    "aspectRatio": "1 / 1",
                    "margin": "auto",
                },
            ),
            html.Div(
                id="comparison-plot-container",
                children=[
                    html.Div([
                        html.H5("HoroPCA", style={"textAlign": "center", "margin": "0 0 1rem 0", "color": "#333"}),
                        dcc.Graph(
                            id="scatter-disk-1",
                            figure=None,
                            style={"width": "700px", "height": "1050px"},
                            config={
                                "displayModeBar": True,
                                "displaylogo": False,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                "modeBarButtonsToAdd": [],
                                "showTips": True,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "horopca_plot",
                                    "height": 600,
                                    "width": 800,
                                    "scale": 2
                                },
                                "modeBarButtons": [
                                    ["pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                                    ["toImage"]
                                ]
                            },
                        ),
                    ], style={"flex": "1", "display": "flex", "flexDirection": "column", "minWidth": "0", "overflow": "visible"}),
                    html.Div([
                        html.H5("CO-SNE", style={"textAlign": "center", "margin": "0 0 1rem 0", "color": "#333"}),
                        dcc.Graph(
                            id="scatter-disk-2",
                            figure=None,
                            style={"width": "700px", "height": "1050px"},
                            config={
                                "displayModeBar": True,
                                "displaylogo": False,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                "modeBarButtonsToAdd": [],
                                "showTips": True,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "cosne_plot",
                                    "height": 600,
                                    "width": 800,
                                    "scale": 2
                                },
                                "modeBarButtons": [
                                    ["pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                                    ["toImage"]
                                ]
                            },
                        ),
                    ], style={"flex": "1", "display": "flex", "flexDirection": "column"}),
                ],
                style={
                    "display": "none",
                    "width": "100%",
                    "height": "100%",
                    "maxWidth": "1400px",
                    "maxHeight": "1200px",
                    "margin": "auto",
                    "gap": "1rem",
                    "flexDirection": "row",
                    "overflow": "visible",
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
            "justifyContent": "center",
            "alignItems": "center",
            "minHeight": 0,
            "overflow": "visible",
        },
    )

def _tree_node(title: str, content: html.Div, is_current: bool = False) -> html.Div:
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
            html.Div(id="cmp-header"),
            html.Div(id="cmp-instructions"),
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
                            html.Div(
                                [
                                    html.Div(id="tree-levels-above"),
                                    html.Div(
                                        style={
                                            "height": "1rem",
                                            "width": "2px",
                                            "backgroundColor": "#007bff",
                                            "margin": "0 auto",
                                            "position": "relative",
                                        }
                                    ),
                                    html.Div(id="tree-selected-level"),
                                    html.Div(
                                        style={
                                            "height": "1rem",
                                            "width": "2px",
                                            "backgroundColor": "#007bff",
                                            "margin": "0 auto",
                                            "position": "relative",
                                        }
                                    ),
                                    html.Div(id="tree-levels-below"),
                                ],
                                id="tree-traversal",
                            ),
                        ],
                    ),
                ],
                id="tree-traversal-section",
                style={"display": "none"},
            ),
            html.Div(id="cmp")
        ],
        style={
            "width": "320px",
            "padding": "1rem",
            "backgroundColor": "white",
            "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            "flexShrink": 0,
            "overflowY": "auto",
        },
    )

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
            dcc.Store(id="data-store"),
            dcc.Store(id="labels-store"),
            dcc.Store(id="feature-names-store"),
            dcc.Store(id="target-names-store"),
            dcc.Store(id="images-store"),
            dcc.Store(id="meta-store"),
            dcc.Store(id="points-store"),
            dcc.Store(id="emb"),
            dcc.Store(id="sel", data=[]),
            dcc.Store(id="mode", data="compare"),
            dcc.Store(id="interpolated-point"),
            dcc.Store(id="comparison-mode", data=False),
            html.Div(
                [_config_panel(), _centre_panel(), _cmp_panel()],
                style={
                    "display": "flex",
                    "flex": 1,
                    "minHeight": 0,
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
            "overflow": "hidden",
        },
    ) 