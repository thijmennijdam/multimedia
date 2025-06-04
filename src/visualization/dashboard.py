"""
Dashboard component for the Hyperbolic Learning Dashboard.
Modern shadcn-inspired design with configurable visualizations.
"""
import dash
from dash import html, dcc, Input, Output, State, callback
import numpy as np
from ..poincare import PoincareDisk, PoincareVisualization
from ..distance import DistanceCalculator, DistanceVisualization
from ..hierarchy import HierarchyBuilder, HierarchyVisualization
from ..visualization3d import Poincare3DVisualization, Network3DVisualization
from ..utils.style_config import COLORS, LAYOUT_STYLES, FIGURE_STYLES, create_styled_div

class Dashboard:
    def __init__(self):
        """Initialize the dashboard."""
        self.app = dash.Dash(__name__)
        
        # Initialize core components
        self.disk = PoincareDisk()
        self.distance_calculator = DistanceCalculator(self.disk)
        self.hierarchy_builder = HierarchyBuilder(self.disk)
        
        # Initialize all visualizations
        self.poincare_viz = PoincareVisualization(self.disk)
        self.distance_viz = DistanceVisualization(self.distance_calculator)
        self.hierarchy_viz = HierarchyVisualization(self.hierarchy_builder)
        self.poincare_3d_viz = Poincare3DVisualization(self.disk)
        self.network_3d_viz = Network3DVisualization(self.disk, self.hierarchy_builder)
        
        # Define available visualizations
        self.available_visualizations = {
            'poincare-disk': {
                'name': 'Poincaré Disk',
                'viz': self.poincare_viz,
                'icon': '🔵'
            },
            'distance-map': {
                'name': 'Distance Map',
                'viz': self.distance_viz,
                'icon': '🗺️'
            },
            'hierarchy-view': {
                'name': 'Hierarchy View',
                'viz': self.hierarchy_viz,
                'icon': '🌳'
            },
            'statistics': {
                'name': 'Statistics',
                'viz': None,  # Special case
                'icon': '📊'
            },
            'poincare-3d': {
                'name': '3D Poincaré Ball',
                'viz': self.poincare_3d_viz,
                'icon': '🌐'
            },
            'network-3d': {
                'name': '3D Network',
                'viz': self.network_3d_viz,
                'icon': '🕸️'
            }
        }
        
        # Default layout configuration
        self.default_layout = [
            'poincare-disk',
            'distance-map', 
            'hierarchy-view',
            'statistics'
        ]
        
        # Add sample data
        self._add_sample_data()
        
        self.setup_layout()
        self.setup_callbacks()

    def _add_sample_data(self):
        """Add some sample points to the disk for demonstration."""
        sample_points = [
            (0.1, 0.1, "Center"),
            (0.3, 0.4, "Northeast"),
            (0.25, 0.45, "Northeast-2"),
            (-0.4, 0.2, "Northwest"),
            (-0.35, 0.15, "Northwest-2"),
            (0.2, -0.3, "Southeast"),
            (-0.2, -0.4, "Southwest"),
            (0.0, 0.6, "North"),
            (0.7, 0.0, "East"),
            (-0.6, 0.0, "West"),
        ]
        
        for x, y, label in sample_points:
            try:
                self.disk.add_point(x, y, label)
            except ValueError:
                pass

    def setup_layout(self):
        """Set up the modern configurable dashboard layout."""
        self.app.layout = create_styled_div([
            # Modern Navbar
            create_styled_div([
                html.H1("Hyperbolic Learning Dashboard", style=LAYOUT_STYLES['navbar_title']),
                create_styled_div([
                    html.Button(
                        "🔧 Configure",
                        id="config-modal-open",
                        style=LAYOUT_STYLES['sidebar_toggle']
                    ),
                    html.Button(
                        "⚙️ Controls",
                        id="sidebar-toggle",
                        style=LAYOUT_STYLES['sidebar_toggle']
                    )
                ], style=LAYOUT_STYLES['navbar_actions'])
            ], style_key='navbar'),
            
            # Configuration Modal
            self._create_config_modal(),
            
            # 3D Poincaré Explorer Modal
            self._create_3d_explorer_modal(),
            
            # Main Content Area
            create_styled_div([
                # Visualization Grid
                create_styled_div([
                    # Quadrant 1 (Top Left)
                    create_styled_div([
                        create_styled_div([
                            html.H3(id="card-title-1", children="Poincaré Disk", style=LAYOUT_STYLES['card_title']),
                            create_styled_div([
                                html.Button("🔍+", id="zoom-in-1", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'cursor': 'pointer'}),
                                html.Button("🔍-", id="zoom-out-1", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'cursor': 'pointer'}),
                                html.Button("⛶", id="fullscreen-1", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['primary'], 'color': 'white', 'cursor': 'pointer'})
                            ], style={'display': 'flex', 'gap': '4px'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', **LAYOUT_STYLES['card_header']}),
                        create_styled_div([
                            dcc.Graph(
                                id='visualization-1',
                                figure=self.poincare_viz.create_figure(),
                                style={'height': '100%', 'width': '100%'},
                                config={'displayModeBar': False}
                            )
                        ], style_key='card_content')
                    ], style_key='card'),
                    
                    # Quadrant 2 (Top Right)
                    create_styled_div([
                        create_styled_div([
                            html.H3(id="card-title-2", children="Distance Map", style=LAYOUT_STYLES['card_title']),
                            create_styled_div([
                                html.Button("🔍+", id="zoom-in-2", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'cursor': 'pointer'}),
                                html.Button("🔍-", id="zoom-out-2", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'cursor': 'pointer'}),
                                html.Button("⛶", id="fullscreen-2", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['primary'], 'color': 'white', 'cursor': 'pointer'})
                            ], style={'display': 'flex', 'gap': '4px'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', **LAYOUT_STYLES['card_header']}),
                        create_styled_div([
                            dcc.Graph(
                                id='visualization-2',
                                figure=self.distance_viz.create_figure(),
                                style={'height': '100%', 'width': '100%'},
                                config={'displayModeBar': False}
                            )
                        ], style_key='card_content')
                    ], style_key='card'),
                    
                    # Quadrant 3 (Bottom Left)
                    create_styled_div([
                        create_styled_div([
                            html.H3(id="card-title-3", children="Hierarchy View", style=LAYOUT_STYLES['card_title']),
                            create_styled_div([
                                html.Button("🔍+", id="zoom-in-3", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'cursor': 'pointer'}),
                                html.Button("🔍-", id="zoom-out-3", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'cursor': 'pointer'}),
                                html.Button("⛶", id="fullscreen-3", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['primary'], 'color': 'white', 'cursor': 'pointer'})
                            ], style={'display': 'flex', 'gap': '4px'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', **LAYOUT_STYLES['card_header']}),
                        create_styled_div([
                            dcc.Graph(
                                id='visualization-3',
                                figure=self.hierarchy_viz.create_figure(),
                                style={'height': '100%', 'width': '100%'},
                                config={'displayModeBar': False}
                            )
                        ], style_key='card_content')
                    ], style_key='card'),
                    
                    # Quadrant 4 (Bottom Right)
                    create_styled_div([
                        create_styled_div([
                            html.H3(id="card-title-4", children="Statistics", style=LAYOUT_STYLES['card_title']),
                            create_styled_div([
                                html.Button("🔍+", id="zoom-in-4", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'cursor': 'pointer'}),
                                html.Button("🔍-", id="zoom-out-4", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'cursor': 'pointer'}),
                                html.Button("⛶", id="fullscreen-4", style={'margin': '0 2px', 'fontSize': '12px', 'padding': '4px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['primary'], 'color': 'white', 'cursor': 'pointer'})
                            ], style={'display': 'flex', 'gap': '4px'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', **LAYOUT_STYLES['card_header']}),
                        create_styled_div([
                            dcc.Graph(
                                id='visualization-4',
                                figure=self.create_statistics_figure(),
                                style={'height': '100%', 'width': '100%'},
                                config={'displayModeBar': False}
                            )
                        ], style_key='card_content')
                    ], style_key='card'),
                ], style_key='visualization_grid'),
                
                # Collapsible Sidebar
                self._create_sidebar()
            ], style_key='main_content'),
            
            # Stores for state management
            dcc.Store(id='sidebar-state', data={'open': False}),
            dcc.Store(id='layout-config', data=self.default_layout),
            dcc.Store(id='config-modal-state', data={'open': False}),
            dcc.Store(id='3d-explorer-state', data={'open': False}),
            dcc.Store(id='selected-points', data=[]),
            dcc.Store(id='quadrant-zoom-levels', data={'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0})
        ], style_key='app_container')

    def _create_config_modal(self):
        """Create the configuration modal."""
        return create_styled_div([
            create_styled_div([
                create_styled_div([
                    html.H2("Configure Dashboard Layout", style={'margin': '0 0 20px 0', 'fontSize': '24px', 'fontWeight': '600'}),
                    html.P("Select visualizations for each quadrant:", style={'margin': '0 0 20px 0', 'color': COLORS['foreground_muted']}),
                    
                    # Grid configuration
                    create_styled_div([
                        # Quadrant selectors
                        create_styled_div([
                            html.H4("Top Left", style={'margin': '0 0 10px 0', 'fontSize': '16px'}),
                            dcc.Dropdown(
                                id='quadrant-1-selector',
                                options=[
                                    {'label': f"{viz['icon']} {viz['name']}", 'value': key}
                                    for key, viz in self.available_visualizations.items()
                                ],
                                value='poincare-disk',
                                style={'marginBottom': '20px'}
                            )
                        ]),
                        create_styled_div([
                            html.H4("Top Right", style={'margin': '0 0 10px 0', 'fontSize': '16px'}),
                            dcc.Dropdown(
                                id='quadrant-2-selector',
                                options=[
                                    {'label': f"{viz['icon']} {viz['name']}", 'value': key}
                                    for key, viz in self.available_visualizations.items()
                                ],
                                value='distance-map',
                                style={'marginBottom': '20px'}
                            )
                        ]),
                        create_styled_div([
                            html.H4("Bottom Left", style={'margin': '0 0 10px 0', 'fontSize': '16px'}),
                            dcc.Dropdown(
                                id='quadrant-3-selector',
                                options=[
                                    {'label': f"{viz['icon']} {viz['name']}", 'value': key}
                                    for key, viz in self.available_visualizations.items()
                                ],
                                value='hierarchy-view',
                                style={'marginBottom': '20px'}
                            )
                        ]),
                        create_styled_div([
                            html.H4("Bottom Right", style={'margin': '0 0 10px 0', 'fontSize': '16px'}),
                            dcc.Dropdown(
                                id='quadrant-4-selector',
                                options=[
                                    {'label': f"{viz['icon']} {viz['name']}", 'value': key}
                                    for key, viz in self.available_visualizations.items()
                                ],
                                value='statistics',
                                style={'marginBottom': '20px'}
                            )
                        ])
                    ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '30px'}),
                    
                    # Action buttons
                    create_styled_div([
                        html.Button(
                            "Apply Changes",
                            id="config-apply",
                            style=LAYOUT_STYLES['button_primary']
                        ),
                        html.Button(
                            "Cancel",
                            id="config-cancel",
                            style=LAYOUT_STYLES['button_secondary']
                        )
                    ], style={'display': 'flex', 'gap': '10px', 'justifyContent': 'flex-end'})
                ], style={
                    'backgroundColor': COLORS['background'],
                    'padding': '40px',
                    'borderRadius': '12px',
                    'maxWidth': '600px',
                    'width': '90vw',
                    'maxHeight': '80vh',
                    'overflowY': 'auto',
                    'boxShadow': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
                })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'width': '100%',
                'height': '100%'
            })
        ], id='config-modal')

    def _create_3d_explorer_modal(self):
        """Create the 3D Poincaré Explorer modal with dual panels."""
        return create_styled_div([
            create_styled_div([
                # Header
                create_styled_div([
                    html.H2("🌐 3D Poincaré Explorer", style={
                        'margin': '0 0 20px 0', 
                        'fontSize': '28px', 
                        'fontWeight': '600',
                        'color': COLORS['primary']
                    }),
                    html.Button(
                        "✕",
                        id="3d-explorer-close",
                        style={
                            'position': 'absolute',
                            'top': '20px',
                            'right': '20px',
                            'fontSize': '18px',
                            'border': 'none',
                            'backgroundColor': 'transparent',
                            'color': COLORS['foreground_muted'],
                            'cursor': 'pointer'
                        }
                    )
                ], style={'position': 'relative', 'borderBottom': f'1px solid {COLORS["border"]}', 'paddingBottom': '20px', 'marginBottom': '20px'}),
                
                # Main Content - Two Panel Layout
                create_styled_div([
                    # Left Panel - 3D Visualization
                    create_styled_div([
                        create_styled_div([
                            html.H3("Hyperbolic Space", style={'fontSize': '18px', 'fontWeight': '600', 'marginBottom': '15px'}),
                            create_styled_div([
                                html.Button("🔍+", id="3d-zoom-in", style=LAYOUT_STYLES['button_secondary']),
                                html.Button("🔍-", id="3d-zoom-out", style=LAYOUT_STYLES['button_secondary']),
                                html.Button("🔄", id="3d-reset-view", style=LAYOUT_STYLES['button_secondary']),
                                html.Button("📸", id="3d-screenshot", style=LAYOUT_STYLES['button_secondary'])
                            ], style={'display': 'flex', 'gap': '8px', 'marginBottom': '15px'})
                        ]),
                        dcc.Graph(
                            id='3d-explorer-graph',
                            style={'height': '600px', 'width': '100%'},
                            config={
                                'displayModeBar': True,
                                'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                                'modeBarButtonsToRemove': ['autoScale2d']
                            }
                        )
                    ], style={
                        'width': '65%',
                        'padding': '20px',
                        'backgroundColor': COLORS['background'],
                        'borderRadius': '8px',
                        'border': f'1px solid {COLORS["border"]}'
                    }),
                    
                    # Right Panel - Information and Controls
                    create_styled_div([
                        # Point Information Section
                        create_styled_div([
                            html.H3("📍 Point Information", style={'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '15px'}),
                            html.Div(id="selected-point-info", children=[
                                html.P("Click on a point in the 3D space to see its details", 
                                      style={'color': COLORS['foreground_muted'], 'fontStyle': 'italic'})
                            ])
                        ], style={
                            'marginBottom': '25px',
                            'padding': '15px',
                            'backgroundColor': COLORS['background_secondary'],
                            'borderRadius': '6px',
                            'border': f'1px solid {COLORS["border"]}'
                        }),
                        
                        # Comparison Section
                        create_styled_div([
                            html.H3("⚖️ Point Comparison", style={'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '15px'}),
                            html.Div(id="comparison-info", children=[
                                html.P("Select two points to compare their relationship", 
                                      style={'color': COLORS['foreground_muted'], 'fontStyle': 'italic'})
                            ]),
                            html.Button(
                                "Clear Selection",
                                id="clear-selection",
                                style=LAYOUT_STYLES['button_secondary']
                            )
                        ], style={
                            'marginBottom': '25px',
                            'padding': '15px',
                            'backgroundColor': COLORS['background_secondary'],
                            'borderRadius': '6px',
                            'border': f'1px solid {COLORS["border"]}'
                        }),
                        
                        # Controls Section
                        create_styled_div([
                            html.H3("🎛️ Controls", style={'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '15px'}),
                            
                            create_styled_div([
                                html.Label("Zoom Level", style={'fontSize': '14px', 'fontWeight': '500', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Slider(
                                    id='3d-explorer-zoom',
                                    min=0.5,
                                    max=5.0,
                                    step=0.1,
                                    value=1.0,
                                    marks={i: {'label': str(i), 'style': {'fontSize': '10px'}} for i in [0.5, 1, 2, 3, 4, 5]},
                                    tooltip={"placement": "bottom", "always_visible": False}
                                ),
                            ], style={'marginBottom': '20px'}),
                            
                            create_styled_div([
                                html.Label("Point Size", style={'fontSize': '14px', 'fontWeight': '500', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Slider(
                                    id='3d-explorer-point-size',
                                    min=4,
                                    max=20,
                                    step=1,
                                    value=8,
                                    marks={i: {'label': str(i), 'style': {'fontSize': '10px'}} for i in [4, 8, 12, 16, 20]},
                                    tooltip={"placement": "bottom", "always_visible": False}
                                ),
                            ], style={'marginBottom': '20px'}),
                            
                            create_styled_div([
                                html.Label("View Mode", style={'fontSize': '14px', 'fontWeight': '500', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.RadioItems(
                                    id='3d-explorer-view-mode',
                                    options=[
                                        {'label': 'Points Only', 'value': 'points'},
                                        {'label': 'With Clusters', 'value': 'clusters'},
                                        {'label': 'Show Hierarchy', 'value': 'hierarchy'}
                                    ],
                                    value='points',
                                    labelStyle={
                                        'display': 'block', 
                                        'marginBottom': '8px',
                                        'fontSize': '13px',
                                        'cursor': 'pointer'
                                    },
                                    inputStyle={'marginRight': '8px'}
                                ),
                            ], style={'marginBottom': '20px'}),
                            
                            html.Button(
                                "🎲 Add Random Point",
                                id="add-point-3d-explorer",
                                style=LAYOUT_STYLES['button_primary']
                            ),
                            html.Div(id="add-point-3d-output", style={'marginTop': '10px', 'fontSize': '12px'})
                        ], style={
                            'padding': '15px',
                            'backgroundColor': COLORS['background_secondary'],
                            'borderRadius': '6px',
                            'border': f'1px solid {COLORS["border"]}'
                        }),
                    ], style={
                        'width': '33%',
                        'padding': '20px',
                        'backgroundColor': COLORS['background'],
                        'borderRadius': '8px',
                        'border': f'1px solid {COLORS["border"]}',
                        'maxHeight': '650px',
                        'overflowY': 'auto'
                    })
                ], style={
                    'display': 'flex',
                    'gap': '20px',
                    'height': '100%'
                })
            ], style={
                'backgroundColor': COLORS['background'],
                'padding': '30px',
                'borderRadius': '12px',
                'width': '95vw',
                'height': '90vh',
                'maxWidth': '1400px',
                'boxShadow': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
                'overflow': 'hidden'
            })
        ], id='3d-explorer-modal', style={
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'width': '100vw',
            'height': '100vh',
            'backgroundColor': 'rgba(0, 0, 0, 0.7)',
            'display': 'none',
            'alignItems': 'center',
            'justifyContent': 'center',
            'zIndex': '1000'
        })

    def _create_sidebar(self):
        """Create the collapsible sidebar."""
        return create_styled_div([
            html.H3("Controls", style={'fontSize': '18px', 'fontWeight': '600', 'marginBottom': '20px'}),
            
            create_styled_div([
                html.Label("Zoom Level", style=LAYOUT_STYLES['control_label']),
                dcc.Slider(
                    id='zoom-slider',
                    min=1,
                    max=10,
                    step=0.1,
                    value=1,
                    marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(1, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ], style_key='control_group'),
            
            create_styled_div([
                html.Label("Point Size", style=LAYOUT_STYLES['control_label']),
                dcc.Slider(
                    id='point-size-slider',
                    min=4,
                    max=16,
                    step=1,
                    value=8,
                    marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(4, 17, 4)},
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ], style_key='control_group'),
            
            create_styled_div([
                html.Label("View Mode", style=LAYOUT_STYLES['control_label']),
                dcc.RadioItems(
                    id='view-mode',
                    options=[
                        {'label': 'Points', 'value': 'points'},
                        {'label': 'Clusters', 'value': 'clusters'},
                        {'label': 'Hierarchy', 'value': 'hierarchy'}
                    ],
                    value='points',
                    labelStyle={
                        'display': 'block', 
                        'marginBottom': '8px',
                        'fontSize': '14px',
                        'cursor': 'pointer'
                    },
                    inputStyle={'marginRight': '8px'}
                ),
            ], style_key='control_group'),
            
            create_styled_div([
                html.Label("Hierarchy Threshold", style=LAYOUT_STYLES['control_label']),
                dcc.Slider(
                    id='hierarchy-threshold',
                    min=0.1,
                    max=2.0,
                    step=0.1,
                    value=0.5,
                    marks={0.5: {'label': '0.5', 'style': {'fontSize': '12px'}}, 
                           1.0: {'label': '1.0', 'style': {'fontSize': '12px'}}, 
                           1.5: {'label': '1.5', 'style': {'fontSize': '12px'}}},
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ], style_key='control_group'),
            
            create_styled_div([
                html.Button(
                    "Add Random Point",
                    id="add-point-btn",
                    style=LAYOUT_STYLES['button_primary']
                ),
                html.Div(id="add-point-output", style={'marginTop': '12px', 'fontSize': '14px'})
            ], style_key='control_group'),
        ], id='sidebar', style=LAYOUT_STYLES['sidebar_hidden'])

    def get_visualization_figure(self, viz_type, zoom_level=1, point_size=5, view_mode='points'):
        """Get a figure for the specified visualization type."""
        if viz_type == 'statistics':
            return self.create_statistics_figure()
        elif viz_type in self.available_visualizations:
            viz = self.available_visualizations[viz_type]['viz']
            if viz:
                return viz.create_figure(zoom_level, point_size, view_mode)
        
        # Fallback
        return self.poincare_viz.create_figure(zoom_level, point_size, view_mode)

    def create_statistics_figure(self):
        """Create the statistics visualization."""
        import plotly.graph_objects as go
        
        # Get statistics
        num_points = len(self.disk.get_points())
        
        if not self.hierarchy_builder.get_hierarchy():
            self.hierarchy_builder.build_hierarchy()
        
        hierarchy = self.hierarchy_builder.get_hierarchy()
        num_clusters = len(hierarchy['children']) if hierarchy else 0
        hierarchy_depth = self.hierarchy_builder.get_levels()
        
        # Calculate average distance
        points = self.disk.get_points()
        if len(points) > 1:
            total_distance = 0
            count = 0
            for i, p1 in enumerate(points):
                for p2 in points[i+1:]:
                    dist = self.disk.hyperbolic_distance(
                        (p1['x'], p1['y']),
                        (p2['x'], p2['y'])
                    )
                    total_distance += dist
                    count += 1
            avg_distance = total_distance / count if count > 0 else 0
        else:
            avg_distance = 0
        
        # Create modern bar chart
        categories = ['Points', 'Clusters', 'Depth', 'Avg Distance']
        values = [num_points, num_clusters, hierarchy_depth, round(avg_distance, 2)]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=COLORS['accent'],
            text=values,
            textposition='auto',
            textfont={'color': 'white', 'size': 12},
            hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
        ))

        layout = FIGURE_STYLES['layout'].copy()
        layout.update({
            'xaxis': {'title': None, 'showgrid': False},
            'yaxis': {'title': None, 'showgrid': True, 'gridcolor': COLORS['border']},
            'margin': dict(l=40, r=40, t=20, b=40),
            'showlegend': False
        })
        
        fig.update_layout(layout)
        return fig

    def setup_callbacks(self):
        """Set up the dashboard callbacks."""
        # Sidebar toggle
        @self.app.callback(
            [Output('sidebar', 'style'),
             Output('sidebar-state', 'data')],
            [Input('sidebar-toggle', 'n_clicks')],
            [State('sidebar-state', 'data')]
        )
        def toggle_sidebar(n_clicks, sidebar_state):
            if n_clicks:
                new_state = not sidebar_state['open']
                style = LAYOUT_STYLES['sidebar'] if new_state else LAYOUT_STYLES['sidebar_hidden']
                return style, {'open': new_state}
            return LAYOUT_STYLES['sidebar_hidden'], {'open': False}

        # Config modal toggle
        @self.app.callback(
            Output('config-modal', 'style'),
            [Input('config-modal-open', 'n_clicks'),
             Input('config-cancel', 'n_clicks'),
             Input('config-apply', 'n_clicks')]
        )
        def toggle_config_modal(open_clicks, cancel_clicks, apply_clicks):
            ctx = dash.callback_context
            
            # Base modal style
            modal_style = {
                'position': 'fixed',
                'top': '0',
                'left': '0',
                'width': '100vw',
                'height': '100vh',
                'backgroundColor': 'rgba(0, 0, 0, 0.5)',
                'alignItems': 'center',
                'justifyContent': 'center',
                'zIndex': '100',
                'display': 'none'  # Hidden by default
            }
            
            # If no trigger, return hidden
            if not ctx.triggered:
                return modal_style
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Show modal when configure button is clicked
            if button_id == 'config-modal-open' and open_clicks:
                modal_style['display'] = 'flex'
            
            return modal_style

        # Update layout configuration
        @self.app.callback(
            Output('layout-config', 'data'),
            [Input('config-apply', 'n_clicks')],
            [State('quadrant-1-selector', 'value'),
             State('quadrant-2-selector', 'value'),
             State('quadrant-3-selector', 'value'),
             State('quadrant-4-selector', 'value')]
        )
        def update_layout_config(apply_clicks, q1, q2, q3, q4):
            if apply_clicks:
                return [q1, q2, q3, q4]
            return dash.no_update

        # Update all visualizations and titles
        @self.app.callback(
            [Output('visualization-1', 'figure'),
             Output('visualization-2', 'figure'),
             Output('visualization-3', 'figure'),
             Output('visualization-4', 'figure'),
             Output('card-title-1', 'children'),
             Output('card-title-2', 'children'),
             Output('card-title-3', 'children'),
             Output('card-title-4', 'children')],
            [Input('zoom-slider', 'value'),
             Input('point-size-slider', 'value'),
             Input('view-mode', 'value'),
             Input('hierarchy-threshold', 'value'),
             Input('layout-config', 'data'),
             Input('quadrant-zoom-levels', 'data')]
        )
        def update_all_visualizations(zoom_level, point_size, view_mode, hierarchy_threshold, layout_config, quadrant_zoom_levels):
            self.hierarchy_builder.build_hierarchy(hierarchy_threshold)
            
            figures = []
            titles = []
            
            for i, viz_type in enumerate(layout_config):
                # Use individual quadrant zoom level, fall back to global zoom
                quadrant_zoom = quadrant_zoom_levels.get(str(i+1), zoom_level)
                figure = self.get_visualization_figure(viz_type, quadrant_zoom, point_size, view_mode)
                figures.append(figure)
                
                viz_info = self.available_visualizations.get(viz_type, {'name': 'Unknown'})
                titles.append(f"{viz_info.get('icon', '')} {viz_info['name']}")
            
            return figures + titles

        # Add random point
        @self.app.callback(
            Output('add-point-output', 'children'),
            [Input('add-point-btn', 'n_clicks')]
        )
        def add_random_point(n_clicks):
            if n_clicks:
                r = np.random.uniform(0, 0.8)
                theta = np.random.uniform(0, 2*np.pi)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                try:
                    self.disk.add_point(x, y, f"Point-{len(self.disk.get_points())+1}")
                    return html.Div(f"✓ Added point at ({x:.2f}, {y:.2f})", 
                                    style={'color': COLORS['success'], 'fontSize': '13px'})
                except ValueError as e:
                    return html.Div(f"✗ Error: {str(e)}", 
                                    style={'color': COLORS['destructive'], 'fontSize': '13px'})
            return ""

        # Quadrant zoom controls
        @self.app.callback(
            Output('quadrant-zoom-levels', 'data'),
            [Input('zoom-in-1', 'n_clicks'),
             Input('zoom-out-1', 'n_clicks'),
             Input('zoom-in-2', 'n_clicks'),
             Input('zoom-out-2', 'n_clicks'),
             Input('zoom-in-3', 'n_clicks'),
             Input('zoom-out-3', 'n_clicks'),
             Input('zoom-in-4', 'n_clicks'),
             Input('zoom-out-4', 'n_clicks')],
            [State('quadrant-zoom-levels', 'data')]
        )
        def update_quadrant_zoom(z1_in, z1_out, z2_in, z2_out, z3_in, z3_out, z4_in, z4_out, zoom_levels):
            ctx = dash.callback_context
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                # Determine which quadrant and action
                if 'zoom-in-1' in button_id and z1_in:
                    zoom_levels['1'] = min(zoom_levels['1'] * 1.2, 5.0)
                elif 'zoom-out-1' in button_id and z1_out:
                    zoom_levels['1'] = max(zoom_levels['1'] / 1.2, 0.2)
                elif 'zoom-in-2' in button_id and z2_in:
                    zoom_levels['2'] = min(zoom_levels['2'] * 1.2, 5.0)
                elif 'zoom-out-2' in button_id and z2_out:
                    zoom_levels['2'] = max(zoom_levels['2'] / 1.2, 0.2)
                elif 'zoom-in-3' in button_id and z3_in:
                    zoom_levels['3'] = min(zoom_levels['3'] * 1.2, 5.0)
                elif 'zoom-out-3' in button_id and z3_out:
                    zoom_levels['3'] = max(zoom_levels['3'] / 1.2, 0.2)
                elif 'zoom-in-4' in button_id and z4_in:
                    zoom_levels['4'] = min(zoom_levels['4'] * 1.2, 5.0)
                elif 'zoom-out-4' in button_id and z4_out:
                    zoom_levels['4'] = max(zoom_levels['4'] / 1.2, 0.2)
            
            return zoom_levels

        # Fullscreen modal toggles
        @self.app.callback(
            Output('3d-explorer-modal', 'style'),
            [Input('fullscreen-1', 'n_clicks'),
             Input('fullscreen-2', 'n_clicks'),
             Input('fullscreen-3', 'n_clicks'),
             Input('fullscreen-4', 'n_clicks'),
             Input('3d-explorer-close', 'n_clicks')],
            [State('layout-config', 'data')]
        )
        def toggle_fullscreen_modal(fs1, fs2, fs3, fs4, close_clicks, layout_config):
            ctx = dash.callback_context
            
            modal_style_hidden = {
                'position': 'fixed',
                'top': '0',
                'left': '0',
                'width': '100vw',
                'height': '100vh',
                'backgroundColor': 'rgba(0, 0, 0, 0.7)',
                'display': 'none',
                'alignItems': 'center',
                'justifyContent': 'center',
                'zIndex': '1000'
            }
            
            modal_style_visible = modal_style_hidden.copy()
            modal_style_visible['display'] = 'flex'
            
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                # Close modal
                if button_id == '3d-explorer-close':
                    return modal_style_hidden
                
                # Check if clicked quadrant has a 3D visualization
                if button_id.startswith('fullscreen-'):
                    quadrant = int(button_id.split('-')[1]) - 1
                    viz_type = layout_config[quadrant]
                    if viz_type in ['poincare-3d', 'network-3d']:
                        return modal_style_visible
            
            return modal_style_hidden

        # 3D Explorer graph update
        @self.app.callback(
            Output('3d-explorer-graph', 'figure'),
            [Input('3d-explorer-zoom', 'value'),
             Input('3d-explorer-point-size', 'value'),
             Input('3d-explorer-view-mode', 'value'),
             Input('add-point-3d-explorer', 'n_clicks'),
             Input('3d-zoom-in', 'n_clicks'),
             Input('3d-zoom-out', 'n_clicks'),
             Input('3d-reset-view', 'n_clicks')]
        )
        def update_3d_explorer_graph(zoom_level, point_size, view_mode, add_point_clicks, 
                                   zoom_in_clicks, zoom_out_clicks, reset_clicks):
            # Use Poincaré 3D visualization as default
            return self.poincare_3d_viz.create_figure(zoom_level, point_size, view_mode)

        # Point selection and information display
        @self.app.callback(
            [Output('selected-point-info', 'children'),
             Output('comparison-info', 'children'),
             Output('selected-points', 'data')],
            [Input('3d-explorer-graph', 'clickData'),
             Input('clear-selection', 'n_clicks')],
            [State('selected-points', 'data')]
        )
        def handle_point_selection(click_data, clear_clicks, selected_points):
            ctx = dash.callback_context
            
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                # Clear selection
                if trigger_id == 'clear-selection':
                    return (
                        [html.P("Click on a point in the 3D space to see its details", 
                               style={'color': COLORS['foreground_muted'], 'fontStyle': 'italic'})],
                        [html.P("Select two points to compare their relationship", 
                               style={'color': COLORS['foreground_muted'], 'fontStyle': 'italic'})],
                        []
                    )
                
                # Point selection
                if trigger_id == '3d-explorer-graph' and click_data:
                    points = click_data.get('points', [])
                    if points:
                        clicked_point = points[0]
                        point_info = {
                            'x': clicked_point.get('x', 0),
                            'y': clicked_point.get('y', 0),
                            'z': clicked_point.get('z', 0),
                            'text': clicked_point.get('text', 'Unknown')
                        }
                        
                        # Add to selected points (max 2 for comparison)
                        if point_info not in selected_points:
                            if len(selected_points) >= 2:
                                selected_points = [selected_points[1], point_info]
                            else:
                                selected_points.append(point_info)
                        
                        # Generate point information display
                        point_info_display = self._create_point_info_display(point_info)
                        
                        # Generate comparison display if we have 2 points
                        comparison_display = self._create_comparison_display(selected_points) if len(selected_points) == 2 else [
                            html.P(f"Selected {len(selected_points)} point(s). Select another to compare.", 
                                  style={'color': COLORS['foreground_muted'], 'fontStyle': 'italic'})
                        ]
                        
                        return point_info_display, comparison_display, selected_points
            
            # Default return
            return (
                [html.P("Click on a point in the 3D space to see its details", 
                       style={'color': COLORS['foreground_muted'], 'fontStyle': 'italic'})],
                [html.P("Select two points to compare their relationship", 
                       style={'color': COLORS['foreground_muted'], 'fontStyle': 'italic'})],
                selected_points
            )

        # Add point in 3D explorer
        @self.app.callback(
            Output('add-point-3d-output', 'children'),
            [Input('add-point-3d-explorer', 'n_clicks')]
        )
        def add_random_point_3d(n_clicks):
            if n_clicks:
                r = np.random.uniform(0, 0.8)
                theta = np.random.uniform(0, 2*np.pi)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                try:
                    self.disk.add_point(x, y, f"3D-Point-{len(self.disk.get_points())+1}")
                    return html.Div(f"✓ Added point at ({x:.2f}, {y:.2f})", 
                                    style={'color': COLORS['success'], 'fontSize': '12px'})
                except ValueError as e:
                    return html.Div(f"✗ Error: {str(e)}", 
                                    style={'color': COLORS['destructive'], 'fontSize': '12px'})
            return ""

    def _create_point_info_display(self, point_info):
        """Create display for selected point information."""
        # Calculate abstraction level based on distance from center
        distance_from_center = np.sqrt(point_info['x']**2 + point_info['y']**2)
        abstraction_level = "High" if distance_from_center < 0.3 else "Medium" if distance_from_center < 0.7 else "Low"
        
        return [
            html.H4(f"📍 {point_info['text']}", style={'margin': '0 0 10px 0', 'fontSize': '14px', 'fontWeight': '600'}),
            html.P(f"Position: ({point_info['x']:.3f}, {point_info['y']:.3f}, {point_info['z']:.3f})", 
                   style={'margin': '5px 0', 'fontSize': '12px'}),
            html.P(f"Distance from center: {distance_from_center:.3f}", 
                   style={'margin': '5px 0', 'fontSize': '12px'}),
            html.P(f"Abstraction level: {abstraction_level}", 
                   style={'margin': '5px 0', 'fontSize': '12px', 'fontWeight': '500'}),
            html.P(f"Height (specificity): {point_info['z']:.3f}", 
                   style={'margin': '5px 0', 'fontSize': '12px'}),
            html.Hr(style={'margin': '10px 0', 'border': f'1px solid {COLORS["border"]}'}),
            html.P("📝 Textual Meaning:", style={'margin': '5px 0', 'fontSize': '12px', 'fontWeight': '500'}),
            html.P(f"This point represents a concept with {abstraction_level.lower()} abstraction level. "
                   f"It is positioned {distance_from_center:.1f} units from the central pole, "
                   f"indicating its conceptual specificity.", 
                   style={'margin': '5px 0', 'fontSize': '11px', 'color': COLORS['foreground_muted'], 'lineHeight': '1.4'})
        ]

    def _create_comparison_display(self, selected_points):
        """Create display for comparing two selected points."""
        if len(selected_points) != 2:
            return [html.P("Select two points to compare", style={'color': COLORS['foreground_muted'], 'fontStyle': 'italic'})]
        
        point1, point2 = selected_points
        
        # Calculate Euclidean distance in 3D
        euclidean_dist = np.sqrt((point1['x'] - point2['x'])**2 + 
                                (point1['y'] - point2['y'])**2 + 
                                (point1['z'] - point2['z'])**2)
        
        # Calculate hyperbolic distance (approximate)
        hyperbolic_dist = self.disk.hyperbolic_distance(
            (point1['x'], point1['y']), 
            (point2['x'], point2['y'])
        )
        
        # Determine relationship
        if hyperbolic_dist < 0.5:
            relationship = "Very similar concepts"
        elif hyperbolic_dist < 1.0:
            relationship = "Related concepts"
        elif hyperbolic_dist < 1.5:
            relationship = "Somewhat related"
        else:
            relationship = "Distant concepts"
        
        return [
            html.H4("⚖️ Comparing Two Points", style={'margin': '0 0 10px 0', 'fontSize': '14px', 'fontWeight': '600'}),
            html.P(f"Point A: {point1['text']}", style={'margin': '5px 0', 'fontSize': '12px', 'fontWeight': '500'}),
            html.P(f"Point B: {point2['text']}", style={'margin': '5px 0', 'fontSize': '12px', 'fontWeight': '500'}),
            html.Hr(style={'margin': '10px 0', 'border': f'1px solid {COLORS["border"]}'}),
            html.P(f"Euclidean distance: {euclidean_dist:.3f}", style={'margin': '5px 0', 'fontSize': '12px'}),
            html.P(f"Hyperbolic distance: {hyperbolic_dist:.3f}", style={'margin': '5px 0', 'fontSize': '12px'}),
            html.P(f"Relationship: {relationship}", style={'margin': '5px 0', 'fontSize': '12px', 'fontWeight': '500', 'color': COLORS['accent']}),
            html.Hr(style={'margin': '10px 0', 'border': f'1px solid {COLORS["border"]}'}),
            html.P("📊 Analysis:", style={'margin': '5px 0', 'fontSize': '12px', 'fontWeight': '500'}),
            html.P(f"These two concepts are positioned at different levels in the hyperbolic space, "
                   f"with a hyperbolic distance of {hyperbolic_dist:.2f}. This suggests they are {relationship.lower()}.",
                   style={'margin': '5px 0', 'fontSize': '11px', 'color': COLORS['foreground_muted'], 'lineHeight': '1.4'})
        ]

    def run(self, debug=True, port=8050):
        """Run the dashboard."""
        self.app.run(debug=debug, port=8050) 