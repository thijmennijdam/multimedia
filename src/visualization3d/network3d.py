"""
3D Hyperbolic Network visualization.
Shows hierarchical networks in hyperbolic space expanding from central abstract pole.
"""
import plotly.graph_objects as go
import numpy as np
from ..utils.style_config import COLORS, FIGURE_STYLES

class Network3DVisualization:
    def __init__(self, disk, hierarchy_builder):
        """
        Initialize the 3D hyperbolic network visualization.
        
        Args:
            disk: PoincareDisk instance
            hierarchy_builder: HierarchyBuilder instance for connections
        """
        self.disk = disk
        self.hierarchy_builder = hierarchy_builder

    def create_figure(self, zoom_level=1, point_size=5, view_mode='points'):
        """
        Create a 3D hyperbolic network figure.
        
        Args:
            zoom_level: Zoom level for the visualization
            point_size: Size of nodes in the visualization
            view_mode: View mode ('points', 'clusters', or 'hierarchy')
            
        Returns:
            plotly.graph_objects.Figure: The created 3D figure
        """
        fig = go.Figure()

        # Add the hyperbolic surface framework
        self._add_hyperbolic_framework(fig, zoom_level)
        
        # Add network nodes and connections
        self._add_hyperbolic_network(fig, point_size, view_mode, zoom_level)

        # Set up 3D layout for hyperbolic network
        max_range = 3.0 / zoom_level
        layout = FIGURE_STYLES['layout'].copy()
        layout.update({
            'scene': {
                'xaxis': {
                    'title': 'X (Conceptual Direction)',
                    'range': [-max_range, max_range],
                    'showgrid': True,
                    'gridcolor': COLORS['border'],
                    'backgroundcolor': 'rgba(0,0,0,0)'
                },
                'yaxis': {
                    'title': 'Y (Conceptual Direction)',
                    'range': [-max_range, max_range],
                    'showgrid': True,
                    'gridcolor': COLORS['border'],
                    'backgroundcolor': 'rgba(0,0,0,0)'
                },
                'zaxis': {
                    'title': 'Z (Abstraction → Specificity)',
                    'range': [0, max_range],
                    'showgrid': True,
                    'gridcolor': COLORS['border'],
                    'backgroundcolor': 'rgba(0,0,0,0)'
                },
                'bgcolor': 'rgba(0,0,0,0)',
                'camera': {
                    'eye': {'x': 2.5, 'y': 2.5, 'z': 1.8}
                },
                'aspectmode': 'manual',
                'aspectratio': dict(x=1, y=1, z=0.7)
            },
            'showlegend': False,
            'margin': dict(l=0, r=0, t=0, b=0)
        })

        fig.update_layout(layout)
        return fig

    def _add_hyperbolic_framework(self, fig, zoom_level):
        """Add minimal hyperbolic surface framework for context."""
        # Light hyperbolic surface outline
        n_points = 20
        max_radius = 2.5 / zoom_level
        k = 0.3
        
        # Add just the boundary circles for context
        for radius in [0.8 * max_radius, max_radius]:
            theta = np.linspace(0, 2*np.pi, 50)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = k * (x**2 + y**2)
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=COLORS['border'], width=1, dash='dot'),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Central pole
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(
                size=8,
                color=COLORS['primary'],
                symbol='diamond',
                line=dict(width=1, color='white')
            ),
            text=['Abstract Core'],
            hovertemplate='<b>%{text}</b><br>Most abstract concept<extra></extra>',
            showlegend=False
        ))

    def _add_hyperbolic_network(self, fig, point_size, view_mode, zoom_level):
        """Add 3D hyperbolic network with nodes and hierarchical edges."""
        points = self.disk.get_points()
        if not points:
            return

        # Create 3D positions in hyperbolic space
        positions_3d = self._create_hyperbolic_positions(points, zoom_level)
        
        # Add hierarchical connections first (so they appear behind nodes)
        self._add_hierarchical_connections(fig, positions_3d)
        
        # Add cluster connections if in cluster view
        if view_mode == 'clusters':
            self._add_cluster_connections(fig, positions_3d)
        
        # Add nodes last (so they appear on top)
        self._add_hyperbolic_nodes(fig, positions_3d, point_size, view_mode)

    def _create_hyperbolic_positions(self, points, zoom_level):
        """Create 3D positions in hyperbolic space from 2D disk points."""
        positions = []
        k = 0.3  # Curvature parameter
        scale_factor = 2.0 / zoom_level
        
        for i, point in enumerate(points):
            x_2d, y_2d = point['x'], point['y']
            
            # Map from 2D disk to hyperbolic 3D space
            x_3d = x_2d * scale_factor
            y_3d = y_2d * scale_factor
            
            # Height based on distance from center (abstraction level)
            base_height = k * (x_3d**2 + y_3d**2)
            
            # Add small variation to prevent overlap and create network depth
            height_variation = np.random.normal(0, 0.1)
            z_3d = base_height + height_variation
            
            # Ensure minimum height
            z_3d = max(z_3d, 0.05)
            
            pos_3d = {
                'x': x_3d,
                'y': y_3d,
                'z': z_3d,
                'label': point.get('label', f'Node {i}'),
                'original': point,
                'abstraction_level': base_height
            }
            positions.append(pos_3d)
        
        return positions

    def _add_hierarchical_connections(self, fig, positions):
        """Add hierarchical connections between related concepts."""
        # Connect nodes that are close in hyperbolic distance
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions[i+1:], i+1):
                # Calculate hyperbolic distance
                dist = self.disk.hyperbolic_distance(
                    (pos1['original']['x'], pos1['original']['y']),
                    (pos2['original']['x'], pos2['original']['y'])
                )
                
                # Connect concepts with moderate hyperbolic distance
                if 0.3 < dist < 1.2:
                    # Connection strength based on abstraction level difference
                    height_diff = abs(pos1['z'] - pos2['z'])
                    opacity = max(0.2, 1.0 - height_diff)
                    
                    # Line color based on abstraction levels
                    if min(pos1['z'], pos2['z']) < 0.3:  # Connection to abstract concepts
                        line_color = COLORS['primary']
                        width = 3
                    else:  # Connection between specific concepts
                        line_color = COLORS['accent']
                        width = 2
                    
                    fig.add_trace(go.Scatter3d(
                        x=[pos1['x'], pos2['x']],
                        y=[pos1['y'], pos2['y']],
                        z=[pos1['z'], pos2['z']],
                        mode='lines',
                        line=dict(
                            color=line_color,
                            width=width
                        ),
                        opacity=opacity,
                        showlegend=False,
                        hoverinfo='skip'
                    ))

    def _add_cluster_connections(self, fig, positions):
        """Add cluster-based connections for cluster view."""
        # Group positions by similar abstraction levels
        abstraction_groups = {}
        for pos in positions:
            level = round(pos['abstraction_level'] * 3) / 3  # Group into thirds
            if level not in abstraction_groups:
                abstraction_groups[level] = []
            abstraction_groups[level].append(pos)
        
        # Connect nodes within the same abstraction level
        for level, group in abstraction_groups.items():
            if len(group) > 1:
                for i, pos1 in enumerate(group):
                    for pos2 in group[i+1:]:
                        # Add dotted connections within clusters
                        fig.add_trace(go.Scatter3d(
                            x=[pos1['x'], pos2['x']],
                            y=[pos1['y'], pos2['y']],
                            z=[pos1['z'], pos2['z']],
                            mode='lines',
                            line=dict(
                                color=COLORS['success'],
                                width=1,
                                dash='dot'
                            ),
                            opacity=0.4,
                            showlegend=False,
                            hoverinfo='skip'
                        ))

    def _add_hyperbolic_nodes(self, fig, positions, point_size, view_mode):
        """Add nodes to the hyperbolic network."""
        x_coords = [pos['x'] for pos in positions]
        y_coords = [pos['y'] for pos in positions]
        z_coords = [pos['z'] for pos in positions]
        labels = [pos['label'] for pos in positions]
        
        # Color and size nodes based on abstraction level
        colors = []
        sizes = []
        
        for pos in positions:
            # Color based on abstraction level
            if pos['abstraction_level'] < 0.2:  # Very abstract
                colors.append(COLORS['primary'])
                sizes.append(point_size + 4)
            elif pos['abstraction_level'] < 0.8:  # Moderately abstract
                colors.append(COLORS['accent'])
                sizes.append(point_size + 2)
            else:  # Very specific
                colors.append(COLORS['success'])
                sizes.append(point_size)
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1, color='white'),
                opacity=0.9
            ),
            text=labels,
            hovertemplate='<b>%{text}</b><br>' +
                         'Conceptual Position: (%{x:.2f}, %{y:.2f})<br>' +
                         'Abstraction Level: %{z:.2f}<extra></extra>',
            showlegend=False
        )) 