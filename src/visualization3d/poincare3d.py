"""
3D Hyperbolic space visualization.
Represents hyperbolic geometry with negative curvature expanding from a central pole.
"""
import plotly.graph_objects as go
import numpy as np
from ..utils.style_config import COLORS, FIGURE_STYLES

class Poincare3DVisualization:
    def __init__(self, disk):
        """
        Initialize the 3D hyperbolic space visualization.
        
        Args:
            disk: PoincareDisk instance to visualize in 3D hyperbolic space
        """
        self.disk = disk

    def create_figure(self, zoom_level=1, point_size=5, view_mode='points'):
        """
        Create a 3D figure of hyperbolic space with negative curvature.
        
        Args:
            zoom_level: Zoom level for the visualization
            point_size: Size of points in the visualization
            view_mode: View mode ('points', 'clusters', or 'hierarchy')
            
        Returns:
            plotly.graph_objects.Figure: The created 3D figure
        """
        fig = go.Figure()

        # Create the hyperbolic surface (hyperboloid model)
        self._add_hyperbolic_surface(fig, zoom_level)
        
        # Add hyperbolic grid lines
        self._add_hyperbolic_grid(fig, zoom_level)
        
        # Add points based on view mode
        if view_mode == 'points':
            self._add_hyperbolic_points(fig, point_size, zoom_level)
        elif view_mode == 'clusters':
            self._add_hyperbolic_clusters(fig, point_size, zoom_level)
        elif view_mode == 'hierarchy':
            self._add_hyperbolic_hierarchy(fig, point_size, zoom_level)

        # Set up 3D layout for hyperbolic space
        max_range = 3.0 / zoom_level
        layout = FIGURE_STYLES['layout'].copy()
        layout.update({
            'scene': {
                'xaxis': {
                    'title': 'X (Specificity)',
                    'range': [-max_range, max_range],
                    'showgrid': True,
                    'gridcolor': COLORS['border'],
                    'backgroundcolor': 'rgba(0,0,0,0)'
                },
                'yaxis': {
                    'title': 'Y (Specificity)',
                    'range': [-max_range, max_range],
                    'showgrid': True,
                    'gridcolor': COLORS['border'],
                    'backgroundcolor': 'rgba(0,0,0,0)'
                },
                'zaxis': {
                    'title': 'Z (Abstraction Level)',
                    'range': [0, max_range],
                    'showgrid': True,
                    'gridcolor': COLORS['border'],
                    'backgroundcolor': 'rgba(0,0,0,0)'
                },
                'bgcolor': 'rgba(0,0,0,0)',
                'camera': {
                    'eye': {'x': 2, 'y': 2, 'z': 1.5}
                },
                'aspectmode': 'manual',
                'aspectratio': dict(x=1, y=1, z=0.6)
            },
            'showlegend': False,
            'margin': dict(l=0, r=0, t=0, b=0)
        })

        fig.update_layout(layout)
        return fig

    def _add_hyperbolic_surface(self, fig, zoom_level):
        """Add the hyperbolic surface showing negative curvature expanding from central pole."""
        # Create hyperbolic paraboloid surface: z = k * (x² + y²)
        # This represents the expansion from the central pole (most abstract) outward
        n_points = 40
        max_radius = 2.5 / zoom_level
        
        # Create radial and angular coordinates
        r = np.linspace(0, max_radius, n_points)
        theta = np.linspace(0, 2*np.pi, n_points)
        R, Theta = np.meshgrid(r, theta)
        
        # Convert to Cartesian coordinates
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        
        # Hyperbolic surface: height increases with distance from center
        # This represents how specificity increases as you move away from the abstract center
        k = 0.3  # Curvature parameter
        Z = k * (X**2 + Y**2)
        
        # Add the main hyperbolic surface
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.15,
            colorscale=[[0, COLORS['background']], [1, COLORS['accent']]],
            showscale=False,
            hoverinfo='skip',
            name='Hyperbolic Space'
        ))
        
        # Add central pole marker (most abstract concept)
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(
                size=12,
                color=COLORS['primary'],
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            text=['Central Pole<br>(Most Abstract)'],
            hovertemplate='<b>%{text}</b><br>The most abstract concept<extra></extra>',
            showlegend=False
        ))

    def _add_hyperbolic_grid(self, fig, zoom_level):
        """Add grid lines showing hyperbolic geometry."""
        max_radius = 2.5 / zoom_level
        k = 0.3  # Same curvature parameter as surface
        
        # Add concentric circles at different abstraction levels
        for radius in np.linspace(0.5, max_radius, 5):
            theta = np.linspace(0, 2*np.pi, 50)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = k * (x**2 + y**2)
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=COLORS['grid'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add radial lines from center outward
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            r = np.linspace(0, max_radius, 30)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = k * (x**2 + y**2)
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=COLORS['grid'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    def _add_hyperbolic_points(self, fig, point_size, zoom_level):
        """Add points mapped to hyperbolic space from the 2D disk."""
        points = self.disk.get_points()
        if not points:
            return

        # Map 2D disk points to 3D hyperbolic space
        x_3d, y_3d, z_3d = [], [], []
        labels = []
        colors = []
        
        k = 0.3  # Same curvature parameter
        
        for point in points:
            x, y = point['x'], point['y']
            
            # Map from Poincaré disk to hyperbolic space
            # Distance from center determines abstraction level
            r_disk = np.sqrt(x**2 + y**2)
            
            if r_disk < 0.99:  # Avoid boundary
                # Scale the coordinates for hyperbolic space
                # Points closer to center (more abstract) stay lower
                # Points farther from center (more specific) go higher
                scale_factor = 2.0 / zoom_level
                x_hyp = x * scale_factor
                y_hyp = y * scale_factor
                
                # Height based on hyperbolic distance from center
                # More specific concepts are higher up
                z_hyp = k * (x_hyp**2 + y_hyp**2)
                
                x_3d.append(x_hyp)
                y_3d.append(y_hyp)
                z_3d.append(z_hyp)
                labels.append(point.get('label', 'Point'))
                
                # Color based on abstraction level (height)
                # Higher points (more specific) are redder
                # Lower points (more abstract) are bluer
                colors.append(z_hyp)

        if x_3d:
            fig.add_trace(go.Scatter3d(
                x=x_3d, y=y_3d, z=z_3d,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=colors,
                    colorscale='Viridis',
                    line=dict(width=1, color='white'),
                    showscale=False
                ),
                text=labels,
                hovertemplate='<b>%{text}</b><br>Specificity: %{x:.2f}, %{y:.2f}<br>Abstraction Level: %{z:.2f}<extra></extra>',
                showlegend=False
            ))

    def _add_hyperbolic_clusters(self, fig, point_size, zoom_level):
        """Add 3D cluster visualization in hyperbolic space."""
        # Similar to points but with cluster coloring
        self._add_hyperbolic_points(fig, point_size, zoom_level)

    def _add_hyperbolic_hierarchy(self, fig, point_size, zoom_level):
        """Add 3D hierarchy visualization showing concept relationships."""
        points = self.disk.get_points()
        if len(points) < 2:
            return
            
        # Add points first
        self._add_hyperbolic_points(fig, point_size, zoom_level)
        
        k = 0.3
        scale_factor = 2.0 / zoom_level
        
        # Add hierarchy connections
        for i, point1 in enumerate(points):
            for point2 in points[i+1:]:
                # Calculate hyperbolic distance
                dist = self.disk.hyperbolic_distance(
                    (point1['x'], point1['y']),
                    (point2['x'], point2['y'])
                )
                
                # Only connect nearby points (hierarchical relationships)
                if dist < 1.0:
                    # Map both points to 3D
                    x1, y1 = point1['x'] * scale_factor, point1['y'] * scale_factor
                    z1 = k * (x1**2 + y1**2)
                    
                    x2, y2 = point2['x'] * scale_factor, point2['y'] * scale_factor
                    z2 = k * (x2**2 + y2**2)
                    
                    # Add connection line
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2],
                        mode='lines',
                        line=dict(
                            color=COLORS['accent'], 
                            width=3,
                            dash='dot'
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    )) 