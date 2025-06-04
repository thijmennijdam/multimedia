"""
Visualization module for distance calculations.
"""
import plotly.graph_objects as go
from ..utils.style_config import COLORS, FIGURE_STYLES

class DistanceVisualization:
    def __init__(self, calculator):
        """
        Initialize the distance visualization.
        
        Args:
            calculator: DistanceCalculator instance to visualize
        """
        self.calculator = calculator

    def create_figure(self, zoom_level=1, point_size=5, view_mode='points'):
        """
        Create a figure of the distance map.
        
        Args:
            zoom_level: Zoom level for the visualization
            point_size: Size of points in the visualization
            view_mode: View mode ('points', 'clusters', or 'hierarchy')
            
        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        fig = go.Figure()
        
        # Calculate and add the distance map
        X, Y, Z = self.calculator.calculate_distance_map()
        fig.add_trace(go.Heatmap(
            z=Z,
            colorscale='Viridis',
            showscale=True,
            hovertemplate='Distance: %{z:.2f}<extra></extra>'
        ))

        # Add points if in points mode
        if view_mode == 'points':
            self._add_points(fig, point_size)

        layout = FIGURE_STYLES['layout'].copy()
        layout.update({
            'xaxis': {
                'title': 'x',
                'range': [-1/zoom_level, 1/zoom_level],
                'showgrid': False
            },
            'yaxis': {
                'title': 'y',
                'range': [-1/zoom_level, 1/zoom_level],
                'scaleanchor': 'x',
                'scaleratio': 1,
                'showgrid': False
            },
            'margin': dict(l=40, r=60, t=40, b=40)
        })
        
        fig.update_layout(layout)
        return fig

    def _add_points(self, fig, point_size):
        """Add points to the figure."""
        points = self.calculator.disk.get_points()
        if points:
            x = [p['x'] for p in points]
            y = [p['y'] for p in points]
            labels = [p.get('label', f"Point {i}") for i, p in enumerate(points)]
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                name='Points',
                marker=dict(
                    size=point_size,
                    color='white',
                    line=dict(width=2, color=COLORS['accent'])
                ),
                text=labels,
                hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
                showlegend=False
            )) 