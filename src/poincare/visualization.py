"""
Visualization module for the Poincaré disk.
"""
import plotly.graph_objects as go
from ..utils.style_config import COLORS, FIGURE_STYLES

class PoincareVisualization:
    def __init__(self, disk):
        """
        Initialize the Poincaré disk visualization.
        
        Args:
            disk: PoincareDisk instance to visualize
        """
        self.disk = disk

    def create_figure(self, zoom_level=1, point_size=5, view_mode='points'):
        """
        Create a figure of the Poincaré disk.
        
        Args:
            zoom_level: Zoom level for the visualization
            point_size: Size of points in the visualization
            view_mode: View mode ('points', 'clusters', or 'hierarchy')
            
        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        fig = go.Figure()
        
        # Add the disk boundary
        x, y = self.disk.get_boundary_points()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name='Disk Boundary',
            line=dict(color=COLORS['disk_boundary'], width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add grid lines
        for x, y in self.disk.get_grid_points():
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color=COLORS['grid'], width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add points based on view mode
        if view_mode == 'points':
            self._add_points(fig, point_size)
        elif view_mode == 'clusters':
            self._add_clusters(fig, point_size)
        elif view_mode == 'hierarchy':
            self._add_hierarchy(fig, point_size)

        # Get the base layout from FIGURE_STYLES
        layout = FIGURE_STYLES['layout'].copy()
        
        # Add specific layout settings
        layout.update({
            'xaxis': {
                'title': 'x',
                'range': [-1.2/zoom_level, 1.2/zoom_level],
                'zeroline': True,
                'zerolinecolor': COLORS['grid'],
                'zerolinewidth': 1,
                'showgrid': False
            },
            'yaxis': {
                'title': 'y',
                'range': [-1.2/zoom_level, 1.2/zoom_level],
                'zeroline': True,
                'zerolinecolor': COLORS['grid'],
                'zerolinewidth': 1,
                'scaleanchor': 'x',
                'scaleratio': 1,
                'showgrid': False
            },
            'showlegend': False,
            'margin': dict(l=40, r=40, t=40, b=40)
        })

        # Update the layout
        fig.update_layout(layout)

        return fig

    def _add_points(self, fig, point_size):
        """Add points to the figure."""
        points = self.disk.get_points()
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
                    color=COLORS['point'],
                    line=dict(width=1, color='white')
                ),
                text=labels,
                hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
                showlegend=False
            ))

    def _add_clusters(self, fig, point_size):
        """Add clusters to the figure."""
        # TODO: Implement cluster visualization
        pass

    def _add_hierarchy(self, fig, point_size):
        """Add hierarchy to the figure."""
        # TODO: Implement hierarchy visualization
        pass 