"""
Visualization module for hierarchical structures.
"""
import plotly.graph_objects as go
from ..utils.style_config import COLORS, FIGURE_STYLES

class HierarchyVisualization:
    def __init__(self, builder):
        """
        Initialize the hierarchy visualization.
        
        Args:
            builder: HierarchyBuilder instance to visualize
        """
        self.builder = builder

    def create_figure(self, zoom_level=1, point_size=5, view_mode='points'):
        """
        Create a figure of the hierarchy.
        
        Args:
            zoom_level: Zoom level for the visualization
            point_size: Size of points in the visualization
            view_mode: View mode ('points', 'clusters', or 'hierarchy')
            
        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        fig = go.Figure()
        
        hierarchy = self.builder.get_hierarchy()
        if hierarchy:
            self._add_hierarchy(fig, hierarchy, point_size)

        layout = FIGURE_STYLES['layout'].copy()
        max_levels = max(self.builder.get_levels(), 2)
        layout.update({
            'xaxis': {
                'range': [-1, 1],
                'showticklabels': False,
                'showgrid': False,
                'zeroline': False
            },
            'yaxis': {
                'range': [-0.5, max_levels - 0.5],
                'showticklabels': False,
                'showgrid': False,
                'zeroline': False
            },
            'margin': dict(l=40, r=40, t=40, b=40),
            'showlegend': False
        })
        
        fig.update_layout(layout)
        return fig

    def _add_hierarchy(self, fig, node, point_size, level=0, x=0, width=1):
        """Add a node and its children to the figure."""
        # Calculate positions for children
        num_children = len(node['children'])
        if num_children > 0:
            child_width = width / num_children
            for i, child in enumerate(node['children']):
                child_x = x - width/2 + child_width/2 + i * child_width
                self._add_hierarchy(fig, child, point_size, level + 1, child_x, child_width)
        
        # Add lines to children
        for i, child in enumerate(node['children']):
            child_x = x - width/2 + width * (i + 0.5) / num_children
            fig.add_trace(go.Scatter(
                x=[x, child_x],
                y=[level, level + 1],
                mode='lines',
                line=dict(color=COLORS['grid'], width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Calculate node info
        num_points = len(node.get('points', []))
        node_label = f"Level {level}"
        if level == 0:
            node_label = "Root"
        
        # Add node
        fig.add_trace(go.Scatter(
            x=[x],
            y=[level],
            mode='markers',
            marker=dict(
                size=max(8, min(point_size + 5, 20)),
                color=COLORS['secondary'],
                line=dict(width=1, color='white')
            ),
            text=[node_label],
            hovertemplate=f'<b>{node_label}</b><br>Points: {num_points}<br>Children: {num_children}<extra></extra>',
            showlegend=False
        )) 