"""
Style configuration for the Hyperbolic Learning Dashboard.
Modern shadcn-inspired design system.
"""
from dash import html

# Modern color scheme inspired by shadcn/ui
COLORS = {
    # Background colors
    'background': '#ffffff',
    'background_secondary': '#f8fafc',
    'background_muted': '#f1f5f9',
    
    # Foreground colors
    'foreground': '#0f172a',
    'foreground_muted': '#64748b',
    'foreground_secondary': '#475569',
    
    # Primary colors
    'primary': '#0f172a',
    'primary_foreground': '#f8fafc',
    
    # Secondary colors
    'secondary': '#f1f5f9',
    'secondary_foreground': '#0f172a',
    
    # Accent colors
    'accent': '#3b82f6',
    'accent_foreground': '#ffffff',
    'accent_muted': '#dbeafe',
    
    # Border colors
    'border': '#e2e8f0',
    'border_muted': '#f1f5f9',
    
    # Status colors
    'success': '#10b981',
    'warning': '#f59e0b',
    'destructive': '#ef4444',
    
    # Visualization specific
    'disk_boundary': '#475569',
    'grid': '#e2e8f0',
    'point': '#3b82f6',
    'point_highlight': '#ef4444'
}

# Layout styles with modern design
LAYOUT_STYLES = {
    'app_container': {
        'backgroundColor': COLORS['background'],
        'minHeight': '100vh',
        'fontFamily': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        'color': COLORS['foreground'],
        'lineHeight': '1.5'
    },
    'navbar': {
        'backgroundColor': COLORS['background'],
        'borderBottom': f"1px solid {COLORS['border']}",
        'padding': '0 24px',
        'height': '64px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'position': 'sticky',
        'top': '0',
        'zIndex': '50',
        'backdropFilter': 'blur(8px)',
        'backgroundColor': 'rgba(255, 255, 255, 0.95)'
    },
    'navbar_title': {
        'fontSize': '20px',
        'fontWeight': '600',
        'color': COLORS['foreground'],
        'margin': '0'
    },
    'navbar_actions': {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '12px'
    },
    'main_content': {
        'display': 'flex',
        'height': 'calc(100vh - 64px)',
        'overflow': 'hidden'
    },
    'visualization_grid': {
        'flex': '1',
        'padding': '24px',
        'display': 'grid',
        'gridTemplateColumns': '1fr 1fr',
        'gridTemplateRows': '1fr 1fr',
        'gap': '24px',
        'backgroundColor': COLORS['background_secondary']
    },
    'card': {
        'backgroundColor': COLORS['background'],
        'border': f"1px solid {COLORS['border']}",
        'borderRadius': '8px',
        'overflow': 'hidden',
        'boxShadow': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
    },
    'card_header': {
        'padding': '16px 20px 12px',
        'borderBottom': f"1px solid {COLORS['border']}"
    },
    'card_title': {
        'fontSize': '16px',
        'fontWeight': '600',
        'color': COLORS['foreground'],
        'margin': '0'
    },
    'card_content': {
        'padding': '0',
        'height': 'calc(100% - 60px)'
    },
    'sidebar': {
        'width': '320px',
        'backgroundColor': COLORS['background'],
        'borderLeft': f"1px solid {COLORS['border']}",
        'padding': '24px',
        'overflowY': 'auto',
        'transition': 'transform 0.2s ease-in-out'
    },
    'sidebar_hidden': {
        'width': '0',
        'padding': '0',
        'overflow': 'hidden',
        'transition': 'all 0.2s ease-in-out'
    },
    'sidebar_toggle': {
        'backgroundColor': COLORS['background'],
        'border': f"1px solid {COLORS['border']}",
        'borderRadius': '6px',
        'padding': '8px 12px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'fontWeight': '500',
        'color': COLORS['foreground'],
        'transition': 'all 0.2s ease-in-out',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px'
    },
    'control_group': {
        'marginBottom': '24px'
    },
    'control_label': {
        'fontSize': '14px',
        'fontWeight': '500',
        'color': COLORS['foreground'],
        'marginBottom': '8px',
        'display': 'block'
    },
    'button_primary': {
        'backgroundColor': COLORS['accent'],
        'color': COLORS['accent_foreground'],
        'border': 'none',
        'borderRadius': '6px',
        'padding': '10px 16px',
        'fontSize': '14px',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'all 0.2s ease-in-out',
        'width': '100%'
    },
    'button_secondary': {
        'backgroundColor': COLORS['secondary'],
        'color': COLORS['secondary_foreground'],
        'border': f"1px solid {COLORS['border']}",
        'borderRadius': '6px',
        'padding': '10px 16px',
        'fontSize': '14px',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'all 0.2s ease-in-out'
    }
}

# Plotly figure styles with modern theme
FIGURE_STYLES = {
    'layout': {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {
            'family': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            'color': COLORS['foreground'],
            'size': 12
        },
        'xaxis': {
            'gridcolor': COLORS['border'],
            'zerolinecolor': COLORS['border'],
            'tickfont': {'color': COLORS['foreground_muted'], 'size': 11},
            'title': {'font': {'color': COLORS['foreground_muted'], 'size': 12}}
        },
        'yaxis': {
            'gridcolor': COLORS['border'],
            'zerolinecolor': COLORS['border'],
            'tickfont': {'color': COLORS['foreground_muted'], 'size': 11},
            'title': {'font': {'color': COLORS['foreground_muted'], 'size': 12}}
        },
        'colorway': [COLORS['accent'], COLORS['success'], COLORS['warning'], COLORS['destructive']]
    }
}

def create_styled_div(children, style_key=None, style=None, **kwargs):
    """
    Create a div with the specified style and attributes.
    
    Args:
        children: The content of the div
        style_key: A key from LAYOUT_STYLES to use as the style
        style: A direct style dictionary to use (overrides style_key if both are provided)
        **kwargs: Additional HTML attributes (id, className, etc.)
    """
    if style is not None:
        return html.Div(children, style=style, **kwargs)
    elif style_key is not None:
        return html.Div(children, style=LAYOUT_STYLES[style_key], **kwargs)
    else:
        return html.Div(children, **kwargs) 