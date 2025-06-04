import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import umap.umap_ as umap
from sklearn.datasets import load_digits

# Load and reduce data
digits = load_digits()
umap_2d = umap.UMAP(output_metric='hyperboloid', n_components=2, random_state=42)
xy = umap_2d.fit_transform(digits.data)
z = np.sqrt(1 + np.sum(xy**2, axis=1, keepdims=True))
embedding = np.hstack([xy, z])  # 🔽 scale all coords to ~[-1, 1]

# Hyperboloid mesh
x_vals = np.linspace(-3, 3, 50)
y_vals = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.sqrt(1 + X**2 + Y**2)

# Create app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("3D Hyperbolic UMAP with Toggleable Hyperboloid"),
    dcc.Checklist(
        id='show-surface',
        options=[{"label": " Show hyperboloid", "value": "show"}],
        value=["show"],
        inline=True,
        style={'margin': '10px'}
    ),
    dcc.Graph(id='scatter-plot-3d', style={'height': '80vh'}),
])

@app.callback(
    Output('scatter-plot-3d', 'figure'),
    Input('show-surface', 'value')
)
def update_figure(show_surface):
    # Scatter points
    scatter = go.Scatter3d(
        x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
        mode='markers',
        marker=dict(size=3, color=digits.target, colorscale='Spectral'),
        name="Embedding"
    )

    # Optional surface
    data = [scatter]
    if "show" in show_surface:
        surface = go.Surface(
            x=X, y=Y, z=Z,
            showscale=False,
            opacity=0.2,
            colorscale='Greys',
            name="Hyperboloid"
        )
        data.append(surface)

    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            xaxis=dict(range=[-1.2, 1.2]),
            yaxis=dict(range=[-1.2, 1.2]),
            zaxis=dict(range=[1, 2]),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="3D UMAP Projection on Hyperboloid Surface"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)