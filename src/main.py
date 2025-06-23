import dash
from src.layout import make_layout
from src.callbacks import register_callbacks


def build_app(debug: bool = False) -> dash.Dash:
    app = dash.Dash(__name__, prevent_initial_callbacks="initial_duplicate")
    app.layout = make_layout()
    register_callbacks(app)
    return app


if __name__ == "__main__":
    app = build_app(debug=True)
    app.run(debug=True, host="0.0.0.0", port=8080)