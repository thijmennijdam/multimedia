import dash
from src.layout import make_layout
from src.callbacks import register_callbacks


def build_app(debug: bool = False) -> dash.Dash:
    # CSS to remove number input spinners
    external_stylesheets = [{
        'href': 'data:text/css;charset=utf-8,' + '''
            input[type=number]::-webkit-outer-spin-button,
            input[type=number]::-webkit-inner-spin-button {
                -webkit-appearance: none !important;
                margin: 0 !important;
            }
            input[type=number] {
                -moz-appearance: textfield !important;
            }
        '''.replace('\n', '').replace(' ', '%20'),
        'rel': 'stylesheet'
    }]
    
    app = dash.Dash(__name__, prevent_initial_callbacks=True, external_stylesheets=external_stylesheets)
    app.layout = make_layout()
    register_callbacks(app)
    return app


if __name__ == "__main__":
    app = build_app(debug=True)
    app.run(debug=True, host="0.0.0.0", port=8080)