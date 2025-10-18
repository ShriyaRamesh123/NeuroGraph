import dash
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
from importance_regions_plot import get_static_figure
from functional_connectivity_plot import get_animated_figure
from Final_model_run import process_data  # Your custom AI logic

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app = dash.Dash(__name__)

path="data-20250421T122540Z-001/data/cpac/nofilt_noglobal/"
file_id=None




app.layout = html.Div(
   
    #className='black-scroll',
    children=[
     dcc.Store(id='prediction-cache'),
    # === Prediction Section (Above Tabs) ===
    html.Div([
        html.H2(
    "🧠 AI Model Prediction",
    className='HeadingDiv'),

        html.Div( children=[
            html.Label("Enter File ID:", className='SubheadDiv', style={'font-size': '25px','padding':'5px', 'text-align':'center'}),
            dcc.Input(id='file-id-input', className="custom-input", type='text', value='', debounce=True,
                      placeholder='e.g., NYU_0051070'),
            dbc.Button("Run Prediction", id="predict-btn", color="primary",style={'align-item':'center'}),
        ]),
        dcc.Loading(
        id="loading-predict",
        type="circle",  # or "default", "dot"
        color="cyan",
        children=html.Div(id='model-output', style={'paddingBottom': '10px'})),

        
    ]),

    # === Visualization Tabs ===
    html.Div(className="BodyDiv", children=[
        html.H2("Important Brain ROIs Graph Visualization"),
    dcc.Tabs([
        dcc.Tab(label='Important ROIs', className='custom-tab', children=[
            dcc.Graph(id='static-graph', className='custom-tab', style={'height': '70vh', 'width': '100%', 'BackgroundColor':'black'})
        ]),
        dcc.Tab(label='Time Series plot of ROIs', children=[
            dcc.Graph(id='animated-graph', style={'height': '50vh', 'width': '100%', 'BackgroundColor':'black', 'color':'black'})
        ]),
    ])])
], style={'backgroundColor': 'transperant', 'color': 'black', 'padding': '0px'})


@app.callback(
    Output("prediction-cache", "data"),
    Input("predict-btn", "n_clicks"),
    State("file-id-input", "value"),
    prevent_initial_call=True
)
def compute_prediction(n_clicks, file_id):
    if not file_id:
        return dash.no_update
    prediction, confidence = process_data(file_id, path=path)
    if (prediction=='ASD'):
        confidence=1-confidence
    static_fig = get_static_figure(file_id)  # must NOT reload or reprocess
    animated_fig = get_animated_figure(file_id)

    return {
        "file_id": file_id,
        "prediction": prediction,
        "confidence":confidence,
        "static_fig": static_fig,
        "animated_fig": animated_fig
    }
#'''
@app.callback(
    Output("model-output", "children"),
    Input("prediction-cache", "data"),
    prevent_initial_call=True
)
def update_prediction_display(data):
    if not data:
        return "No prediction available."
    children=notification = [
    html.Div(className="notiglow"),
    html.Div(className="notiborderglow"),
    html.Div(f"Prediction: {data['prediction']}", className="notititle"),
    html.Div(f"Confidence: {data['confidence']*100:.2f}", className="notibody")
]
    return html.Div(children=children, className="notification")
#'''    
@app.callback(
    Output("static-graph", "figure"),
    Output("animated-graph", "figure"),
    Input("prediction-cache", "data"),
    prevent_initial_call=True
)
def update_graphs(data):
    return data["static_fig"], data["animated_fig"]


if __name__ == '__main__':
    app.run(debug=True)
