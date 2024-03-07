import dash
from dash import Dash, html, dcc
import plotly.graph_objects as go

from tools import *
from models import *
from static import *


# ----------- Simulation ----------------------------
steps = 20
model = abmodel(static_map_v0(), all_agents())
model.run_model(steps)
model_results = model.datacollector.get_model_vars_dataframe()
print(model_results)

# ---------------------------------------------------
app = Dash(__name__, external_stylesheets=['custom_styles.css'])



app.layout = html.Div([

    
html.Div(
    
    children = [html.Div(className = "div-logo", 
                         style = {"background": "#2c5c97", 'padding': '20px', 'border-radius': '10px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}, 
                         children = [html.Img(className="logo", src = ("https://opendatabim.io/wp-content/uploads/2021/12/GitHub-Mark-Light-64px-1.png")), 
                                     html.H1("Agent-Based Modeling in Finance - Team Banana", style = {'font-size': '36px',  # Increase font size, 
                                                                                                       'font-family': 'Roboto', 
                                                                                                       'color': 'white',  # Match color with the logo background 
                                                                                                       'margin-left': '10px',  # Add spacing between logo and title 
                                                                                                       'margin-tap': '-50px'}) ]),
        html.A(id = 'gh-link', children = ['View on GitHub'], href = 'https://github.com/Gordonhosy/ABM_FX', style = {'font-size': '18px', 'text-align': 'center', 'color': 'white', 'text-decoration': 'none', 'display': 'block', 'margin-top': '-30px'})]
    ),





    html.Footer("© 2024 Agent-Based Modeling Team Banana. All rights reserved.", style={'text-align': 'center', 'color': '#555', 'padding': '10px', 'margin-top': '20px'})

]) 




if __name__ == '__main__':
    app.run_server(debug=True)