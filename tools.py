import numpy as np
import pandas as pd 
from corporates import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# tools for model

def random_corporate(agent_id, corporate_type, static_map, model, country=None):
    '''
    generate random corporates at random postitions
    '''

    map_a = static_map.currencyA_map_init
    map_b = static_map.currencyB_map_init
    world_map = map_a + map_b


    if country == None:
        country = str(np.random.choice(corporate_type.params.country, p=[0.88, 0.12])) # ratio according to empirical data
    else:
        country = country
    
    if (country == "A"):
        non_zero_index = np.nonzero(map_a)
        shuffle_index_len = len(non_zero_index[0])
        random_pos_index = np.random.randint(0, shuffle_index_len, size=1)
        
    elif (country == "B"):
        non_zero_index = np.nonzero(map_b)
        shuffle_index_len = len(non_zero_index[0])
        random_pos_index = np.random.randint(0, shuffle_index_len, size=1)

    x = int(non_zero_index[0][random_pos_index])
    y = int(non_zero_index[1][random_pos_index])

    level = int(np.random.uniform(corporate_type.params.level_min, corporate_type.params.level_max + 1))
    vision = level

    if country == 'A':
        amount_A = int(np.random.uniform(corporate_type.params.asset_min, corporate_type.params.asset_max + 1))
        currencyA = int(amount_A * level)

        amount_B = int(np.random.uniform(corporate_type.params.asset_min, corporate_type.params.asset_max + 1))
        currencyB = int(amount_B * level / 2)

        cost_currencyA = int(np.random.uniform((corporate_type.params.costs_min + corporate_type.params.costs_max)/2, corporate_type.params.costs_max + 1)) 
        cost_currencyB = int(np.random.uniform(corporate_type.params.costs_min, (corporate_type.params.costs_min + corporate_type.params.costs_max)/2))

    elif country == "B":
        amount_A = int(np.random.uniform(corporate_type.params.asset_min, corporate_type.params.asset_max + 1))
        currencyA = int(amount_A * level / 2) 

        amount_B = int(np.random.uniform(corporate_type.params.asset_min, corporate_type.params.asset_max + 1))
        currencyB = int(amount_B * level)
        
        cost_currencyA = int(np.random.uniform(corporate_type.params.costs_min, (corporate_type.params.costs_min + corporate_type.params.costs_max)/2)) 
        cost_currencyB = int(np.random.uniform((corporate_type.params.costs_min + corporate_type.params.costs_max)/2, corporate_type.params.costs_max + 1))


    if level == 1:
        imp_utility = np.arange(1.04, 1.14, 0.01)
    elif level == 2:
        imp_utility = np.arange(1.04, 1.11, 0.01)
    elif level == 3:
        imp_utility = np.arange(1.03, 1.09, 0.01)
    elif level == 4:
        imp_utility = np.arange(1.03, 1.07, 0.01)
    elif level == 5:
        imp_utility = np.arange(1.02, 1.05, 0.01)
    
    return corporate_type.agent(agent_id,
                           model,
                           (x,y), 
                           moore = False,
                           country = country,
                           currencyA = currencyA,
                           currencyB = currencyB,
                           cost_currencyA = cost_currencyA,
                           cost_currencyB = cost_currencyB,
                           level = level,
                           vision = vision,
                           imp_utility = imp_utility)


def central_bank_A(agent_id, central_bank_type, static_map, model):
    '''
    generate random central bank with random initial economic situation(or not just make two completely different macro environment)
    '''
    x = int(np.random.uniform(0, static_map.width))
    y = int(np.random.uniform(0, static_map.height))
    country = "A"

    inflation_rate = 0.06
    interest_rate = 0.0025
    growth_rate = interest_rate - inflation_rate 
    target_inflation_rate = 0.02
    currencyA = 1000000
    currencyB = 0
    lend = 0

    agent_central_bank = central_bank_type.agent(agent_id, 
                                                 model, 
                                                 (x,y), 
                                                 moore = False, 
                                                 interest_rate = interest_rate, 
                                                 inflation_rate = inflation_rate, 
                                                 growth_rate = growth_rate,
                                                 currencyA = currencyA,
                                                 currencyB = currencyB,
                                                 lend = lend,
                                                 target_inflation_rate = target_inflation_rate,
                                                 country = country)

    return agent_central_bank


def central_bank_B(agent_id, central_bank_type, static_map, model):
    '''
    generate random central bank with random initial economic situation(or not just make two completely different macro environment)
    '''
    x = int(np.random.uniform(0, static_map.width))
    y = int(np.random.uniform(0, static_map.height))
    country = "B"

    inflation_rate = 0.01
    interest_rate = -0.0025
    growth_rate = interest_rate - inflation_rate 
    target_inflation_rate = 0.015
    currencyA = 0
    currencyB = 500000
    lend = 0

    agent_central_bank = central_bank_type.agent(agent_id, 
                                                 model, 
                                                 (x,y), 
                                                 moore = False, 
                                                 interest_rate = interest_rate, 
                                                 inflation_rate = inflation_rate, 
                                                 growth_rate = growth_rate,
                                                 currencyA = currencyA,
                                                 currencyB = currencyB,
                                                 lend = lend,
                                                 target_inflation_rate = target_inflation_rate,
                                                 country = country)

    return agent_central_bank

def random_bank(agent_id, bank_type, init_pos, model):
    '''
    generate local banks at fixed positions
    '''
    x = init_pos[0]
    y = init_pos[1]
    
    # if it is a bank in country A --> USA
    if y < 60: 
        currencyA = int(np.random.uniform(bank_type.params.local_asset_min, bank_type.params.local_asset_max + 1))
        currencyB = int(np.random.uniform(bank_type.params.foreign_asset_min, bank_type.params.foreign_asset_max + 1))
        cost_currencyA = int(np.random.uniform(bank_type.params.local_costs_min, bank_type.params.local_costs_max + 1))
        cost_currencyB = int(np.random.uniform(bank_type.params.foreign_costs_min, bank_type.params.foreign_costs_max + 1))
    
    # if it is a bank in country B --> JAPAN
    elif y >= 60:
        currencyA = int(np.random.uniform(bank_type.params.foreign_asset_min, bank_type.params.foreign_asset_max + 1))
        currencyB = int(np.random.uniform(bank_type.params.local_asset_min, bank_type.params.local_asset_max + 1))
        cost_currencyA = int(np.random.uniform(bank_type.params.foreign_costs_min, bank_type.params.foreign_costs_max + 1))
        cost_currencyB = int(np.random.uniform(bank_type.params.local_costs_min, bank_type.params.local_costs_max + 1))

    vision = int(np.random.uniform(bank_type.params.vision_min, bank_type.params.vision_max + 1))
    
    return bank_type.agent(agent_id,
                           model,
                           (x,y), 
                           moore = False,
                           currencyA = currencyA,
                           currencyB = currencyB,
                           cost_currencyA = cost_currencyA,
                           cost_currencyB = cost_currencyB,
                           vision = vision)


def random_arb(agent_id, arb_type, init_pos, model):
    '''
    generate arbitragers at fixed positions
    '''
    x = init_pos[0]
    y = init_pos[1]

    currencyA = int(np.random.uniform(arb_type.params.asset_min, arb_type.params.asset_max + 1))
    currencyB = int(np.random.uniform(arb_type.params.asset_min, arb_type.params.asset_max + 1))
    cost_currencyA = int(np.random.uniform(arb_type.params.costs_min, arb_type.params.costs_max + 1))
    cost_currencyB = int(np.random.uniform(arb_type.params.costs_min, arb_type.params.costs_max + 1))
    
    vision = int(np.random.uniform(arb_type.params.vision_min, arb_type.params.vision_max + 1))
    
    return arb_type.agent(agent_id,
                           model,
                           (x,y), 
                           moore = False,
                           currencyA = currencyA,
                           currencyB = currencyB,
                           cost_currencyA = cost_currencyA,
                           cost_currencyB = cost_currencyB,
                           vision = vision)

def random_fund(agent_id, fund_type, init_pos, strategy, model):
    '''
    generate arbitragers at fixed positions
    '''
    x = init_pos[0]
    y = init_pos[1]

    currencyA = int(np.random.uniform(fund_type.params.asset_min, fund_type.params.asset_max + 1))
    currencyB = int(np.random.uniform(fund_type.params.asset_min, fund_type.params.asset_max + 1))
    cost_currencyA = int(np.random.uniform(fund_type.params.costs_min, fund_type.params.costs_max + 1))
    cost_currencyB = int(np.random.uniform(fund_type.params.costs_min, fund_type.params.costs_max + 1))
    
    return fund_type.agent(agent_id,
                           model,
                           (x,y), 
                           moore = False,
                           currencyA = currencyA,
                           currencyB = currencyB,
                           cost_currencyA = cost_currencyA,
                           cost_currencyB = cost_currencyB,
                           strat = strategy)



# ---------------- Visualization Tools --------------------
import datetime as dt
import yfinance as yf 
from fredapi import Fred 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf


# yfinance likes the tickers formatted as a list
api_key = 'a0aee094a9b908bd7c16baece8df8419'
start_date = dt.datetime(2000,1,1)
end_date =  dt.datetime(2024,1,1)


def get_price_df(model, model_results, steps):

    best_bid = []
    best_ask = []
    mid_price = []


    for i in range(steps):

        interbank_bid, interbank_ask = model.bank_details.lob(step = i)

        try:
            best_bid_price = np.sort(list(interbank_bid.keys()))[-1]
        except:
            best_bid_price = np.NaN
        
        try:
            best_ask_price = np.sort(list(interbank_ask.keys()))[0]
        except:
            best_ask_price = np.NaN

        best_bid.append(best_bid_price)
        best_ask.append(best_ask_price)
        mid_price.append((best_bid_price + best_ask_price)/2)



    interest_rate_1_data, interest_rate_2_data = zip(*model_results['interest_rate'].values)

    price_df = pd.DataFrame()
    price_df['time_steps'] = np.arange(1, steps +1)
    price_df['Ask(1)'] = best_ask
    price_df['Bid(1)'] = best_bid
    price_df['Mid Price'] = mid_price

    price_df['interest_rate_1'] = interest_rate_1_data
    price_df['interest_rate_2'] = interest_rate_2_data
    price_df['interest_rate_diff'] = price_df['interest_rate_1'] - price_df['interest_rate_2']
    price_df['return_mid_price'] = np.log(price_df['Mid Price']).diff()

    return price_df

def get_market_data(start_date, end_date, api_key):

    yahoo_df = yf.download("JPY=X", start = start_date, end = end_date, interval='1d')
    fred = Fred(api_key = api_key)

    fed_rate = fred.get_series('DFF').reset_index(name = 'fed fund effective rate')
    jp_rate = fred.get_series('IRSTCI01JPM156N').reset_index(name = 'BOJ effective rate')
    interest_rate_df = pd.merge(fed_rate, jp_rate)
    interest_rate_df['interest_rate_diff'] = interest_rate_df['fed fund effective rate'] - interest_rate_df['BOJ effective rate'] 

    # yahoo_df = pd.merge(yahoo_df, interest_rate_df, left_index=True, right_on='index')
    yahoo_df['return'] = np.log(yahoo_df['Close']).diff(1)

    merge_df = pd.merge(yahoo_df, interest_rate_df, left_index = True, right_on = 'index')

    return yahoo_df, interest_rate_df, merge_df



def plot_central_bank(model_results):

    steps_data = model_results['Step'].values
    interest_rate_1_data, interest_rate_2_data = zip(*model_results['interest_rate'].values)
    inflation_rate_1_data, inflation_rate_2_data = zip(*model_results['inflation_rate'].values)

    growth_rate_1_data, growth_rate_2_data = zip(*model_results['growth_rate'].values)
    target_interest_rate_1_data, target_interest_rate_2_data = zip(*model_results['target_interest_rate'].values)
    target_inflation_rate_1_data, target_inflation_rate_2_data = zip(*model_results['target_inflation_rate'].values)


    fig = make_subplots(rows = 2, cols = 2, subplot_titles=['Central Bank 1', 'Central Bank 2'])
    temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))

    inflation_rate_1 = go.Scatter(x = steps_data, y = inflation_rate_1_data, name = 'Inflation Rate(1)', mode = 'lines', line = dict(color = 'blue'))
    growth_rate_1 = go.Scatter(x = steps_data, y = growth_rate_1_data, name = 'Growth Rate(1)', mode = 'lines', line = dict(color = 'black'))
    interest_rate_1 = go.Bar(x = steps_data, y = interest_rate_1_data, name = 'Interest Rate(1)', marker = dict(color = 'green'))
    # target_interest_rate_1 = go.Scatter(x = steps_data, y = target_interest_rate_1_data, name = 'target interest rate (1)',  line = dict(color = 'red'))

    inflation_rate_2 = go.Scatter(x = steps_data, y = inflation_rate_2_data, name = 'Inflation Rate(2)', mode = 'lines', line = dict(color = 'blue'))
    growth_rate_2 = go.Scatter(x = steps_data, y = growth_rate_2_data, name = 'Growth Rate(2)', mode = 'lines', line = dict(color = 'black'))
    interest_rate_2 = go.Bar(x = steps_data, y = interest_rate_2_data, name = 'Interest Rate(2)', marker = dict(color = 'green'))
    # target_interest_rate_2 = go.Scatter(x = steps_data, y = target_interest_rate_2_data, name = 'target interest rate (2)',  line = dict(color = 'red'))


    fig.add_trace(inflation_rate_1, row = 1, col = 1)
    fig.add_trace(interest_rate_1, row = 1, col = 1)
    fig.add_trace(interest_rate_1, row = 2, col = 1)
    fig.add_trace(growth_rate_1, row = 2, col = 1)
    # fig.add_trace(target_interest_rate_1, row = 1, col = 1)

    fig.add_trace(inflation_rate_2, row = 1, col = 2)
    fig.add_trace(interest_rate_2, row = 1, col = 2)
    fig.add_trace(interest_rate_2, row = 2, col = 2)
    fig.add_trace(growth_rate_2, row = 2, col = 2)
    # fig.add_trace(target_interest_rate_2, row = 1, col = 2)

    fig.update_layout(title_text = 'Central Bank Agent Behavior and Macroeconomics', showlegend=True)

    fig.update_layout(template = temp,
                      hovermode = 'closest',
                      margin = dict(l = 40, r = 40, t = 100, b = 40),
                      height = 800, 
                      width = 1200, 
                      showlegend = True,
                      xaxis = dict(tickfont=dict(size=10)),  
                      yaxis = dict(side = "left", tickfont = dict(size=10)),
                      xaxis_showgrid = False, 
                      legend = dict(yanchor = "bottom", y = 0.45, xanchor = "left", x = 0.01,  orientation="h"))

    return fig 

def plot_map(model):

    map_a = model.static_map.currencyA_map_init
    map_b = model.static_map.currencyB_map_init
    world_map = map_a + map_b

    fig = px.imshow(world_map, color_continuous_scale='dense')
    temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))

    fig.update_layout(template = temp,
                    title_text = 'Overall Map Layout - Mimic US and Japan',
                    hovermode = 'closest', 
                    margin = dict(l = 20, r = 20, t = 100, b = 20),
                    height = 500, 
                    width = 1200, 
                    showlegend = True, 
                    legend = dict(yanchor = "top", 
                                    y = 0.99,
                                    xanchor = "left",
                                    x = 0.01))

    return fig

def plot_agent_movement(model, agent_position_df):

    temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))
    map_a = model.static_map.currencyA_map_init
    map_b = model.static_map.currencyB_map_init
    world_map = map_a + map_b


    fig = px.scatter(agent_position_df, 
                    x = "x", 
                    y = "y", 
                    animation_frame = "steps", 
                    animation_group = "agent_id", 
                    color = "agent_type", 
                    color_discrete_sequence = ["black", "red", "green"],
                    hover_name = "agent_type")

    fig.add_heatmap(z = world_map, 
                    colorscale='dense', 
                    colorbar=dict(title='Level of Resources'))

    fig.update_yaxes(range = [agent_position_df['y'].max() + 5, agent_position_df['y'].min() - 5])

    fig.update_layout(template = temp,
                    title_text = 'Position of the Agents on The Map',
                    hovermode = 'closest', 
                    margin = dict(l = 20, r = 20, t = 50, b = 20),
                    height = 700, 
                    width = 1300, 
                    showlegend = True, 
                    legend = dict(yanchor = "bottom", 
                                    y = 0.99,
                                    xanchor = "left",
                                    x = 0.01))
    return fig

def built_agent_position_df(model, steps):

    agent_position_df = pd.DataFrame()

    for i in range(steps):

        # corporations
        pos_corps = pd.DataFrame(model.corporate_details.agent_pos[i], index = model.corporate_details.agent_id[i], columns = ["y", "x"]).reset_index(names = 'agent_id')
        pos_corps['steps'] = i
        pos_corps['agent_type'] = 'corps'

        # banks 
        pos_banks = pd.DataFrame(model.bank_details.agent_pos[i], index = model.bank_details.agent_id[i], columns = ["y", "x"]).reset_index(names = 'agent_id')
        pos_banks['steps'] = i
        pos_banks['agent_type'] = 'banks'

        # international banks
        #pos_international_banks = pd.DataFrame(model.international_bank_details.agent_pos[i], index = model.international_bank_details.agent_id[i], columns = ["y", "x"]).reset_index(names = 'agent_id')
        #pos_international_banks['steps'] = i
        #pos_international_banks['agent_type'] = 'international_banks'


        # aggregate position df
        #frames = [pos_corps, pos_banks, pos_international_banks]
        frames = [pos_corps, pos_banks]
        all_position = pd.concat(frames)
        
        if i == 0:
            agent_position_df  = all_position
        
        else:
            frames = [agent_position_df, all_position]
            agent_position_df = pd.concat(frames)


    return agent_position_df

def plot_agent_population(agent_position_df):

    number_of_agents_df = agent_position_df.groupby(['steps', 'agent_type'])['agent_id'].count().reset_index(name = 'Number of Agents')
    number_of_agents_corps = number_of_agents_df[number_of_agents_df['agent_type'] == 'corps']


    fig = make_subplots(rows=1, cols=1)
    temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))
    corps = go.Scatter(y = number_of_agents_corps['Number of Agents'], name = 'Population')

    fig.add_trace(corps)

    fig.update_layout(template = temp,
                    title_text = 'Number of Corporation Agent',
                    hovermode = 'closest', 
                    margin = dict(l = 20, r = 20, t = 50, b = 20),
                    height = 300, 
                    width = 600, 
                    showlegend = True, 
                    legend = dict(yanchor = "bottom", 
                                    y = 0.99,
                                    xanchor = "left",
                                    x = 0.01))

    return fig

def plot_central_bank_behaviors_diagram(image_path):

    fig = go.Figure()

    # Constants
    img_width = 1600
    img_height = 900
    scale_factor = 0.5

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=image_path)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    # Disable the autosize on double click because it adds unwanted margins around the image
    # More detail: https://plotly.com/python/configuration-options/
    return fig

def corporate_value_and_interest_rate_plot(model, model_results, steps):

    steps_data = model_results['Step'].values
    corporate_value = [model.corporate_details.by_step(i)['Firm Value'].mean() for i in range(steps)]
    interest_rate_1_data, interest_rate_2_data = zip(*model_results['interest_rate'].values)


    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))
    interest_rate_1 = go.Scatter(x = steps_data, y = interest_rate_1_data, name = 'Interest Rate(1)', line = dict(color = 'green'))
    interest_rate_2 = go.Scatter(x = steps_data, y = interest_rate_2_data, name = 'Interest Rate(2)', line = dict(color = 'grey'))
    corporate_value_line = go.Scatter(x = steps_data, y = corporate_value, name = 'corporate value', line = dict(color = 'blue'))

    fig.add_trace(corporate_value_line, row = 1, col =1, secondary_y='True')
    fig.add_trace(interest_rate_1, row = 1, col = 1)
    fig.add_trace(interest_rate_2, row = 1, col = 1)


    fig.update_layout(template = temp,
                      title = "Corporate Agent Value",
                      hovermode = 'closest',
                      margin = dict(l = 30, r = 30, t = 50, b = 20),
                      height = 400, 
                      width = 800, 
                      showlegend = True,
                      xaxis = dict(title = 'Steps', tickfont = dict(size=10)),
                      yaxis = dict(title = 'Value', side="left", tickfont = dict(size=10)),
                      legend = dict(yanchor = "bottom", y = 0.95, xanchor = "left", x = 0.01,  orientation="h"))

    return fig 

def bank_value_and_interest_rate_plot(model, model_results, steps):

    steps_data = model_results['Step'].values
    bank_value = [model.bank_details.by_step(i)['Firm Value'].mean() / 10 for i in range(steps)]
    traded_volume = model_results['(Corporate V0)'].values
    corporate_value = [model.corporate_details.by_step(i)['Firm Value'].mean() for i in range(steps)]
    interest_rate_1_data, interest_rate_2_data = zip(*model_results['interest_rate'].values)

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))
    interest_rate_1 = go.Bar(x = steps_data, y = interest_rate_1_data, name = 'Interest Rate(1)', marker = dict(color = 'green', opacity = 0.6))
    interest_rate_2 = go.Bar(x = steps_data, y = interest_rate_2_data, name = 'Interest Rate(2)', marker = dict(color = 'grey',  opacity = 0.6))
    bank_value_line = go.Scatter(x = steps_data, y = bank_value, name = 'bank value', line = dict(color = 'red'))
    corporate_value_line = go.Scatter(x = steps_data, y = corporate_value, name = 'corporate value', line = dict(color = 'blue'))
    # traded_volume_line = go.Scatter(x = steps_data, y = traded_volume, name = 'traded volume', line = dict(color = 'orange'))

    fig.add_trace(bank_value_line, row = 1, col = 1, secondary_y='True')
    fig.add_trace(corporate_value_line, row = 1, col = 1)
    # fig.add_trace(traded_volume_line, row = 1, col= 1)
    # fig.add_trace(interest_rate_1, row = 1, col = 1)
    # fig.add_trace(interest_rate_2, row = 1, col = 1)


    fig.update_layout(template = temp,
                      title = "Bank Agent Value",
                      hovermode = 'closest',
                      margin = dict(l = 30, r = 30, t = 50, b = 20),
                      height = 400, 
                      width = 800, 
                      showlegend = True,
                      xaxis = dict(title = 'Steps', tickfont = dict(size=10)),
                      yaxis = dict(title = 'Value', side="left", tickfont = dict(size=10)),
                      legend = dict(yanchor = "bottom", y = 1, xanchor = "left", x = 0.01,  orientation="h"))

    return fig 


def plot_volatility_clustering(price_df, yahoo_df):

  fig = make_subplots(rows = 2, cols = 2, subplot_titles = ['Real Market Data', 'Simulation Result'])
  temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))


  daily_price_line = go.Scatter(x = yahoo_df.index, y = yahoo_df['Close'], name = 'Spot Rate')
  daily_return_line = go.Scatter(x = yahoo_df.index, y = yahoo_df['return'], name = 'Daily Return')

  simulation_price_line = go.Scatter(x = price_df['time_steps'], y = price_df['Mid Price'], name = 'simulation min price', line = dict(color = ' black'))
  simulation_return_line = go.Scatter(x = price_df['time_steps'], y = price_df['return_mid_price'], name = 'simulation mid  price return', line = dict(color = 'gray'))

  fig.update_layout(title_text = 'Stylized Fact - Volatility Clustering', showlegend=True)
  fig.add_trace(daily_price_line, row = 1, col = 1)
  fig.add_trace(daily_return_line, row = 2, col = 1)
  fig.add_trace(simulation_price_line, row = 1, col = 2)
  fig.add_trace(simulation_return_line, row = 2, col = 2)


  fig.update_layout(template = temp,
                      hovermode = 'closest',
                        margin = dict(l = 40, r = 40, t = 60, b = 40),
                        height = 500, 
                        width = 1200, 
                        showlegend = True,
                        xaxis = dict(tickfont=dict(size=10)),  
                        yaxis = dict(side = "left", tickfont = dict(size=10)),
                        xaxis_showgrid = False, 
                        legend = dict(yanchor = "bottom", y = 0.45, xanchor = "left", x = 0.01,  orientation="h"))

  return fig


def plot_fat_tail(yahoo_df, price_df):

    # Calculate mean and standard deviation
    mu, sigma = yahoo_df['return'].mean(), yahoo_df['return'].std()
    simulated_mu, simulated_sigma = yahoo_df['return'].mean(), yahoo_df['return'].std()

    # hist_fig = go.Figure()
    hist_fig = make_subplots(rows = 1, cols = 2, subplot_titles = ['Real Market Data', 'Simulation Result'])
    temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))

    daily_return_histogram = go.Histogram(x = (yahoo_df['return']- mu) / sigma, 
                                        nbinsx = 200, 
                                        histnorm = 'density', 
                                        marker_color = 'rgba(0, 128, 0, 0.6)',
                                        name = 'USD/JPY Daily Return')

    simulation_return_histogram = go.Histogram(x = (price_df['return_mid_price'] - simulated_mu)/simulated_sigma, 
                                                nbinsx = 200, 
                                                histnorm = 'density', 
                                                marker_color = 'rgba(0, 128, 0, 0.6)',
                                                name = 'USD/JPY Simulation Daily Return')


    # Plot the normal distribution
    x = np.linspace(-5, 5, 1000)
    y = norm.pdf(x, 0, 1) * 4000 # Standard normal distribution
    y2 = norm.pdf(x, 0, 1) * 300

    normal_distribution_trace = go.Scatter(x = x, y = y, mode='lines', line=dict(color='black', width=2), name = 'Normal Distribution')
    simulated_normal_distribution_trace = go.Scatter(x = x, y = y2, mode='lines', line=dict(color='black', width=2), name = 'Normal Distribution')


    # Add histogram trace
    hist_fig.add_trace(daily_return_histogram, row = 1, col = 1)
    hist_fig.add_trace(normal_distribution_trace, row = 1, col = 1)
    hist_fig.add_trace(simulation_return_histogram, row = 1, col = 2)
    hist_fig.add_trace(simulated_normal_distribution_trace, row = 1, col = 2)


    hist_fig.update_layout(template = temp,
                        title = 'Stylized Fact - Fat Tail',
                        hovermode = 'closest',
                        xaxis_title='Return Mid Price (Normalized)',
                        yaxis_title='Density',
                        bargap = 0.05,
                        margin = dict(l = 40, r = 40, t = 50, b = 40),
                        height = 600, 
                        width = 1000, 
                        showlegend = True,
                        xaxis = dict(tickfont=dict(size=10), range=[-8, 8]),  
                        xaxis2 = dict(tickfont=dict(size=10), range=[-8, 8]),  
                        yaxis = dict(side = "left", tickfont = dict(size=10)),
                        xaxis_showgrid = False, 
                        legend = dict(yanchor = "bottom", y = 0.95, xanchor = "left", x = 0.01,  orientation = "h"))
    
    return hist_fig


def plot_autocorrelation(yahoo_df, price_df):

    return_data = yahoo_df['return'].dropna()
    simulated_return_data = price_df['return_mid_price'].dropna()

    # Create ACF plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

    fig.suptitle('Stylized Fact - No Serial Autocorrelation', fontweight = 'bold', fontsize = 12) # set title
    plot_acf(return_data, ax = ax1, lags = 40)  # Adjust lags as needed
    plot_acf(simulated_return_data, ax = ax2, lags = 40)  # Adjust lags as needed

    ax1.set_title('Real Market ACF Plot')
    ax2.set_title('Simulated Data ACF Plot')
    ax1.set_xlabel('Lag')
    ax2.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')

    return fig 

def plot_interest_rate_and_price(merge_df, price_df):

    # Create subplots with one row and one column
    fig = make_subplots(rows=1, cols=2, specs=[[{'secondary_y': True}, {'secondary_y': True}]], subplot_titles = ['Real Market Data', 'Simulation Result'])
    temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))

    # Define your traces
    daily_price_line = go.Scatter(x = merge_df.index, y = merge_df['Close'], name = 'Monthly Spot Rate')
    interest_rate_diff_line = go.Scatter(x = merge_df.index, y = merge_df['interest_rate_diff'], name = 'Interest Rate Diff', line = dict(color = ' red'))

    simulated_price_line = go.Scatter(x = price_df['time_steps'], y = price_df['Mid Price'], name = 'Simulated Mid Price', line = dict(color = 'black'))
    simulated_interest_rate_diff_line = go.Scatter(x = price_df['time_steps'], y = price_df['interest_rate_diff'], name = 'Simulated Interest Rate Difference')

    # Add trace1 to the subplot
    fig.add_trace(daily_price_line, col = 1, row = 1)
    fig.add_trace(interest_rate_diff_line, secondary_y=True, col = 1, row = 1)

    fig.add_trace(simulated_price_line, col = 2, row = 1)
    fig.add_trace(simulated_interest_rate_diff_line, secondary_y=True, col = 2, row = 1)

    fig.update_layout(template = temp,
                    title = 'Stylized Fact - Interest Rate & Spot Rate',
                    hovermode = 'closest',
                    margin = dict(l = 40, r = 40, t = 60, b = 40),
                    height = 500, 
                    width = 1200, 
                    showlegend = True,
                    xaxis = dict(tickfont=dict(size=10)),  
                    yaxis = dict(side = "left", tickfont = dict(size=10)),
                    xaxis_showgrid = False, 
                    legend = dict(yanchor = "bottom", y = 0.95, xanchor = "left", x = 0.01,  orientation="h"))

    # Show the figure
    return fig


def plot_aggregation_guaussianity(price_df, yahoo_df):

    resample_price_df_1 = price_df.groupby(price_df.index // 5).last()
    resample_price_df_2 = price_df.groupby(price_df.index // 10).last()


    resample_price_df_1['return_mid_price'] = np.log(resample_price_df_1['Mid Price']).diff()
    resample_price_df_2['return_mid_price'] = np.log(resample_price_df_2['Mid Price']).diff()


    monthly_df = yahoo_df.to_period('M').groupby('Date').last()
    yearly_df = yahoo_df.to_period('Y').groupby('Date').last()

    monthly_df['return'] = np.log(monthly_df['Close']).diff()
    yearly_df['return'] = np.log(yearly_df['Close']).diff()


    fig = make_subplots(rows=2, cols=3, subplot_titles = ['Small Time Steps', 'Medium Time Steps', 'Large Time Steps', 'Daily', 'Monthly', 'Yearly'])
    temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))

    simulated_daily_return_histogram = go.Histogram(x = price_df['return_mid_price'], nbinsx = 100,  histnorm = 'density', marker_color = 'blue')
    simulated_monthly_return_histogram = go.Histogram(x = resample_price_df_1['return_mid_price'], nbinsx = 20,  histnorm = 'density', marker_color = 'blue')
    simulated_yearly_return_histogram = go.Histogram(x = resample_price_df_2['return_mid_price'], nbinsx = 10,  histnorm = 'density', marker_color = 'blue')

    daily_return_histogram = go.Histogram(x = yahoo_df['return'], nbinsx = 300,  histnorm = 'density', marker_color = 'rgba(0, 128, 0, 0.6)')
    monthly_return_histogram = go.Histogram(x = monthly_df['return'], nbinsx = 30,  histnorm = 'density', marker_color = 'rgba(0, 128, 0, 0.6)')
    yearly_return_histogram = go.Histogram(x = yearly_df['return'], nbinsx = 6,  histnorm = 'density', marker_color = 'rgba(0, 128, 0, 0.6)')


    fig.add_trace(simulated_daily_return_histogram, row = 1, col = 1)
    fig.add_trace(simulated_monthly_return_histogram, row = 1, col = 2)
    fig.add_trace(simulated_yearly_return_histogram, row = 1, col = 3)


    fig.add_trace(daily_return_histogram, row = 2, col = 1)
    fig.add_trace(monthly_return_histogram, row = 2, col = 2)
    fig.add_trace(yearly_return_histogram, row = 2, col = 3)


    fig.update_layout(template = temp,
                        title = 'Stylized Fact - Aggregational Gaussianity',
                        hovermode = 'closest',
                        bargap = 0.05,
                        margin = dict(l = 40, r = 40, t = 80, b = 40),
                        height = 600, 
                        width = 1000, 
                        showlegend = False,
                        xaxis = dict(tickfont=dict(size=10)),  
                        xaxis2 = dict(tickfont=dict(size=10)),  
                        yaxis = dict(side = "left", tickfont = dict(size=10)),
                        xaxis_showgrid = False, 
                        legend = dict(yanchor = "bottom", y = 0.95, xanchor = "left", x = 0.01,  orientation = "h"))

    return fig 