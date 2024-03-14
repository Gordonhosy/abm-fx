import mesa
import tools
import random
from corporates import *
from banks import *
from central_bank import *
from resources import *
from static import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class abmodel(mesa.Model):
    '''
    Generalized model to run the simulation
    '''

    def __init__(self, static_map, all_agents):
        self.static_map = static_map
        self.running = True
        self._steps = 0
        self.all_agents = all_agents
            
        # initiate the mesa grid class
        self.grid = mesa.space.MultiGrid(self.static_map.width, self.static_map.height, torus = False)
        
        # initiate scheduler
        self.schedule = mesa.time.RandomActivationByType(self)
        
        # initate the grid with the static map
        currencyA_map_init = self.static_map.currencyA_map_init
        currencyB_map_init = self.static_map.currencyB_map_init
        
                # -------- initiate data collector -------- 
        model_reporters = {"Step": lambda m: m.schedule.steps}
        agent_reporters = {}
        for _, agent_types in self.all_agents.__dict__.items():

            for agent_type in agent_types:
            
                # Separate corporate agent data collector and central bank data collector
                if agent_type.name == 'Corporate V0':
                    
                    # Number of Corporates / Total Trade Volume / GeoMean(Trade Price) / Trade Network
                    model_reporters[f'({agent_type.name})'] = lambda m: len(m.schedule.agents_by_type[agent_corporate_v0])
                    model_reporters[f'Trade Volume ({agent_type.name})'] = lambda m: sum(len(a.traded_partners) for a in m.schedule.agents_by_type[agent_corporate_v0].values())
                    model_reporters[f'Price ({agent_type.name})'] = lambda m: self.geometric_mean(a.traded_prices for a in m.schedule.agents_by_type[agent_corporate_v0].values())
                
                    # better include pos and trading range for visualization
                    agent_reporters[f'Trade Network ({agent_type.name})'] = lambda a: self.get_trade_partners(a)
                
                if agent_type.name == 'Central Bank V0':
                    
                    # Number of Central Banks / Interest Rate / Inflation Rate
                    model_reporters[f'({agent_type.name})'] = lambda m: len(m.schedule.agents_by_type[self.all_agents.central_banks[0].agent])      
                    model_reporters[f'interest_rate'] = lambda m: tuple([agent.interest_rate for agent in m.schedule.agents if isinstance(agent, self.all_agents.central_banks[0].agent)])                                                       
                    model_reporters[f'inflation_rate'] = lambda m: tuple([agent.inflation_rate for agent in m.schedule.agents if isinstance(agent, self.all_agents.central_banks[0].agent)])
                    model_reporters[f'growth_rate'] = lambda m: tuple([agent.growth_rate for agent in m.schedule.agents if isinstance(agent, self.all_agents.central_banks[0].agent)])                                                       
                    model_reporters[f'target_interest_rate'] = lambda m: tuple([agent.target_interest_rate for agent in m.schedule.agents if isinstance(agent, self.all_agents.central_banks[0].agent)])                                                       
                    model_reporters[f'target_inflation_rate'] = lambda m: tuple([agent.target_inflation_rate for agent in m.schedule.agents if isinstance(agent, self.all_agents.central_banks[0].agent)])

                    
                if (agent_type.name == 'Local Bank') | (agent_type.name == 'International Bank'):
                    
                    model_reporters[f'({agent_type.name})'] = lambda m: len(m.schedule.agents_by_type[agent_bank])
                    model_reporters[f'Trade Volume ({agent_type.name})'] = lambda m: sum(len(a.traded_corps) for a in m.schedule.agents_by_type[agent_bank].values())
                    model_reporters[f'Price ({agent_type.name})'] = lambda m: self.geometric_mean(a.traded_prices for a in m.schedule.agents_by_type[agent_bank].values())
                    model_reporters[f'Hedge Volume ({agent_type.name})'] = lambda m: sum(len(a.hedged_banks) for a in m.schedule.agents_by_type[agent_bank].values())
                    model_reporters[f'Hedge Price ({agent_type.name})'] = lambda m: self.geometric_mean(a.hedged_prices for a in m.schedule.agents_by_type[agent_bank].values())
                
                    # better include pos and trading range for visualization
                    agent_reporters[f'Trade Network ({agent_type.name})'] = lambda a: self.get_trade_partners(a)
                    
                if agent_type.name == 'Arbitrager':
                    
                    model_reporters[f'({agent_type.name})'] = lambda m: len(m.schedule.agents_by_type[agent_arbitrager])
                    model_reporters[f'Trade Volume ({agent_type.name})'] = lambda m: sum(len(a.traded_banks) for a in m.schedule.agents_by_type[agent_arbitrager].values())
                    model_reporters[f'Price ({agent_type.name})'] = lambda m: self.geometric_mean(a.traded_prices for a in m.schedule.agents_by_type[agent_arbitrager].values())
                    
                if agent_type.name == 'Speculator':
                    
                    model_reporters[f'({agent_type.name})'] = lambda m: len(m.schedule.agents_by_type[agent_speculator])
                    model_reporters[f'Trade Volume ({agent_type.name})'] = lambda m: sum(len(a.traded_partners) for a in m.schedule.agents_by_type[agent_speculator].values())
                    model_reporters[f'Price ({agent_type.name})'] = lambda m: self.geometric_mean(a.traded_prices for a in m.schedule.agents_by_type[agent_speculator].values())

        self.datacollector = mesa.DataCollector(model_reporters = model_reporters, agent_reporters = agent_reporters)

        
        agent_id = 0                
        # init resources
        for _, (x,y) in self.grid.coord_iter():
            # init currency A agent
            agent_currencyA = currencyA_basic(agent_id, self, (x,y), currencyA_map_init[x,y])
            self.grid.place_agent(agent_currencyA, (x,y))
            self.schedule.add(agent_currencyA)
            agent_id += 1
            
            # init currency B agent
            agent_currencyB = currencyB_basic(agent_id, self, (x,y), currencyB_map_init[x,y])
            self.grid.place_agent(agent_currencyB, (x,y))
            self.schedule.add(agent_currencyB)
            agent_id += 1

        # init corporate
        for corporate_type in self.all_agents.corporates:
            for _ in range(corporate_type.params.init_population):
                agent_corporate = tools.random_corporate(agent_id, corporate_type, self.static_map, self)
                self.grid.place_agent(agent_corporate, agent_corporate.pos)
                self.schedule.add(agent_corporate)
                agent_id += 1
        
        # init central bank (Two Central Bank)
        for central_bank_type in self.all_agents.central_banks:
            for j in range(central_bank_type.params.number_of_central_bank):

                if j == 0:
                    agent_central_bank = tools.central_bank_A(agent_id, central_bank_type, self.static_map, self)
                    self.grid.place_agent(agent_central_bank, agent_central_bank.pos)
                    self.schedule.add(agent_central_bank)
                    agent_id += 1

                else:
                    agent_central_bank = tools.central_bank_B(agent_id, central_bank_type, self.static_map, self)
                    self.grid.place_agent(agent_central_bank, agent_central_bank.pos)
                    self.schedule.add(agent_central_bank)
                    agent_id += 1

        
        # init local banks
        for bank_type in self.all_agents.banks:
            for init_pos in bank_type.params.init_pos:
                try:
                    agent_rand_bank = tools.random_bank(agent_id, bank_type, init_pos, self)
                    self.grid.place_agent(agent_rand_bank, agent_rand_bank.pos)
                    self.schedule.add(agent_rand_bank)
                    agent_id += 1
                except:
                    print("Fail Position: ", init_pos)

                
        # init arbitragers
        for arb_type in self.all_agents.arbitragers:
            for init_pos in arb_type.params.init_pos:
                agent_rand_arb = tools.random_arb(agent_id, arb_type, init_pos, self)
                self.grid.place_agent(agent_rand_arb, agent_rand_arb.pos)
                self.schedule.add(agent_rand_arb)
                agent_id += 1
                
        # init speculators
        for speculator_type in self.all_agents.speculators:
            for idx, init_pos in enumerate(speculator_type.params.init_pos):
                agent_rand_fund = tools.random_fund(agent_id, speculator_type, init_pos, speculator_type.params.strategies[idx], self)
                self.grid.place_agent(agent_rand_fund, agent_rand_fund.pos)
                self.schedule.add(agent_rand_fund)
                agent_id += 1        
        
        self.corporate_details = self.corporate_class(self)
        self.bank_details = self.bank_class(self)
        self.international_bank_details = self.international_bank_class(self)
        self.arbitrager_details = self.arbitrager_class(self)
        self.speculator_details = self.speculator_class(self)
        
    def geometric_mean(self, listoflist):
        '''
        helper function to calculate geometric mean of price
        '''
        # first flatten the array
        price_list = [item for sublist in listoflist for item in sublist]
        
        if len(price_list) == 0:
            return 
        else:
            return np.exp(np.log(price_list).mean())
    
    def randomise_agents(self, agent_type):
        '''
        helper function to shuffle surviving agents
        '''
        shuffle_agents = [a for a in self.schedule.agents_by_type[agent_type].values()]
        random.shuffle(shuffle_agents)
        return shuffle_agents
    
    def step(self):
        '''
        prompt agents what to do in each step
        '''
        
        growth_rate = [agent.growth_rate for agent in self.schedule.agents if isinstance(agent, self.all_agents.central_banks[0].agent)]
        interest_rate = [agent.interest_rate for agent in self.schedule.agents if isinstance(agent, self.all_agents.central_banks[0].agent)]

        # currency A
        for agent_currencyA in self.schedule.agents_by_type[currencyA_basic].values():
            agent_currencyA.step(growth_rate = growth_rate[0])
        
        # currency B
        for agent_currencyB in self.schedule.agents_by_type[currencyB_basic].values():
            agent_currencyB.step(growth_rate = growth_rate[1])
            
        # corporates
        # randomise move, earn money and trade sequence to make sure not one is advantaged
        for corporate_type in self.all_agents.corporates:
            corporates_shuffle = self.randomise_agents(corporate_type.agent)
            for corporate in corporates_shuffle:
                corporate.traded_prices = []
                corporate.traded_partners = []
                corporate.traded_amount = []
                corporate.move()
                corporate.update_currency_cost(interest_rate_a = interest_rate[0], interest_rate_b =  interest_rate[1]) # update currency cost based on interest rate


            corporates_shuffle = self.randomise_agents(corporate_type.agent)
            for corporate in corporates_shuffle:
                corporate.earn_money()
                corporate.pay_costs()
                corporate.if_bankrupt()
                corporate.put_order()
        
        # hedege funds
        for speculator_type in self.all_agents.speculators:
            speculator_shuffle = self.randomise_agents(speculator_type.agent)
            for speculator in speculator_shuffle:
                speculator.traded_prices = []
                speculator.traded_partners = []
                speculator.traded_amount = []
                speculator.put_order(self)
                
        self.corporate_details.update_ex_trades(self, self.schedule.steps + 1)
        self.speculator_details.update_ex_trades(self, self.schedule.steps + 1)
            
        # banks
        for bank_type in self.all_agents.banks:
            banks_shuffle = self.randomise_agents(bank_type.agent)
            for bank in banks_shuffle:
                bank.traded_prices = []
                bank.traded_corps = []
                bank.traded_amount = []
                bank.hedged_prices = []
                bank.hedged_banks = []
                bank.hedged_amount = []
                bank.arbed_prices = []
                bank.arbed_desks = []
                bank.arbed_amount = []
                bank.trade_with_corporates()
            
            banks_shuffle = self.randomise_agents(bank_type.agent)
            for bank in banks_shuffle:
                bank.hedge_with_banks()
                bank.pay_costs()
                bank.if_bankrupt()
        
        # arbitragers
        for arb_type in self.all_agents.arbitragers:
            arbs_shuffle = self.randomise_agents(arb_type.agent)
            for arb in arbs_shuffle:
                arb.traded_prices = []
                arb.traded_banks = []
                arb.traded_amount = []
                arb.arbitrage_lob()
                # need to update LOB here, else all arbitragers will arb on the same price
                for bank_type in self.all_agents.banks:
                    for bank in [a for a in self.schedule.agents_by_type[bank_type.agent].values()]:
                        bank.update_bid_ask()
                
        
        self.corporate_details.update_trades(self, self.schedule.steps + 1)
        self.bank_details.update(self, self.schedule.steps + 1)
        self.international_bank_details.update(self, self.schedule.steps + 1)
        self.arbitrager_details.update(self, self.schedule.steps + 1)
        self.speculator_details.update_trades(self, self.schedule.steps + 1)
        
        # central banks
        for central_bank_type in self.all_agents.central_banks:
            central_banks_shuffle = self.randomise_agents(central_bank_type.agent)
            for central_bank in central_banks_shuffle:
                central_bank.step()

        self.schedule.steps += 1
        self._steps += 1
        self.datacollector.collect(self)
        
    
    def run_model(self, steps = 1000):
        '''
        helper function to run multiple steps
        '''
        for _ in range(steps):
            self.step()
        
    ##### need to think of how to generalise this, or maybe just dont record this #####
    def get_trade_partners(self, agent):
        '''
        ### THIS IS ONLY FOR CORPORATE V0 ###
        return trade partners of the round for reporting
        '''
        if isinstance(agent, agent_corporate_v0):
            return agent.traded_partners
        else:
            return None
        
    
    class corporate_class():
        '''
        Corporate class to return information of:
        1. all corporates of a certain time step
        2. all time steps of one corporate
        '''
        def __init__(self, model):
            self.agent_type = agent_corporate_v0
            self.agent_id = {}
            self.agent_pos = {}
            self.agent_currencyA = {}
            self.agent_currencyB = {}
            self.agent_quote = {}
            self.agent_traded_price = {}
            self.agent_traded_amount = {}
            self.agent_traded_with = {}
            step = 0
            self.update_ex_trades(model, step)
            self.update_trades(model, step)
    
            
        def update_ex_trades(self, model, step):
            '''
            Update the corporate details of each step before trades (because of the quote)
            '''
            self.agent_id[step] = [a.unique_id for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_pos[step] = [a.pos for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_quote[step] = [(a.trade_direction, a.price, a.amount) for a in model.schedule.agents_by_type[self.agent_type].values()]
            
        def update_trades(self, model, step):
            '''
            Update the corporate trades of each step
            ''' 
            for a in model.schedule.agents_by_type[self.agent_type].values():
                if len(a.traded_prices) != 0:
                    self.agent_type.trade_happened = True
                else:
                    self.agent_type.trade_happened = False
                    
            self.agent_currencyA[step] = [a.currencyA for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_currencyB[step] = [a.currencyB for a in model.schedule.agents_by_type[self.agent_type].values()]                
            self.agent_traded_price[step] = [a.traded_prices for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_traded_amount[step] = [a.traded_amount for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_traded_with[step] = [a.traded_partners for a in model.schedule.agents_by_type[self.agent_type].values()]
            
        def by_agent(self, agent_id):
            '''
            Return time series of a corporate's value, trades and position
            '''
            steps_ts = []
            pos_ts = []
            currencyA_ts = []
            currencyB_ts = []
            quote_ts = []
            price_ts = []
            amount_ts = []
            with_ts = []
            
            for step, agent_ids in self.agent_id.items():
                if agent_id in agent_ids:
                    idx = agent_ids.index(agent_id)
                    steps_ts.append(step)
                    pos_ts.append(self.agent_pos[step][idx])
                    currencyA_ts.append(self.agent_currencyA[step][idx])
                    currencyB_ts.append(self.agent_currencyB[step][idx])
                    quote_ts.append(self.agent_quote[step][idx])
                    price_ts.append(self.agent_traded_price[step][idx])
                    amount_ts.append(self.agent_traded_amount[step][idx])
                    with_ts.append(self.agent_traded_with[step][idx])
                else:
                    break
                    
            return pd.DataFrame({
                        'Step': steps_ts,
                        'Position': pos_ts,
                        'Currency A': currencyA_ts,
                        'Currency B': currencyB_ts,
                        'Quotes': quote_ts,
                        'Traded Price': price_ts,
                        'Traded Amount': amount_ts,
                        'Traded with': with_ts
                       })
  
            
        def by_step(self, step):
            '''
            Return the snapshot of all corporates of a certain timestep
            '''
            return pd.DataFrame({
                        'Agent ID': self.agent_id[step],
                        'Position': self.agent_pos[step],
                        'Currency A': self.agent_currencyA[step],
                        'Currency B': self.agent_currencyB[step],
                        'Quotes': self.agent_quote[step],
                        'Traded Price': self.agent_traded_price[step],
                        'Traded Amount': self.agent_traded_amount[step],
                        'Traded with': self.agent_traded_with[step]
                       })
        
        def all_ids(self):
            '''
            Return all agent ids
            '''
            return self.agent_id[0]
        
        
    class bank_class():
        '''
        Bank class to return information of:
        1. all banks of a certain time step
        2. all time steps of one bank
        '''
        def __init__(self, model):
            self.agent_type = agent_bank
            self.agent_id = {}
            self.agent_pos = {}
            self.agent_currencyA = {}
            self.agent_currencyB = {}
            self.agent_traded_price = {}
            self.agent_traded_amount = {}
            self.agent_traded_with = {}
            self.agent_hedged_price = {}
            self.agent_hedged_amount = {}
            self.agent_hedged_with = {}
            self.agent_arbed_price = {}
            self.agent_arbed_amount = {}
            self.agent_arbed_with = {}
            self.agent_bid_book = {}
            self.agent_ask_book = {}
            step = 0
            self.update(model, step)
    
            
        def update(self, model, step):
            '''
            Update the banks details of each step
            '''
            self.agent_id[step] = [a.unique_id for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_pos[step] = [a.pos for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_currencyA[step] = [a.currencyA for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_currencyB[step] = [a.currencyB for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_traded_price[step] = [a.traded_prices for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_traded_amount[step] = [a.traded_amount for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_traded_with[step] = [a.traded_corps for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_hedged_price[step] = [a.hedged_prices for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_hedged_amount[step] = [a.hedged_amount for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_hedged_with[step] = [a.hedged_banks for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_arbed_price[step] = [a.arbed_prices for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_arbed_amount[step] = [a.arbed_amount for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_arbed_with[step] = [a.arbed_desks for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_bid_book[step] = [a.bid_book for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_ask_book[step] = [a.ask_book for a in model.schedule.agents_by_type[self.agent_type].values()]
            
        def by_agent(self, agent_id):
            '''
            Return time series of a bank's value, trades and position
            '''
            steps_ts = []
            pos_ts = []
            currencyA_ts = []
            currencyB_ts = []
            quote_ts = []
            price_ts = []
            amount_ts = []
            with_ts = []
            hedge_price_ts = []
            hedge_amount_ts = []
            hedge_with_ts = []
            arb_price_ts = []
            arb_amount_ts = []
            arb_with_ts = []            
            
            for step, agent_ids in self.agent_id.items():
                if agent_id in agent_ids:
                    idx = agent_ids.index(agent_id)
                    steps_ts.append(step)
                    pos_ts.append(self.agent_pos[step][idx])
                    currencyA_ts.append(self.agent_currencyA[step][idx])
                    currencyB_ts.append(self.agent_currencyB[step][idx])
                    price_ts.append(self.agent_traded_price[step][idx])
                    amount_ts.append(self.agent_traded_amount[step][idx])
                    with_ts.append(self.agent_traded_with[step][idx])
                    hedge_price_ts.append(self.agent_hedged_price[step][idx])
                    hedge_amount_ts.append(self.agent_hedged_amount[step][idx])
                    hedge_with_ts.append(self.agent_hedged_with[step][idx])
                    arb_price_ts.append(self.agent_arbed_price[step][idx])
                    arb_amount_ts.append(self.agent_arbed_amount[step][idx])
                    arb_with_ts.append(self.agent_arbed_with[step][idx])
                else:
                    break
                    
            return pd.DataFrame({
                        'Step': steps_ts,
                        'Position': pos_ts,
                        'Currency A': currencyA_ts,
                        'Currency B': currencyB_ts,
                        'Traded Price': price_ts,
                        'Traded Amount': amount_ts,
                        'Traded with': with_ts,
                        'Hedged Price': hedge_price_ts,
                        'Hedged Amount': hedge_amount_ts,
                        'Hedged with': hedge_with_ts,
                        'Arbed Price': arb_price_ts,
                        'Arbed Amount': arb_amount_ts,
                        'Arbed with': arb_with_ts
                       })
  
            
        def by_step(self, step):
            '''
            Return the snapshot of all banks of a certain timestep
            '''
            return pd.DataFrame({
                        'Agent ID': self.agent_id[step],
                        'Position': self.agent_pos[step],
                        'Currency A': self.agent_currencyA[step],
                        'Currency B': self.agent_currencyB[step],
                        'Traded Price': self.agent_traded_price[step],
                        'Traded Amount': self.agent_traded_amount[step],
                        'Traded with': self.agent_traded_with[step],
                        'Hedged Price': self.agent_hedged_price[step],
                        'Hedged Amount': self.agent_hedged_amount[step],
                        'Hedged with': self.agent_hedged_with[step],
                        'Arbed Price': self.agent_arbed_price[step],
                        'Arbed Amount': self.agent_arbed_amount[step],
                        'Arbed with': self.agent_arbed_with[step]                
                       })
        
        def all_ids(self):
            '''
            Return all agent ids
            '''
            return self.agent_id[0]
        
        def lob(self, step, agent):
            '''
            Return the limit order book of the interbank market of a certain time step
            '''
            interbank_bid = {}
            for bid_book in self.agent_bid_book[step]:
                for bid in bid_book:
                    if bid[0] in interbank_bid:
                        interbank_bid[bid[0]] += bid[1]
                    else:
                        interbank_bid[bid[0]] = bid[1]
            
            interbank_ask = {}
            for ask_book in self.agent_ask_book[step]:
                for ask in ask_book:
                    if ask[0] in interbank_ask:
                        interbank_ask[ask[0]] += ask[1]
                    else:
                        interbank_ask[ask[0]] = ask[1]
            

            # ----- Plotly Version ------
            fig = make_subplots(rows=1, cols=1)
            temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))
            bid_price = go.Bar(x = list(interbank_bid.keys()), y = list(interbank_bid.values()), name = 'Bid', marker = dict(color = 'darkblue'))
            ask_price = go.Bar(x = list(interbank_ask.keys()), y = list(interbank_ask.values()), name = 'Ask', marker = dict(color = 'darkred'))

            fig.add_trace(bid_price, row = 1, col = 1)
            fig.add_trace(ask_price, row = 1, col = 1)

            fig.update_layout(title_text = 'Limit Order Book of Interbank Market - ' + str(agent), showlegend=True)

            fig.update_layout(template = temp,
                            hovermode = 'closest',
                            margin = dict(l = 30, r = 20, t = 50, b = 20),
                            height = 400, 
                            width = 600, 
                            showlegend = True,
                            xaxis = dict(tickfont=dict(size=10)),
                            yaxis = dict(side = "left", tickfont = dict(size=10)),
                            xaxis_showgrid = False, 
                            legend = dict(yanchor = "bottom", y = 0.9, xanchor = "left", x = 0.01,  orientation="h"))
    
            # ----- Matplotlib Version ------
            # fig = plt.figure()
            # plt.bar(interbank_bid.keys(), interbank_bid.values(), color = 'darkblue', alpha = 0.3, width=0.01, label = 'Bid')
            # plt.bar(interbank_ask.keys(), interbank_ask.values(), color = 'darkred', alpha = 0.4, width=0.01, label = 'Ask')
            # plt.legend()
            # plt.close()
                        
            return fig
        
        def flatten(self, l):
            '''
            Return flatten list
            '''
            return [xs for x in l for xs in x]
        
        def bid_ask_prices(self):
            '''
            Return the vwap price of bid and ask for banks
            '''
            buy_vwap = [] # bid (since the bank is passive, so only buy at bids)
            sell_vwap = [] # ask
            
            for step in self.agent_id.keys():
                traded_price = self.flatten(self.agent_traded_price[step])
                traded_amount = self.flatten(self.agent_traded_amount[step])
                hedged_price = self.flatten(self.agent_hedged_price[step])
                hedged_amount = self.flatten(self.agent_hedged_amount[step])
                arbed_price = self.flatten(self.agent_arbed_price[step])
                arbed_amount = self.flatten(self.agent_arbed_amount[step])
                
                ##### is without hedging price more accurate? #####
                
                prices = np.array(traded_price + hedged_price + arbed_price)
                amounts = np.array(traded_amount + hedged_amount + arbed_amount)
                
                buy_prices = np.where(amounts > 0, prices, 0)
                sell_prices = np.where(amounts < 0, prices, 0)
                buy_amounts = np.where(amounts > 0, amounts, 0)
                sell_amounts = np.where(amounts < 0, amounts, 0)
                
                buy_vwap.append(np.dot(buy_prices, buy_amounts)/sum(buy_amounts))
                sell_vwap.append(np.dot(sell_prices, sell_amounts)/sum(sell_amounts))
                
            return buy_vwap, sell_vwap
                
        
        def price_plot(self):
            '''
            Plots the vwap bid ask time series
            '''
            buy_vwap, sell_vwap = self.bid_ask_prices()
                
            fig = make_subplots(rows=1, cols=1)
            temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size = 12)))
            bid_price = go.Scatter(y = buy_vwap, name = 'bid price', line = dict(color = 'darkblue'))
            ask_price = go.Scatter(y = sell_vwap, name = 'ask price', line = dict(color = 'darkred'))

            fig.add_trace(bid_price)
            fig.add_trace(ask_price)

            fig.update_layout(title_text = 'VWAP Bid/Ask', showlegend=True)
            fig.update_layout(template = temp,
                            hovermode = 'closest',
                            margin = dict(l = 30, r = 20, t = 50, b = 20),
                            height = 400, 
                            width = 600, 
                            showlegend = True,
                            xaxis = dict(tickfont=dict(size=10)),
                            yaxis = dict(side = "left", tickfont = dict(size=10)),
                            xaxis_showgrid = False, 
                            legend = dict(yanchor = "bottom", y = 0.9, xanchor = "left", x = 0.01,  orientation="h"))
            
            # -------- Matplotlib Version -------- 
            # fig = plt.figure()
            # plt.plot(buy_vwap, color = 'darkblue', alpha = 0.3, label = 'Bid')
            # plt.plot(sell_vwap, color = 'darkred', alpha = 0.4, label = 'Ask')
            # plt.legend()
            # plt.close()
            return fig
        
        
    class international_bank_class(bank_class):
        def __init__(self, model):
            super().__init__(model)
            self.agent_type = agent_international_bank
            step = 0
            self.update(model, step)
            
            
    class arbitrager_class():
        '''
        Arbitrager class to return information of:
        1. all banks of a certain time step
        2. all time steps of one arbitrager
        '''
        def __init__(self, model):
            self.agent_type = agent_arbitrager
            self.agent_id = {}
            self.agent_pos = {}
            self.agent_currencyA = {}
            self.agent_currencyB = {}
            self.agent_traded_price = {}
            self.agent_traded_amount = {}
            self.agent_traded_with = {}
            self.agent_hedged_price = {}
            self.agent_hedged_amount = {}
            self.agent_hedged_with = {}
            self.agent_bid_book = {}
            self.agent_ask_book = {}
            step = 0
            self.update(model, step)
    
            
        def update(self, model, step):
            '''
            Update the arbitragers details of each step
            '''
            self.agent_id[step] = [a.unique_id for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_pos[step] = [a.pos for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_currencyA[step] = [a.currencyA for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_currencyB[step] = [a.currencyB for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_traded_price[step] = [a.traded_prices for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_traded_amount[step] = [a.traded_amount for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_traded_with[step] = [a.traded_banks for a in model.schedule.agents_by_type[self.agent_type].values()]

            
        def by_agent(self, agent_id):
            '''
            Return time series of a arbitrager's value, trades and position
            '''
            steps_ts = []
            pos_ts = []
            currencyA_ts = []
            currencyB_ts = []
            quote_ts = []
            price_ts = []
            amount_ts = []
            with_ts = []
            
            for step, agent_ids in self.agent_id.items():
                if agent_id in agent_ids:
                    idx = agent_ids.index(agent_id)
                    steps_ts.append(step)
                    pos_ts.append(self.agent_pos[step][idx])
                    currencyA_ts.append(self.agent_currencyA[step][idx])
                    currencyB_ts.append(self.agent_currencyB[step][idx])
                    price_ts.append(self.agent_traded_price[step][idx])
                    amount_ts.append(self.agent_traded_amount[step][idx])
                    with_ts.append(self.agent_traded_with[step][idx])
                else:
                    break
                    
            return pd.DataFrame({
                        'Step': steps_ts,
                        'Position': pos_ts,
                        'Currency A': currencyA_ts,
                        'Currency B': currencyB_ts,
                        'Traded Price': price_ts,
                        'Traded Amount': amount_ts,
                        'Traded with': with_ts,
                       })
  
            
        def by_step(self, step):
            '''
            Return the snapshot of all banks of a certain timestep
            '''
            return pd.DataFrame({
                        'Agent ID': self.agent_id[step],
                        'Position': self.agent_pos[step],
                        'Currency A': self.agent_currencyA[step],
                        'Currency B': self.agent_currencyB[step],
                        'Traded Price': self.agent_traded_price[step],
                        'Traded Amount': self.agent_traded_amount[step],
                        'Traded with': self.agent_traded_with[step],
                       })
        
        def all_ids(self):
            '''
            Return all agent ids
            '''
            return self.agent_id[0]
        
        
    class speculator_class():
        '''
        Speculator class to return information of:
        1. all speculators of a certain time step
        2. all time steps of one speculator
        '''
        def __init__(self, model):
            self.agent_type = agent_speculator
            self.agent_id = {}
            self.agent_pos = {}
            self.agent_currencyA = {}
            self.agent_currencyB = {}
            self.agent_quote = {}
            self.agent_traded_price = {}
            self.agent_traded_amount = {}
            self.agent_traded_with = {}
            step = 0
            self.update_ex_trades(model, step)
            self.update_trades(model, step)
    
            
        def update_ex_trades(self, model, step):
            '''
            Update the speculator details of each step before trades (because of the quote)
            '''
            self.agent_id[step] = [a.unique_id for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_pos[step] = [a.pos for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_quote[step] = [(a.trade_direction, a.price, a.amount) for a in model.schedule.agents_by_type[self.agent_type].values()]
            
        def update_trades(self, model, step):
            '''
            Update the speculator trades of each step
            ''' 
            for a in model.schedule.agents_by_type[self.agent_type].values():
                if len(a.traded_prices) != 0:
                    self.agent_type.trade_happened = True
                else:
                    self.agent_type.trade_happened = False

            self.agent_currencyA[step] = [a.currencyA for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_currencyB[step] = [a.currencyB for a in model.schedule.agents_by_type[self.agent_type].values()]                    
            self.agent_traded_price[step] = [a.traded_prices for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_traded_amount[step] = [a.traded_amount for a in model.schedule.agents_by_type[self.agent_type].values()]
            self.agent_traded_with[step] = [a.traded_partners for a in model.schedule.agents_by_type[self.agent_type].values()]
            
        def by_agent(self, agent_id):
            '''
            Return time series of a speculator's value, trades and position
            '''
            steps_ts = []
            pos_ts = []
            currencyA_ts = []
            currencyB_ts = []
            quote_ts = []
            price_ts = []
            amount_ts = []
            with_ts = []
            
            for step, agent_ids in self.agent_id.items():
                if agent_id in agent_ids:
                    idx = agent_ids.index(agent_id)
                    steps_ts.append(step)
                    pos_ts.append(self.agent_pos[step][idx])
                    currencyA_ts.append(self.agent_currencyA[step][idx])
                    currencyB_ts.append(self.agent_currencyB[step][idx])
                    quote_ts.append(self.agent_quote[step][idx])
                    price_ts.append(self.agent_traded_price[step][idx])
                    amount_ts.append(self.agent_traded_amount[step][idx])
                    with_ts.append(self.agent_traded_with[step][idx])
                else:
                    break
                    
            return pd.DataFrame({
                        'Step': steps_ts,
                        'Position': pos_ts,
                        'Currency A': currencyA_ts,
                        'Currency B': currencyB_ts,
                        'Quotes': quote_ts,
                        'Traded Price': price_ts,
                        'Traded Amount': amount_ts,
                        'Traded with': with_ts
                       })
  
            
        def by_step(self, step):
            '''
            Return the snapshot of all corporates of a certain timestep
            '''
            return pd.DataFrame({
                        'Agent ID': self.agent_id[step],
                        'Position': self.agent_pos[step],
                        'Currency A': self.agent_currencyA[step],
                        'Currency B': self.agent_currencyB[step],
                        'Quotes': self.agent_quote[step],
                        'Traded Price': self.agent_traded_price[step],
                        'Traded Amount': self.agent_traded_amount[step],
                        'Traded with': self.agent_traded_with[step]
                       })
        
        def all_ids(self):
            '''
            Return all agent ids
            '''
            return self.agent_id[0]

# if __name__ == '__main__':

#     steps = 300
#     model = abmodel(static_map_v0(), all_agents())
#     model.run_model(steps)
#     model_results = model.datacollector.get_model_vars_dataframe()
    
#     print(model_results)