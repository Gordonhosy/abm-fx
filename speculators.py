import mesa
from resources import *
import numpy as np
import random
import math

class agent_speculator(mesa.Agent):
    '''
    Speculator:
    - Aims to buy low and sell high, attempting to beat the market
    - Formulate different strategies
    '''
    
    def __init__(self, agent_id, model, pos, moore, currencyA, currencyB, cost_currencyA, cost_currencyB, strat):
        super().__init__(agent_id, model)
        self.pos = pos
        self.moore = moore
        self.currencyA = currencyA
        self.currencyB = currencyB
        self.cost_currencyA = cost_currencyA
        self.cost_currencyB = cost_currencyB
        self.traded_prices = []
        self.traded_partners = []
        self.traded_amount = []
        self.trade_direction = None
        self.amount = None
        self.price = None
        self.strategy = strat
        
        
    def pay_costs(self):
        '''
        Function for speculator to pay money each step
        '''
        self.currencyA -= self.cost_currencyA
        self.currencyB -= self.cost_currencyB
     
    
    def if_bankrupt(self):
        '''
        Function to check if someone is bankrupted, and remove the agent
        '''
        if (self.currencyA < 0) | (self.currencyB < 0):
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)


    def put_order(self, model):
        '''
        Function for speculator to put orders, based on certain strategy
        The aim is buy low and sell high
        '''
        if self.strategy == 'sample':
            self.trade_direction, self.amount, self.price = self.strategy_sample(model)
        # elif self.strategy == 'strategy 2':
            #self.trade_direction, self.amount, self.price = self.strategy2(model)
        # elif self.strategy == 'strategy 3':
            #self.trade_direction, self.amount, self.price = self.strategy3(model)

            
    def strategy_sample(self, model):
        '''
        Sample strategy to long in odd rounds and short in even rounds
        '''
        # historical market prices
        bids, asks = model.bank_details.bid_ask_prices()
        step = model.schedule.steps + 1
        
        # strategy: attempt to long in odd rounds at last vwap ask, vice versa
        if step == 1:
            return None, None, None # no previous on first round
        if step % 2 == 1:
            if np.isnan(asks[-1]):
                return None, None, None
            else:
                return 'long', 10, round(asks[-1], 2)
        else:
            if np.isnan(bids[-1]):
                return None, None, None
            else:
                return 'short', 10, round(bids[-1], 2)
        