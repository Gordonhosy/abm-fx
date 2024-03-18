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
        self.rest = 0
        self.amount_placed = None
        self.aggressive = 0.01
        self.target_execution = 0.8
        if strat == 'momentum':
            self.ma, self.sd = self.random_ma()
        
        
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
        if self.strategy == 'momentum':
            self.trade_direction, self.amount, self.price = self.strategy_momentum(model)
        # elif self.strategy == 'strategy 3':
            #self.trade_direction, self.amount, self.price = self.strategy3(model)

    def random_ma(self):
        '''
        Function for randomising moving average related strategies
        '''
        ma = [(1, 5), (3, 10), (5, 20)] #, (10, 60), (20, 120)
        sd = [0.5, 1, 1.25] #, 1.5, 2
        return random.choice(ma), random.choice(sd)
    
    def adjust_aggressiveness(self):
        '''
        Function to adjust aggressiveness for placing orders
        '''
        if self.amount_placed is None:
            return
        else:
            if self.amount is None:
                delta = self.target_execution - (self.amount_placed - 0)/self.amount_placed # how many % of order is executed
                self.aggressive = max(self.aggressive * (1 + delta), 0.01)
            else:
                delta = self.target_execution - (self.amount_placed - self.amount)/self.amount_placed # how many % of order is executed
                self.aggressive = max(self.aggressive * (1 + delta), 0.01)
            
    def strategy_momentum(self, model):
        '''
        Momentum strategy to buy when there is upward trend, and short when there is downward trend
        '''
        # historical market prices
        if self.rest == 0:
            bids, asks = model.bank_details.top_of_book()
            bids.pop(0)
            asks.pop(0)
            step = model.schedule.steps + 1

            # strategy: buy/sell when price crosses the Bollinger Bands
            if step < self.ma[1]:
                self.amount_placed = None
                return None, None, None
            else:
                mid = [(x + y)/2 for x, y in zip(bids, asks)]
                long_window = mid[-self.ma[1]:]
                long_ma = np.mean(long_window)
                long_std = np.std(long_window)
                short_ma = np.mean(mid[-self.ma[0]:])

                if short_ma > long_ma + self.sd*long_std:
                    leverage = ((short_ma - long_ma) / long_std) / self.sd
                    bet_size = (self.currencyA + (self.currencyB/mid[-1])) * 0.1
                    if (int(bet_size * leverage) == 0) | (int(bet_size * leverage) * asks[-1]*(1 + self.aggressive) > self.currencyB): # not enough currency B
                        self.amount_placed = None
                        return None, None, None
                    else:
                        self.rest = self.ma[0] - 1
                        self.amount_placed = int(bet_size * leverage)
                        return 'long', int(bet_size * leverage), round(asks[-1]*(1 + self.aggressive), 2)
                elif short_ma < long_ma - self.sd*long_std:
                    leverage = ((long_ma - short_ma) / long_std) / self.sd
                    bet_size = (self.currencyA + (self.currencyB/mid[-1])) * 0.1
                    if (int(bet_size * leverage) == 0) | (int(bet_size * leverage) > self.currencyA): # not enough currency A
                        self.amount_placed = None
                        return None, None, None 
                    else:
                        self.rest = self.ma[0] - 1
                        self.amount_placed = int(bet_size * leverage)
                        return 'short', int(bet_size * leverage), round(bids[-1]*(1 - self.aggressive), 2)
                else:
                    self.amount_placed = None
                    return None, None, None
                
        # after a trade, rest for certain time until the next trade can happen
        else:
            self.rest -= 1
            self.amount_placed = None
            return None, None, None
            
        