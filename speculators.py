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
        self.missing_amount = 0
        self.missing_direction = 0
        # strategy specific parameters
        if (strat == 'momentum') | (strat == 'mean revert'):
            self.ma, self.sd = self.random_ma()
            self.target_execution = 0.8
        if strat == 'uncoveredIR':
            self.borrow_A = 0
            self.borrow_B = 0
            self.target_execution = 1.0
            
        
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
        elif self.strategy == 'mean revert':
            self.trade_direction, self.amount, self.price = self.strategy_mean_revert(model)
        elif self.strategy == 'uncoveredIR':
            self.trade_direction, self.amount, self.price = self.strategy_uncoveredIR(model)    

    def random_ma(self):
        '''
        Function for randomising moving average related strategies
        '''
        ma = [(1, 5), (3, 10), (5, 20), (10, 60)] #, (20, 120)
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
                self.missing_amt = 0
                self.amount_placed = None
                
            else:
                delta = self.target_execution - (self.amount_placed - self.amount)/self.amount_placed # how many % of order is executed
                self.aggressive = max(self.aggressive * (1 + delta), 0.01)
                self.missing_amount = self.amount
                self.amount_placed = None
                
            
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
            
            
    def strategy_mean_revert(self, model):
        '''
        Mean reverting strategy to buy when there is downward trend, and short when there is upward trend
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

                if short_ma < long_ma - self.sd*long_std:
                    leverage = ((long_ma - short_ma) / long_std) / self.sd
                    bet_size = (self.currencyA + (self.currencyB/mid[-1])) * 0.1
                    if (int(bet_size * leverage) == 0) | (int(bet_size * leverage) * asks[-1]*(1 + self.aggressive) > self.currencyB): # not enough currency B
                        self.amount_placed = None
                        return None, None, None
                    else:
                        self.rest = self.ma[0] - 1
                        self.amount_placed = int(bet_size * leverage)
                        return 'long', int(bet_size * leverage), round(asks[-1]*(1 + self.aggressive), 2)
                    
                elif short_ma > long_ma + self.sd*long_std:
                    leverage = ((short_ma - long_ma) / long_std) / self.sd
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
    
    
    def strategy_uncoveredIR(self, model):
        '''
        Uncovered interest rate strategy to benefit from interest rate differentials, betting that the future spot FX does not decline (for A vs B)
        '''
        if step == 0:
            return None, None, None
        else:
            # first compute the interest gains
            central_banks = [agent for agent in model.schedule.agents if isinstance(agent, model.all_agents.central_banks[0].agent)]
            interest_rate = [agent.interest_rate for agent in central_banks]
            self.currencyA += -self.borrowA * interest_rate[0] / 252
            self.currencyB += -self.borrowB * interest_rate[1] / 252
            central_bank[0].currencyA += self.borrowA * interest_rate[0] / 252 ### need to change attribute name ###
            central_bank[1].currencyB += self.borrowB * interest_rate[1] / 252 ### need to change attribute name ###
            
            # see if there are trading opportunities
            ir_diff = interest_rate[0] - interest_rate[1]
            bids, asks = model.bank_details.top_of_book()
            last_mid = bids[-1] + asks[-1]
            if ir_diff > 0.01:
                # sizing
                target_position = self.ir_diff * 10 * (self.currencyA + (self.currencyB/last_mid)) # amount in terms of currency A
                target_in_B = target_position * last_mid
                # need to increase borrow B to buy A
                if (self.borrow_B == 0) | (target_in_B/self.borrow_B > 1.2):
                    # borrow currency B from central bank B
                    add_borrow = int(target_in_B - self.borrow_B)
                    self.borrow_B += add_borrow
                    central_bank[1].currencyB -= add_borrow ### need to change attribute name ###
                    central_bank[1].lend += add_borrow ### need to change attribute name ###
                    
                    # buy treasury in currency A from central bank A
                    self.borrow_A -= int(add_borrow/last_mid) # negative borrow means lend
                    central_bank[0].currencyA += int(add_borrow/last_mid) ### need to change attribute name ###
                    central_bank[0].lend -= int(add_borrow/last_mid) ### need to change attribute name ###
                    
                    total_amount = int(add_borrow/last_mid) + self.missing_direction * self.missing_amount
                    self.amount_placed = total_amount
                    self.missing_direction = 1
                    return 'long', total_amount, round(asks[-1]*(1 + self.aggressive), 2)
                
                elif target_in_B/self.borrow_B < 0.8:
                    reduce_borrow = int(self.borrow_B - target_in_B)
                    self.borrow_B -= reduce_borrow
                    central_bank[1].currencyB += reduce_borrow ### need to change attribute name ###
                    central_bank[1].lend -= reduce_borrow ### need to change attribute name ###
                    
                    self.borrow_A += int(reduce_borrow/last_mid) # negative borrow means lend
                    central_bank[0].currencyA -= int(reduce_borrow/last_mid) ### need to change attribute name ###
                    central_bank[0].lend += int(reduce_borrow/last_mid) ### need to change attribute name ###
                    
                    total_amount = int(reduce_borrow/last_mid) - self.missing_direction * self.missing_amount
                    self.amount_placed = total_amount
                    self.missing_direction = -1
                    return 'short', total_amount, round(bids[-1]*(1 - self.aggressive), 2)
                    
                else:
                    if self.missing_amount > 0:
                        if self.missing_direction == 1:
                            return 'long', self.missing_amount, round(asks[-1]*(1 + self.aggressive), 2)
                        else:
                            return 'short', self.missing_amount, round(bids[-1]*(1 - self.aggressive), 2)
                
            elif ir_diff < -0.01:
                # sizing
                target_position = -self.ir_diff * 10 * (self.currencyA + (self.currencyB/last_mid)) # amount in terms of currency A
                target_in_A = target_position
                # need to increase borrow A buy B
                if (self.borrow_A == 0) | (target_in_A/self.borrow_A > 1.2):
                    # borrow currency A from central bank A
                    add_borrow = int(target_in_A - self.borrow_A)
                    self.borrow_A += add_borrow
                    central_bank[0].currencyA -= add_borrow ### need to change attribute name ###
                    central_bank[0].lend += add_borrow ### need to change attribute name ###
                    
                    # buy treasury in currency A from central bank A
                    self.borrow_B -= int(add_borrow*last_mid) # negative borrow means lend
                    central_bank[1].currencyB += int(add_borrow*last_mid) ### need to change attribute name ###
                    central_bank[1].lend -= int(add_borrow*last_mid) ### need to change attribute name ###
                    
                    total_amount = int(add_borrow*last_mid) - self.missing_direction * self.missing_amount
                    self.amount_placed = total_amount
                    self.missing_direction = -1
                    return 'short', total_amount, round(bids[-1]*(1 - self.aggressive), 2)
                
                elif target_in_A/self.borrow_A < 0.8:
                    reduce_borrow = int(self.borrow_A - target_in_A)
                    self.borrow_A -= reduce_borrow
                    central_bank[0].currencyA += reduce_borrow ### need to change attribute name ###
                    central_bank[0].lend -= reduce_borrow ### need to change attribute name ###
                    
                    self.borrow_B += int(reduce_borrow*last_mid) # negative borrow means lend
                    central_bank[1].currencyB -= int(reduce_borrow*last_mid) ### need to change attribute name ###
                    central_bank[1].lend += int(reduce_borrow*last_mid) ### need to change attribute name ###
                    
                    total_amount = int(reduce_borrow*last_mid) + self.missing_direction * self.missing_amount
                    self.amount_placed = total_amount
                    self.missing_direction = 1
                    return 'long', total_amount, round(asks[-1]*(1 + self.aggressive), 2)
                    
                else:
                    if self.missing_amount > 0:
                        if self.missing_direction == 1:
                            return 'long', self.missing_amount, round(asks[-1]*(1 + self.aggressive), 2)
                        else:
                            return 'short', self.missing_amount, round(bids[-1]*(1 - self.aggressive), 2)
                    
            else:
                if self.missing_amount > 0:
                    if self.missing_direction == 1:
                        return 'long', self.missing_amount, round(asks[-1]*(1 + self.aggressive), 2)
                    else:
                        return 'short', self.missing_amount, round(bids[-1]*(1 - self.aggressive), 2)