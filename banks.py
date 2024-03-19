import mesa
from resources import *
import numpy as np
import random
import math
from corporates import *
from speculators import *

class agent_bank(mesa.Agent):
    '''
    This is the first version of local bank
    Local Bank:
    - The utility of a bank is determined by the Cobb-Douglas utility function
    - The local bank quotes a LOB based on its utility
    - Trade with corporates if corporates ask for price on the LOB
    - At the same time trade with speculators if they quote a price to trade
    - Hedge remaining risk with other banks
    '''
    
    def __init__(self, agent_id, model, pos, moore, currencyA, currencyB, cost_currencyA, cost_currencyB, vision):
        super().__init__(agent_id, model)
        self.pos = pos
        self.moore = moore
        self.currencyA = currencyA
        self.currencyB = currencyB
        self.cost_currencyA = cost_currencyA
        self.cost_currencyB = cost_currencyB
        self.vision = vision
        self.traded_prices = []
        self.traded_corps = []
        self.traded_amount = []
        self.hedged_prices = []
        self.hedged_banks = []
        self.hedged_amount = []
        self.arbed_prices = []
        self.arbed_desks = []
        self.arbed_amount = []
        self.premium = 0.2
        self.bid_book = [] # list of tuples (price, volume)
        self.ask_book = []

    
    def is_occupied_corporates(self, pos):
        '''
        Function to check if a cell is occupied by a corporate
        '''
        if pos == self.pos:
            return False
        
        # check if this cell includes something that is mutually exclusive
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            
            ##### Need to amend the agent corporate later, if there are more corporate types ##### 
            if isinstance(obj, agent_corporate_v0):
                return True
        return False
    
    def is_occupied_speculators(self, pos):
        '''
        Function to check if a cell is occupied by a speculator
        '''
        if pos == self.pos:
            return False
        
        # check if this cell includes something that is mutually exclusive
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            if isinstance(obj, agent_speculator):
                return True
        return False
    
    
    def is_occupied_banks(self, pos):
        '''
        Function to check if a cell is occupied by a bank
        '''
        if pos == self.pos:
            return False
        
        # check if this cell includes something that is mutually exclusive
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            
            if isinstance(obj, agent_bank):
                return True
        return False    
    
    
    def calculate_utility(self, currencyA, currencyB):
        '''
        Function to calculate utility
        '''
        # assume we use Cobb-Douglas
        cost_total = self.cost_currencyA + self.cost_currencyB
        return (currencyA**(self.cost_currencyA/cost_total)) * (currencyB**(self.cost_currencyB/cost_total))   
    
    
    def get_corporates(self, pos):
        '''
        Return the corporate agent of the postition if any
        '''
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            if isinstance(obj, agent_corporate_v0):
                return obj
        return None
    

    def get_speculators(self, pos):
        '''
        Return the speculator agent of the postition if any
        '''
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            if isinstance(obj, agent_speculator):
                return obj
        return None

        
    def get_banks(self, pos):
        '''
        Return the bank agent of the postition if any
        '''
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            if isinstance(obj, agent_bank):
                return obj
        return None
        
        
    def pay_costs(self, interest_rate_a, interest_rate_b):
        '''
        Function for banks to pay money each step
        '''
        self.currencyA -= self.cost_currencyA * (1 + interest_rate_a)
        self.currencyB -= self.cost_currencyB * (1 + interest_rate_b)
     
    
    def if_bankrupt(self):
        '''
        Function to check if someone is bankrupted, and remove the agent
        '''
        if (self.currencyA < 0) | (self.currencyB < 0):
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
    
    
    def calculate_MRS(self):
        return (self.currencyB / self.cost_currencyB) / (self.currencyA / self.cost_currencyA)
    
    
    def sell_currencyB(self, opponent, currencyA, currencyB):
        '''
        Function to execute a trade
        '''
        self.currencyA += currencyA
        opponent.currencyA -= currencyA
        self.currencyB -= currencyB
        opponent.currencyB += currencyB
        
    
    def maybe_sell_currencyB(self, opponent, price, utility_self, utility_opponent):
        '''
        Function to see if both are better off after a trade
        '''
        if price >= 1:
            currencyA_exchange = 1
            currencyB_exchange = round(price, 2)
        else:
            currencyA_exchange = round(1/price, 2)
            currencyB_exchange = 1
        
        new_self_currencyA = self.currencyA + currencyA_exchange
        new_opponent_currencyA = opponent.currencyA - currencyA_exchange
        new_self_currencyB = self.currencyB - currencyB_exchange
        new_opponent_currencyB = opponent.currencyB + currencyB_exchange
        
        # double check if both have enough currencies
        if ((new_self_currencyA < 0) | (new_opponent_currencyA < 0)) | ((new_self_currencyB < 0) | (new_opponent_currencyB < 0)):
            return False
        
        # check if both are better off
        both_better = (self.calculate_utility(new_self_currencyA, new_self_currencyB) > utility_self) \
                        & (self.calculate_utility(new_opponent_currencyA, new_opponent_currencyB) > utility_opponent)
        
        # check if MRS are not crossing
        mrs_not_cross = self.calculate_MRS() > opponent.calculate_MRS()
        
        if (both_better) & (mrs_not_cross):
            self.sell_currencyB(opponent, currencyA_exchange, currencyB_exchange)
            return True
        else:
            return False
    
    
    def trade(self, opponent):
        '''
        Function to see if a trade is beneficial
        '''

        assert self.currencyA >= 0
        assert self.currencyB >= 0
        assert opponent.currencyA >= 0
        assert opponent.currencyB >= 0

        
        # calculate MRS
        mrs_self = self.calculate_MRS()
        mrs_opponent = opponent.calculate_MRS()
        
        # calculate utility
        utility_self = self.calculate_utility(self.currencyA, self.currencyB)
        utility_opponent = opponent.calculate_utility(opponent.currencyA, opponent.currencyB)
        
        # do not trade if MRS are similar
        if math.isclose(mrs_self, mrs_opponent):
            return
        
        # calculate price 
        price = np.sqrt(mrs_self*mrs_opponent)
        
        if (math.isinf(price)) | (math.isclose(price,0)):
            return
    
        if mrs_self > mrs_opponent:
            # self want to buy currency A and sell currency B
            sold = self.maybe_sell_currencyB(opponent, price, utility_self, utility_opponent)
            direction = 1 # self long
            # if criteria not met
            if not sold:
                return
        else:
            # self want to sell currency A and buy currency B
            sold = opponent.maybe_sell_currencyB(self, price, utility_opponent, utility_self)
            direction = -1 # self short
            if not sold:
                return
        
        # store trades
        self.hedged_prices.append(round(price, 2))
        self.hedged_banks.append(opponent.unique_id)
        if price >= 1:
            self.hedged_amount.append(direction*1)
        else:
            self.hedged_amount.append(round(direction*1/price, 2))
        
        self.trade(opponent)
    
    
    def hedge_with_banks(self):
        '''
        Function to hedge with other banks
        '''
        neighbor_banks = [self.get_banks(pos) for pos in self.model.grid.get_neighborhood(self.pos, self.moore, True, self.vision) if self.is_occupied_banks(pos)]
        
        if len(neighbor_banks) == 0:
            return [], []
        
        for opponent in neighbor_banks:
            if opponent:
                self.trade(opponent)
        return
    

    def trade_with_corps_funds(self):
        '''
        Function to trade with other banks
        '''
        self.update_bid_ask()
                
        neighbor_corporates = [self.get_corporates(pos) for pos in self.model.grid.get_neighborhood(self.pos, self.moore, True, self.vision) if self.is_occupied_corporates(pos)]
        neighbor_speculators = [self.get_speculators(pos) for pos in self.model.grid.get_neighborhood(self.pos, self.moore, True, self.vision) if self.is_occupied_speculators(pos)]
        
        neighbors = neighbor_corporates + neighbor_speculators
        
        if len(neighbors) == 0:
            return [], []
        
        random.shuffle(neighbors)
        
        for opponent in neighbors:
            if opponent:
                if (opponent.amount is None) | ((opponent.price is None) | (opponent.trade_direction is None) | (opponent.amount == 0)): # the corporate/speculator does not want to trade
                    pass
                else:
                    self.trade_LOB(opponent)
                    self.update_bid_ask()
        return
    
    
    def trade_LOB(self, opponent):
        '''
        Function to trade with opponent
        '''
        if opponent.trade_direction == 'long':
            # bank is not able to quote an ask price
            if len(self.ask_book) == 0:
                return
            # corporate wants untradeable price
            if opponent.price < self.ask_book[0][0]:
                return
            # corporate wants tradeable price
            else:
                # if vwap is lower than opponent price, bank even more better off        
                vwap = self.calc_vwap(opponent.amount, side = 'ask')
                if vwap is None:
                    return
                elif vwap <= opponent.price:
                    self.execute_sell(opponent, opponent.amount)
                else:
                    # execute as much as possible
                    self.execute_sell(opponent, self.calc_part_amount(opponent.price, side = 'ask'))
                
        elif opponent.trade_direction == 'short':
            # bank is not able to quote a bid price
            if len(self.bid_book) == 0:
                return
            # corporate wants untradeable price
            if opponent.price > self.bid_book[0][0]:
                return
            # corporate wants tradeable price
            else:
                # if vwap is higher than opponent price, bank even more better off        
                vwap = self.calc_vwap(opponent.amount, side = 'bid')
                if vwap is None:
                    return
                elif vwap >= opponent.price:
                    self.execute_buy(opponent, opponent.amount)
                else:
                    # execute as much as possible
                    self.execute_buy(opponent, self.calc_part_amount(opponent.price, side = 'bid'))
                    
            
    def calc_vwap(self, amount, side):
        '''
        calculate the vwap price for a certain amount
        '''
        if side == 'ask':
            book = self.ask_book
        elif side == 'bid':
            book = self.bid_book
            
        volume_required = amount
        vwap = 0
        tradeable = False
        for idx, price_volume in enumerate(book):
            volume_required -= price_volume[1]
            if volume_required <= 0:
                # dont need to take the whole bar 
                vwap += ((price_volume[1] + volume_required)/amount)*price_volume[0]    
                tradeable = True
                break
            vwap += (price_volume[1]/amount)*price_volume[0]
        if tradeable:
            return vwap
        # volume is too much for the bank to handle
        else:
            return None
        
        
    def execute_sell(self, opponent, amount):
        '''
        Execute bank sell A/B, corporate buy A/B
        '''
        self.currencyA -= amount
        self.currencyB += amount*opponent.price
        opponent.currencyA += amount
        opponent.currencyB -= amount*opponent.price
        
        self.traded_prices.append(opponent.price)
        self.traded_corps.append(opponent.unique_id)
        self.traded_amount.append(-amount)
        
        opponent.traded_amount.append(amount)
        opponent.traded_prices.append(opponent.price)
        opponent.traded_partners.append(self.unique_id)
        opponent.amount -= amount
        if math.isclose(opponent.amount, 0):
            opponent.amount = None
        
        
        
    def execute_buy(self, opponent, amount):
        '''
        Execute bank buy A/B, corporate sell A/B
        '''
        self.currencyA += amount
        self.currencyB -= amount*opponent.price
        opponent.currencyA -= amount
        opponent.currencyB += amount*opponent.price
        
        self.traded_prices.append(opponent.price)
        self.traded_corps.append(opponent.unique_id)
        self.traded_amount.append(amount)
        
        opponent.traded_amount.append(-amount)
        opponent.traded_prices.append(opponent.price)
        opponent.traded_partners.append(self.unique_id)
        opponent.amount -= amount
        if math.isclose(opponent.amount, 0):
            opponent.amount = None
            
        
    def calc_part_amount(self, price, side):
        '''
        Calculate the amount that could be execute up to a certain price
        
        Just implemented simplier version here, summing up to the second last possible bar
        '''
        vwap = 0
        amount = 0
        if side == 'ask':
            book = self.ask_book # ascending
            for idx, price_volume in enumerate(book):
                vwap += price_volume[0] * price_volume[1]
                amount += price_volume[1]
                if vwap/amount > price:
                    vwap -= price_volume[0]*price_volume[1]
                    amount -= price_volume[1]
                    break
            return amount
    
        elif side == 'bid':
            book = self.bid_book # descending
            for idx, price_volume in enumerate(book):
                vwap += price_volume[0] * price_volume[1]
                amount += price_volume[1]
                if vwap/amount < price:
                    vwap -= price_volume[0]*price_volume[1]
                    amount -= price_volume[1]
                    break
            return amount

    
    
    def calc_bid_ask(self):
        '''
        Calculate the bid ask the bank can offer under the same utility
        '''
        
        # calculate the combination of currency A and B that gives the same utility
        cost_total = self.cost_currencyA + self.cost_currencyB
        utility = self.currencyA**(self.cost_currencyA/cost_total) * (self.currencyB**(self.cost_currencyB/cost_total))
        indiff_currencyA = np.arange(int(self.currencyA*0.8), int(self.currencyA*1.2))
        indiff_currencyB = (utility/(indiff_currencyA**(self.cost_currencyA/cost_total)))**(cost_total/self.cost_currencyB)
        
        # add premium for banks to earn
        premium_currencyB = np.where(indiff_currencyA - self.currencyA > 0, abs(indiff_currencyB - self.currencyB)*(1-self.premium), abs(indiff_currencyB - self.currencyB)*(1+self.premium))
        # calculate the prices that gives the bank a slightly higher utility (equivalent to the vwap price)
        indiff_bid = [y/x for x, y in zip(indiff_currencyA - self.currencyA, premium_currencyB) if x > 0]
        indiff_ask = [y/abs(x) for x, y in zip(indiff_currencyA - self.currencyA, premium_currencyB) if x < 0]
        indiff_ask.reverse()
        
        # translate the vwap ladders to the normal bid-ask ladders
        bid_book_dict = {}
        for idx, vwap in enumerate(indiff_bid):
            if idx == 0:
                bid_book_dict[round(vwap, 2)] = 1
            else:
                price = round(((idx+1)*vwap - idx*indiff_bid[idx - 1]), 2)
                if price in bid_book_dict:
                    bid_book_dict[price] += 1
                else:
                    bid_book_dict[price] = 1
        
        ask_book_dict = {}
        for idx, vwap in enumerate(indiff_ask):
            if idx == 0:
                ask_book_dict[round(vwap, 2)] = 1
            else:
                price = round(((idx+1)*vwap - idx*indiff_ask[idx - 1]), 2)
                if price in ask_book_dict:
                    ask_book_dict[price] += 1
                else:
                    ask_book_dict[price] = 1
        
        return list(bid_book_dict.items()), list(ask_book_dict.items())
    
    
    def update_bid_ask(self):
        '''
        Update the bid ask based on the current inventory
        '''
        self.bid_book, self.ask_book = self.calc_bid_ask()

    def increase_costs(self, value_a = 0.5, value_b = 0.5):
        # print("Increasing costs...")
        self.cost_currencyA += value_a
        self.cost_currencyB += value_b   
    
# International bank agent inherited from bank agent
class agent_international_bank(agent_bank):
    pass