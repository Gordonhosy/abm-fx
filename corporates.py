import mesa
from resources import *
import numpy as np
import random
import math

class agent_corporate_v0(mesa.Agent):
    '''
    This version of corporate is not realistic, is just for testing
    It is assume to be the only corporate type in the world if used
    Corporate:
    - Aims to find and earn currency A or B
    - Has costs to pay every timestep
    - Bankrupt if cannot pay the cost
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
        self.traded_partners = []

    
    def is_occupied(self, pos):
        '''
        Function to check if a cell is occupied by other agents
        '''
        if pos == self.pos:
            return False
        
        # check if this cell includes something that is mutually exclusive
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            
            # TO DO: need to include other agents as well later
            if isinstance(obj, agent_corporate_v0):
                return True
        return False
    
    
    def calculate_utility(self, currencyA, currencyB):
        '''
        Function to calculate utility
        '''
        # assume we use Cobb-Douglas
        cost_total = self.cost_currencyA + self.cost_currencyB
        return (currencyA**(self.cost_currencyA/cost_total)) * (currencyB**(self.cost_currencyB/cost_total))
    
    
    def get_currency_amount(self, pos, currencyX_basic):
        '''
        Return the amount of currency A if there is any
        '''
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            if isinstance(obj, currencyX_basic):
                return obj.amount
        return 0         
    
    
    def get_corporates(self, pos):
        '''
        Return the agent of the postition if any
        '''
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            if isinstance(obj, agent_corporate_v0):
                return obj
        return None
    
    
    def move(self):
        '''
        Corporate moves according to 4 steps
        '''
        
        # 1. Identify available neighbors
        
        neighbors_available = [i for i in self.model.grid.get_neighborhood(self.pos, self.moore, True, self.vision)\
                    if not self.is_occupied(i)]

        # 2. Calculate utitlities
        utilities = [self.calculate_utility(self.currencyA + self.get_currency_amount(pos, currencyA_basic), \
                        self.currencyB + self.get_currency_amount(pos, currencyB_basic)) for pos in neighbors_available]

        # 3. Find the best cell to move to
        options = [neighbors_available[i] for i in np.argwhere(utilities == np.amax(utilities)).flatten()]
        random.shuffle(options)
        final_decision = options[0] # random choice if more than one max
        
        # 4. Move agent
        self.model.grid.move_agent(self, final_decision)
        
        
    def earn_money(self):
        '''
        Function for corporate to earn money each step
        '''
        contents = self.model.grid.get_cell_list_contents(self.pos)
        for obj in contents:
            if isinstance(obj, currencyA_basic):
                self.currencyA += obj.amount
                obj.amount -= obj.amount
                
            if isinstance(obj, currencyB_basic):
                self.currencyB += obj.amount
                obj.amount -= obj.amount
        
        
    def pay_costs(self):
        '''
        Function for corporate to pay money each step
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
            currencyB_exchange = round(price, 4)
        else:
            currencyA_exchange = round(1/price, 4)
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
        
        if ((self.currencyA == 0) | (self.currencyB == 0)) | ((opponent.currencyA == 0) | (opponent.currencyB == 0)):
            return False
        
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
        
        # TO DO: need to think about why the prices are extreme
        if (math.isinf(price)) | (math.isclose(price,0)):
            return
    
        if mrs_self > mrs_opponent:
            # self want to buy currency A and sell currency B
            sold = self.maybe_sell_currencyB(opponent, price, utility_self, utility_opponent)
            # if criteria not met
            if not sold:
                return
        else:
            # self want to sell currency A and buy currency B
            sold = opponent.maybe_sell_currencyB(self, price, utility_opponent, utility_self)
            if not sold:
                return
        
        # store trades
        self.traded_prices.append(price)
        self.traded_partners.append(opponent.unique_id)
        
        self.trade(opponent)
    
    
    def trade_with_neighbors(self):
        '''
        Function for trading with other agents
        '''
        neighbors = [self.get_corporates(pos) for pos in self.model.grid.get_neighborhood(self.pos, self.moore, True, self.vision) if self.is_occupied(pos)]
        
        if len(neighbors) == 0:
            return [], []
        
        for opponent in neighbors:
            if opponent:
                self.trade(opponent)
        return
    
