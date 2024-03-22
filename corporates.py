import mesa
from resources import *
import numpy as np
import random
import math
import pandas as pd

class agent_corporate_v0(mesa.Agent):
    '''
    This version of corporate is not realistic, is just for testing
    It is assume to be the only corporate type in the world if used
    Corporate:
    - Aims to find and earn currency A or B
    - Has costs to pay every timestep
    - Bankrupt if cannot pay the cost
    '''
    
    def __init__(self, agent_id, model, pos, moore, country, currencyA, currencyB, cost_currencyA, cost_currencyB, level, vision, imp_utility):
        super().__init__(agent_id, model)
        self.pos = pos
        self.moore = moore
        self.country = country
        self.currencyA = currencyA
        self.currencyB = currencyB
        self.cost_currencyA = cost_currencyA
        self.cost_currencyB = cost_currencyB
        self.level = level
        self.vision = vision
        self.trade_happened = False
        self.traded_prices = []
        self.traded_partners = []
        self.traded_amount = []
        self.trade_direction = None # 'long' or 'short'
        self.amount = None
        self.price = None
        self.utilities = imp_utility
        self.improve_utility = np.random.choice(self.utilities)
        self.limit = 0

        # select the map:
        if self.country == "A":
            possible = pd.read_excel(r"../ABM_FX/geographic_data/MAP.xlsx", sheet_name = "US_MAP").values
        elif self.country == "B":
            possible = pd.read_excel(r"../ABM_FX/geographic_data/MAP.xlsx", sheet_name = "JP_MAP").values
        
        self.possible_moves = np.nonzero(possible)
    
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
    
    
    def calculate_utility(self, currencyA, currencyB, mid_price):
        '''
        Function to calculate utility
        '''
        # assume we use Cobb-Douglas
        adj_cost_currencyB = self.cost_currencyB/ mid_price
        cost_total = self.cost_currencyA + adj_cost_currencyB
        return (currencyA**(self.cost_currencyA/cost_total)) * (currencyB**(adj_cost_currencyB/cost_total))
    
    
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
    
    
    def move(self, mid_price):
        '''
        Corporate moves according to 4 steps
        '''
        # 1. Identify available neighbors
        neighbors_unoccupied = [i for i in self.model.grid.get_neighborhood(self.pos, self.moore, True, self.vision)\
                    if not self.is_occupied(i)]
        
        # 1.b Remove out of map positions
        neighbors_available = []
        for loc in neighbors_unoccupied:
            if (loc[0] in self.possible_moves[0]) and (loc[1] in self.possible_moves[1]):
                neighbors_available.append(loc)
            else:
                continue

        # 2. Calculate utitlities
        utilities = [self.calculate_utility(self.currencyA + self.get_currency_amount(pos, currencyA_basic), \
                        self.currencyB + self.get_currency_amount(pos, currencyB_basic), mid_price) for pos in neighbors_available]

        # 3. Find the best cell to move to
        options = [neighbors_available[i] for i in np.argwhere(utilities == np.amax(utilities)).flatten()]

        random.shuffle(options)

        if (self.pos in options) and (self.limit <= 3):
            final_decision = self.pos
            self.limit += 1
        else:
            final_decision = next(iter(options), self.pos) # random choice if more than one max
            self.limit = 0
        
        # 4. Move agent
        self.model.grid.move_agent(self, final_decision)
        
        
    def earn_money(self):
        '''
        Function for corporate to earn money each step
        '''
        contents = self.model.grid.get_cell_list_contents(self.pos)
        factor = 0.5
        for obj in contents:
            if isinstance(obj, currencyA_basic):
                self.currencyA += factor*obj.amount
                obj.amount -= factor*obj.amount
                
            if isinstance(obj, currencyB_basic):
                self.currencyB += factor*obj.amount
                obj.amount -= factor*obj.amount
        
        
    def pay_costs(self, interest_rate_a, interest_rate_b):
        '''
        Function for corporate to pay money each step
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


    def put_order(self, mid_price):
        '''
        Function for corporates to put orders to trade FX
        The aim is to improve the utility by a certain amount
        '''
        
        if self.trade_happened == True:
            if self.improve_utility < max(self.utilities):
                uti = self.improve_utility + 0.01
        else:
            if self.improve_utility > min(self.utilities):
                uti = self.improve_utility - 0.01

        uti = round(self.improve_utility, 2)

        org_utility = self.calculate_utility(self.currencyA, self.currencyB, mid_price)
        target_utility = org_utility * 1.01
        
        adj_cost_currencyB = self.cost_currencyB / mid_price
        cost_total = self.cost_currencyA + adj_cost_currencyB
        
        org_slope = (org_utility ** (cost_total/adj_cost_currencyB)) * (-self.cost_currencyA/adj_cost_currencyB) * (self.currencyA**(-(self.cost_currencyA/adj_cost_currencyB + 1)))
        utility = org_utility
        change_currencyA = 0
        change_currencyB = 0
        mid_price = np.random.normal(loc = mid_price, scale=10, size=None)

        
        if org_slope < -1:
            while ((utility < target_utility) & (abs(change_currencyB) < self.currencyB*0.5)):
                change_currencyA += 1
                change_currencyB -= mid_price
                utility = self.calculate_utility(self.currencyA + change_currencyA, self.currencyB + change_currencyB, mid_price)
                
        else:
            while ((utility < target_utility) & (abs(change_currencyA) < self.currencyA*0.5)):
                change_currencyA -= 1
                change_currencyB += mid_price
                utility = self.calculate_utility(self.currencyA + change_currencyA, self.currencyB + change_currencyB, mid_price)
                
       
        # the changes need to be in opposite directions for a trade to happen
        # if (change_currencyA * change_currencyB < 0) & (not math.isclose(change_currencyA, 0)):
        if utility > target_utility:
            
            if change_currencyA < 0:
                self.trade_direction = 'short'
                self.amount = -round(change_currencyA)
                self.price =  -round(change_currencyB/change_currencyA, 2)
                # some quotes are not sensible after rounding
                if self.currencyA < self.amount:
                    self.trade_direction = 'short'
                    self.amount = -int(change_currencyA)
                    self.price = -round(change_currencyB/change_currencyA, 2)
            else:
                self.trade_direction = 'long'
                self.amount = round(change_currencyA)
                self.price = -round(change_currencyB/change_currencyA, 2)
                # some quotes are not sensible after rounding
                if self.currencyB < self.amount * self.price:
                    self.trade_direction = 'long'
                    self.amount = int(change_currencyA)
                    self.price = -round(change_currencyB/change_currencyA, 2)
        else:
            self.trade_direction = None
            self.amount = None
            self.price = None
        