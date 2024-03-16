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
        if len(options) == 0:
            print(self.unique_id)
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


    def put_order(self):
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

        target_utility = self.calculate_utility(self.currencyA, self.currencyB) * uti
        # target the equilibrium at the new utility
        cost_total = self.cost_currencyA + self.cost_currencyB
        target_currencyA = ((self.cost_currencyB/self.cost_currencyA)*(target_utility**(-cost_total/self.cost_currencyB)))**(-1/(1+(self.cost_currencyA/self.cost_currencyB)))
        target_currencyB = (target_utility/(target_currencyA**(self.cost_currencyA/cost_total)))**(cost_total/self.cost_currencyB)
        
        change_currencyA = target_currencyA - self.currencyA
        change_currencyB = target_currencyB - self.currencyB
        
        # the changes need to be in opposite directions for a trade to happen
        if (change_currencyA * change_currencyB < 0) & (not math.isclose(change_currencyA, 0)):
            
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
        