import mesa
from corporates import *
from resources import *
import tools
import random

class abmodel(mesa.Model):
    '''
    Generalised model to run the simulation
    '''

    def __init__(self, static_map, all_agents):
        self.static_map = static_map
        self.running = True
        self._steps = 0
        self.all_agents = all_agents
            
        # initiate the mesa grid class
        self.grid = mesa.space.MultiGrid(self.static_map.width, self.static_map.height, torus = False)
        
        # initate scheduler
        self.schedule = mesa.time.RandomActivationByType(self)
        
        # initate datacollector
        model_reporters = {"Step": lambda m: m.schedule.steps}
        agent_reporters = {}
        for _, agent_types in self.all_agents.__dict__.items():
            for agent_type in agent_types:
                model_reporters[agent_type.name] = lambda m: m.schedule.get_type_count(agent_type.agent)
                model_reporters[f'Trade Volume ({agent_type.name})'] = lambda m: sum(len(a.traded_partners) for a in m.schedule.agents_by_type[agent_corporate_v0].values())
                model_reporters[f'Price ({agent_type.name})'] = lambda m: self.geometric_mean(a.traded_prices for a in m.schedule.agents_by_type[agent_corporate_v0].values())
                
                # better include pos and trading range for visualisation
                agent_reporters[f'Trade Network ({agent_type.name})'] = lambda a:self.get_trade_partners(a)
                
        self.datacollector = mesa.DataCollector(
            model_reporters = model_reporters,
            agent_reporters = agent_reporters
        )
        
        # initate the grid with the static map
        currencyA_map_init = self.static_map.currencyA_map_init
        currencyB_map_init = self.static_map.currencyB_map_init
        
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
            
        # init corporates
        for corporate_type in self.all_agents.corporates:
            for i in range(corporate_type.params.init_population):
                agent_corporate = tools.random_corporate(agent_id, corporate_type, self.static_map, self)
                self.grid.place_agent(agent_corporate, agent_corporate.pos)
                self.schedule.add(agent_corporate)
                agent_id += 1
    
        # TO DO: init banks, hedge funds, central banks
        
        
    
    def geometric_mean(self, listoflist):
        '''
        helper function to calculate geometric mean of price
        '''
        # first flatten the array
        price_list = [item for sublist in listoflist for item in sublist]
        
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
        
        # currency A
        for agent_currencyA in self.schedule.agents_by_type[currencyA_basic].values():
            agent_currencyA.step()
            
        
        # currency B
        for agent_currencyB in self.schedule.agents_by_type[currencyB_basic].values():
            agent_currencyB.step()
            
        
        # corporates
        # randomise move, earn money and trade sequence to make sure not one is advantaged
        for corporate_type in self.all_agents.corporates:
            corporates_shuffle = self.randomise_agents(corporate_type.agent)
            for corporate in corporates_shuffle:
                corporate.traded_prices = []
                corporate.traded_partners = []
                corporate.move()

            corporates_shuffle = self.randomise_agents(corporate_type.agent)
            for corporate in corporates_shuffle:
                corporate.earn_money()
                corporate.pay_costs()
                corporate.if_bankrupt()

            corporates_shuffle = self.randomise_agents(corporate_type.agent)
            for corporate in corporates_shuffle:
                corporate.trade_with_neighbors()
        
        # TO DO: model how banks, hedge funds, central banks behave
            
        self.schedule.steps += 1
        self._steps += 1
        self.datacollector.collect(self)
    
    
    def run_model(self, steps = 1000):
        '''
        helper function to run multiple steps
        '''
        for i in range(steps):
            self.step()
            
    
    # need to think of how to generalise this        
    def get_trade_partners(self, agent):
        '''
        ### THIS IS ONLY FOR CORPORATE V0 ###
        return trade partners of the round for reporting
        '''
        if isinstance(agent, agent_corporate_v0):
            return agent.traded_partners
        else:
            return None