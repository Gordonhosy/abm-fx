import mesa
import tools
import random
from corporates import *
from central_bank import *
from resources import *
from static import *



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
        
        # init central bank
        for central_bank_type in self.all_agents.central_banks:
            for _ in range(central_bank_type.params.number_of_central_bank):

                agent_central_bank = tools.random_central_bank(agent_id, central_bank_type, self.static_map, self)
                self.grid.place_agent(agent_central_bank, agent_central_bank.pos)
                self.schedule.add(agent_central_bank)
                agent_id += 1
        
        # TO DO: init banks, hedge funds
        
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
            
        # corporate
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
    

# if __name__ == '__main__':

#     steps = 300
#     model = abmodel(static_map_v0(), all_agents())
#     model.run_model(steps)
#     model_results = model.datacollector.get_model_vars_dataframe()
    
#     print(model_results)