import numpy as np
from corporates import *

# tools for model

def random_corporate(agent_id, corporate_type, static_map, model):
    '''
    generate random corporates at random postitions
    '''
    x = int(np.random.uniform(0, static_map.width))
    y = int(np.random.uniform(0, static_map.height))
    currencyA = int(np.random.uniform(corporate_type.params.asset_min, corporate_type.params.asset_max + 1))
    currencyB = int(np.random.uniform(corporate_type.params.asset_min, corporate_type.params.asset_max + 1))
    cost_currencyA = int(np.random.uniform(corporate_type.params.costs_min, corporate_type.params.costs_max + 1))
    cost_currencyB = int(np.random.uniform(corporate_type.params.costs_min, corporate_type.params.costs_max + 1))
    vision = int(np.random.uniform(corporate_type.params.vision_min, corporate_type.params.vision_max + 1))
    
    return corporate_type.agent(agent_id,
                           model,
                           (x,y), 
                           moore = False,
                           currencyA = currencyA,
                           currencyB = currencyB,
                           cost_currencyA = cost_currencyA,
                           cost_currencyB = cost_currencyB,
                           vision = vision)


def random_central_bank(agent_id, central_bank_type, static_map, model):
    '''
    generate random central bank with random initial economic situation(or not just make two completely different macro environment)
    '''
    x = int(np.random.uniform(0, static_map.width))
    y = int(np.random.uniform(0, static_map.height))
    inflation_rate = np.random.uniform(0.01, 0.05)
    interest_rate = 0.0025
    growth_rate = interest_rate - inflation_rate 
    target_inflation_rate = 0.02

    agent_central_bank = central_bank_type.agent(agent_id, 
                                                 model, 
                                                 (x,y), 
                                                 moore = False, 
                                                 interest_rate = interest_rate, 
                                                 inflation_rate = inflation_rate, 
                                                 growth_rate = growth_rate,
                                                 target_inflation_rate = target_inflation_rate)

    return agent_central_bank


def random_bank(agent_id, bank_type, init_pos, model):
    '''
    generate local banks at fixed positions
    '''
    x = init_pos[0]
    y = init_pos[1]
    
    # if it is a bank in country A
    if y < 25: 
        currencyA = int(np.random.uniform(bank_type.params.local_asset_min, bank_type.params.local_asset_max + 1))
        currencyB = int(np.random.uniform(bank_type.params.foreign_asset_min, bank_type.params.foreign_asset_max + 1))
        cost_currencyA = int(np.random.uniform(bank_type.params.local_costs_min, bank_type.params.local_costs_max + 1))
        cost_currencyB = int(np.random.uniform(bank_type.params.foreign_costs_min, bank_type.params.foreign_costs_max + 1))
    
    # if it is a bank in country B
    elif y >= 25:
        currencyA = int(np.random.uniform(bank_type.params.foreign_asset_min, bank_type.params.foreign_asset_max + 1))
        currencyB = int(np.random.uniform(bank_type.params.local_asset_min, bank_type.params.local_asset_max + 1))
        cost_currencyA = int(np.random.uniform(bank_type.params.foreign_costs_min, bank_type.params.foreign_costs_max + 1))
        cost_currencyB = int(np.random.uniform(bank_type.params.local_costs_min, bank_type.params.local_costs_max + 1))
    
    vision = int(np.random.uniform(bank_type.params.vision_min, bank_type.params.vision_max + 1))
    
    return bank_type.agent(agent_id,
                           model,
                           (x,y), 
                           moore = False,
                           currencyA = currencyA,
                           currencyB = currencyB,
                           cost_currencyA = cost_currencyA,
                           cost_currencyB = cost_currencyB,
                           vision = vision)