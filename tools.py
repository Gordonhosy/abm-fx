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