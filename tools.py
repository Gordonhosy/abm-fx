import numpy as np
from corporates import *

# tools for model

def random_corporate(agent_id, corporate_type, static_map, model):
    '''
    generate random corporates at random postitions
    '''
    x = int(np.random.uniform(0, static_map.width))
    y = int(np.random.uniform(0, static_map.height))
    country = str(np.random.choice(corporate_type.params.country))
    level = int(np.random.uniform(corporate_type.params.level_min, corporate_type.params.level_max + 1))
    vision = level

    if country == 'A':
        amount_A = int(np.random.uniform(corporate_type.params.asset_min, corporate_type.params.asset_max + 1))
        currencyA = int(amount_A * level)

        amount_B = int(np.random.uniform(corporate_type.params.asset_min, corporate_type.params.asset_max + 1))
        currencyB = int(amount_B * 0.75 * level)

        cost_currencyA = int(np.random.uniform(corporate_type.params.costs_min, 2.5))
        cost_currencyB = int(np.random.uniform(2.5, corporate_type.params.costs_max + 1))

    elif country == "B":
        amount_A = int(np.random.uniform(corporate_type.params.asset_min, corporate_type.params.asset_max + 1))
        currencyA = int(amount_A *  0.75 * level)

        amount_B = int(np.random.uniform(corporate_type.params.asset_min, corporate_type.params.asset_max + 1))
        currencyB = int(amount_B * level)

        cost_currencyA = int(np.random.uniform(2.5, corporate_type.params.costs_max + 1))
        cost_currencyB = int(np.random.uniform(corporate_type.params.costs_min, 2.5))

    if level == 1:
        imp_utility = np.arange(1.04, 1.14, 0.01)
    elif level == 2:
        imp_utility = np.arange(1.04, 1.11, 0.01)
    elif level == 3:
        imp_utility = np.arange(1.03, 1.09, 0.01)
    elif level == 4:
        imp_utility = np.arange(1.03, 1.07, 0.01)
    elif level == 5:
        imp_utility = np.arange(1.02, 1.05, 0.01)
    
    return corporate_type.agent(agent_id,
                           model,
                           (x,y), 
                           moore = False,
                           country = country,
                           currencyA = currencyA,
                           currencyB = currencyB,
                           cost_currencyA = cost_currencyA,
                           cost_currencyB = cost_currencyB,
                           level = level,
                           vision = vision,
                           imp_utility = imp_utility)


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


def random_arb(agent_id, arb_type, init_pos, model):
    '''
    generate arbitragers at fixed positions
    '''
    x = init_pos[0]
    y = init_pos[1]

    currencyA = int(np.random.uniform(arb_type.params.asset_min, arb_type.params.asset_max + 1))
    currencyB = int(np.random.uniform(arb_type.params.asset_min, arb_type.params.asset_max + 1))
    cost_currencyA = int(np.random.uniform(arb_type.params.costs_min, arb_type.params.costs_max + 1))
    cost_currencyB = int(np.random.uniform(arb_type.params.costs_min, arb_type.params.costs_max + 1))
    
    vision = int(np.random.uniform(arb_type.params.vision_min, arb_type.params.vision_max + 1))
    
    return arb_type.agent(agent_id,
                           model,
                           (x,y), 
                           moore = False,
                           currencyA = currencyA,
                           currencyB = currencyB,
                           cost_currencyA = cost_currencyA,
                           cost_currencyB = cost_currencyB,
                           vision = vision)

def random_fund(agent_id, fund_type, init_pos, strategy, model):
    '''
    generate arbitragers at fixed positions
    '''
    x = init_pos[0]
    y = init_pos[1]

    currencyA = int(np.random.uniform(fund_type.params.asset_min, fund_type.params.asset_max + 1))
    currencyB = int(np.random.uniform(fund_type.params.asset_min, fund_type.params.asset_max + 1))
    cost_currencyA = int(np.random.uniform(fund_type.params.costs_min, fund_type.params.costs_max + 1))
    cost_currencyB = int(np.random.uniform(fund_type.params.costs_min, fund_type.params.costs_max + 1))
    
    return fund_type.agent(agent_id,
                           model,
                           (x,y), 
                           moore = False,
                           currencyA = currencyA,
                           currencyB = currencyB,
                           cost_currencyA = cost_currencyA,
                           cost_currencyB = cost_currencyB,
                           strat = strategy)