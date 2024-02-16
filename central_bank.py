import mesa
import numpy as np 



class agent_central_bank_v0(mesa.Agent):
    """
    Central Bank Agent, very unrealistic one. \\
    Central Bank:
    - Aims to stable the inflation to target level and smooth the economic growth.
    - current idea is try to mimic a economic cycle to see how this affect FX sport rate. \\
    
    Parameters:
    - interest_rate 
    - inflation rate 
    - target inflation rate
    """

    def __init__(self, unique_id, model, pos, moore, interest_rate, inflation_rate, target_inflation_rate):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        self.pos = pos
        self.moore = moore
        # Create the agent's attribute and set the initial values.
        self.target_inflation_rate = target_inflation_rate
        self.interest_rate = interest_rate
        self.inflation_rate = inflation_rate
        self.real_growth = self.interest_rate + self.inflation_rate

    def step(self):

        self.monetary_policy(band = 0.015)

    def monetary_policy(self, band):


        if self.inflation_rate > (1 + 1 * band) * self.target_inflation_rate:

            self.interest_rate += 1 * 0.0025
            self.inflation_rate = self.inflation_rate + 0.2 * (self.target_inflation_rate - self.inflation_rate) + np.random.normal(0, 1) * 0.001 # converge to target inflation
            self.real_growth = self.interest_rate + self.inflation_rate # update real growth

        if self.inflation_rate < (1 - 1 * band) * self.target_inflation_rate:

            if self.interest_rate > 0.0025:

                self.interest_rate -= 1 * 0.0025
                self.inflation_rate = self.inflation_rate + 0.2 * (self.target_inflation_rate - self.inflation_rate) + np.random.normal(0, 1) * 0.001 # converge to target inflation
                self.real_growth = self.interest_rate + self.inflation_rate # update real growth

            else:
                self.inflation_rate += np.random.normal(0, 1) * 0.001
                self.real_growth = self.interest_rate + self.inflation_rate # update real growth

        else:
            self.inflation_rate += np.random.normal(0, 1) * 0.001

            # if inflation is normal, than central bank will readjust the interest rate to neutral rate 
            if self.interest_rate - self.target_inflation_rate > 0:
                self.interest_rate -= 0.0025
                self.real_growth = self.interest_rate + self.inflation_rate # update real growth
        
            if self.interest_rate -  self.target_inflation_rate < 0:
                self.interest_rate += 0.0025
                self.real_growth = self.interest_rate + self.inflation_rate # update real growth







