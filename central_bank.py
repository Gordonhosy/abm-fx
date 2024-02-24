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
    - target inflation rate (2% or 3%)
    """

    def __init__(self, unique_id, model, pos, moore, interest_rate, inflation_rate, growth_rate, target_inflation_rate):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        self.pos = pos
        self.moore = moore
        self.adjust_rate_unit = 0.0025 
        # Create the agent's attribute and set the initial values.

        # Real World Observation
        self.interest_rate = interest_rate # initial nominal interest rate 
        self.inflation_rate = inflation_rate # initial inflation rate
        self.growth_rate = growth_rate # initial economic growth rate

        # Central Bank Target
        self.target_inflation_rate = target_inflation_rate # initial inflation target rate 
        self.target_growth_rate = target_inflation_rate + 0.02 # initial target growth rate (Long term GDP Growth)

    def step(self):
        """
        How central bank response to macroeconomic in every steps:
        - In each steps, there are some economic shocks to the variable(inflation rate, growth rate).
        - Central bank recalculate the target interest rate with Taylor's rule model.
        - Central bank cut/raise fed fund rate.
        - Interest rate and inflation rate converge to the target value.

        The overall idea can be summarize as follow:
        - Monetary Policy (Tightening or Easing) 
        -> Inflation Rate (General economic shock and converge to target)
        -> Interest Rate (Central bank response)
        -> Growth Rate (Function of interest rate, inflation rate, long term growth rate, target interest rate, white noise)
        -> Target Interest Rate (function of growth rate, interest rate, and inflation rate)
        -> Monetary Policy 
        """
        
        self.calculate_target_interest_rate() # Recalculate the target interest in every steps.
        self.adjust_interest_rate() # Central bank observe the economic and response to it (cut, raise).

        self.economic_cycle() # macroeconomic policy is tightening/easing.
        self.calculate_growth_rate() # calculate period end growth rate with inflation rate and interest rate.
        

    def calculate_target_interest_rate(self):
        """
        Taylor's Rule Model:
        The basic idea behind Taylor’s rule model is to provide a systematic and transparent approach for central banks to adjust interest rates in response to changes in inflation and output.
        - This model assumes the equilibrium federal funds rate of 2% above inflation(General Case).
        - Target Interest Rate = Inflation Rate + 0.02 + 0.5(Inflation Rate - Target Inflation Rate) + 0.5(Observe GDP - Long Run Economic Growth Rate)
        - Where growth rate approximate the observe gdp growth and we assume long-run economic growth rate equal to target inflation rate plus 2%.
        """

        self.target_interest_rate =  self.inflation_rate  + 0.02 + 0.5 * (self.inflation_rate - self.target_inflation_rate) + 0.5 * (self.growth_rate - self.target_growth_rate)

        # we assume there's no negative target rate
        if self.target_interest_rate < 0:
            self.target_interest_rate = 0

    def adjust_interest_rate(self):
        """
        Rational behind central bank cut/raise interest rate
        - When target interest rate is higher than interest rate, than central bank will raise interest rate.
        - When target interest rate is lower than interest rate, than central bank will cut interest rate.
        - If central bank adjust interest rate, then inflation should converge to the target inflation rate base on economic theory.
        - If central bank raise/cut rate the smoothing parameter for conversion of inflation function equal to 0.5, otherwise equal to 0.2.
        """

        # When inflation rate within 0~3%, we assume central bank do nothing.
        if  0.00 <= self.inflation_rate <= 0.03:
            self.inflation_rate_converge(smoothing_param = 0.5)
        
        # when economic is pretty bad(deflation occur, recession), than fed will cut rate dramatically.
        elif self.inflation_rate < 0:

            number = 0
            while self.interest_rate - self.adjust_rate_unit >= 0:
                self.interest_rate = self.interest_rate - self.adjust_rate_unit
                self.inflation_rate_converge(smoothing_param = 0.5)
                number += 1
                if self.interest_rate - self.adjust_rate_unit < 0:
                    break
                if number == 10:
                    break
            
            # make sure no negative rate
            if self.interest_rate < 0:
                self.interest_rate = 0
        
        # Normal Monetary Policy.
        else: 
            # target interest rate > interest rate -> raise rate
            if self.target_interest_rate - self.interest_rate >= self.adjust_rate_unit:

                self.interest_rate  = self.interest_rate + self.adjust_rate_unit
                self.inflation_rate_converge(smoothing_param = 0.5)

            # target interest rate < interest rate -> cut rate
            elif (self.interest_rate - self.target_interest_rate  >= self.adjust_rate_unit) & (self.interest_rate > self.adjust_rate_unit):

                self.interest_rate = self.interest_rate - self.adjust_rate_unit 
                self.inflation_rate_converge(smoothing_param = 0.5)

                # make sure no negative rate
                if self.interest_rate < 0:
                    self.interest_rate = 0


    def calculate_growth_rate(self):
        """
        In this model, we assume economic growth rate is the following:
        - growth_rate(t) = 0.1 * (long_term_growth_rate) + 0.2 * (inflation_rate) + 0.2 * (interest_rate_factor) + 0.5 * (growth_rate(t-1)) + white_noise
        """

        self.growth_rate = 0.1 * self.target_growth_rate + 0.2 * (self.target_interest_rate - self.interest_rate) + 0.5 * self.growth_rate + 0.2 * self.inflation_rate

    def inflation_rate_converge(self, smoothing_param):
        """
        Helper function for calculate the converge of inflation rate
        - In Tyler's rule model, we assume observation inflation rate will converge to target inflation rate.
        """
        self.inflation_rate = (1 - smoothing_param) * self.target_inflation_rate + smoothing_param * self.inflation_rate + np.random.normal(0,1) * 0.0001

    def economic_cycle(self):
        """
        Adjust macroeconomic growth rate
        - If monetary policy is tightening than negative inflation shock
        - If monetary policy is easing than positive inflation shock
        """

        # when interest rate < target --> easing environment
        if self.target_interest_rate > self.interest_rate:

            self.inflation_rate = self.inflation_rate + np.abs(np.random.normal(0,1) * 0.01)
        
        # when interest rate > target --> tightening environment
        elif self.target_interest_rate < self.interest_rate:

            self.inflation_rate = self.inflation_rate - np.abs(np.random.normal(0,1) * 0.01)


        

        
        

        






