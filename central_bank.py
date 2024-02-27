import mesa
import numpy as np 



class agent_central_bank_v0(mesa.Agent):
    """
    Central Bank Agent, very unrealistic one. \\
    Central Bank:
    - Aims to stabilise inflation to a target level and smooth economic growth.
    - the current idea is trying to mimic an economic cycle to see how this affects the FX sport rate. \\
    
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
        How Central Bank responds to macroeconomics in every step:
        - In each step, there are some economic shocks to the variable(inflation rate, growth rate).
        - Central banks recalculate the target interest rate with Taylor's rule model.
        - Central bank cut/raised fed fund rate.
        - Interest rate and inflation rate converge to the target value.
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
        - Where growth rate approximates the observed GDP growth and we assume long-run economic growth rate equal to the target inflation rate plus 2%.
        """

        self.target_interest_rate =  self.inflation_rate + 0.02 + 0.5 * (self.inflation_rate - self.target_inflation_rate) + 0.5 * (self.growth_rate - self.target_growth_rate)

        # we assume there's no negative target rate
        if self.target_interest_rate < 0:
            self.target_interest_rate = 0

    def adjust_interest_rate(self):
        """
        The rationale behind central bank cutting/raising the interest rate
        - When the target interest rate is higher than the interest rate, then the central bank will raise the interest rate.
        - When the target interest rate is lower than the interest rate, then the central bank will cut the interest rate.
        - If the central bank adjusts the interest rate, then inflation should converge to the target inflation rate based on economic theory.
        - If the central bank raises/cuts the interest rate the smoothing parameter for conversion of inflation function is equal to 0.5, otherwise equal to 0.2.
        """

        # When inflation rate within 0~3%, we assume central bank do nothing.
        if  0.005 <= self.inflation_rate <= 0.035:
            self.inflation_rate_converge(smoothing_param = 0.5)

        
        # when economic is pretty bad(deflation occur, recession), than fed will cut rate dramatically.
        elif self.inflation_rate <= 0:

            number = 0
            while self.interest_rate - self.adjust_rate_unit >= 0:
                self.interest_rate = self.interest_rate - self.adjust_rate_unit
                self.inflation_rate_converge(smoothing_param = 0.5)
                number += 1

                if number == 15:
                    break
            
            # make sure no negative rate
            if self.interest_rate < 0:
                self.interest_rate = 0
        
        else: # Normal Monetary Policy.
            times = 0
            # target interest rate > interest rate -> raise rate
            while self.target_interest_rate - self.interest_rate >= self.adjust_rate_unit:

                self.interest_rate  = self.interest_rate + self.adjust_rate_unit
                self.inflation_rate_converge(smoothing_param = 0.5)
                times += 1

                if times == 3:
                    break

            # target interest rate < interest rate -> cut rate
            while (self.interest_rate - self.target_interest_rate  >= self.adjust_rate_unit) and (self.interest_rate > self.adjust_rate_unit):

                self.interest_rate = self.interest_rate - self.adjust_rate_unit 
                self.inflation_rate_converge(smoothing_param = 0.5)
                times += 1

                # make sure no negative rate
                if self.interest_rate < 0:
                    self.interest_rate = 0

                if times == 3:
                    break

    def calculate_growth_rate(self):
        """
        In this model, we assume the economic growth rate is the following:
        Growth rate should be a function containing how interest rate(Indicator Function)
        """

        if self.target_interest_rate > self.interest_rate:
            """
            Target interest rate is higher than the current interest rate than it mean a tightening monetary policy
            --> current growth rate too high
            --> output gap become positive
            --> target interest rate increase
            """
            self.growth_rate = self.target_growth_rate + 0.01 * np.random.normal(1, 1)

        elif self.target_interest_rate < self.interest_rate:
            """
            Target interest rate is lower than the current interest rate than it mean a easing monetary policy 
            --> current growth rate is still too low
            --> output gap become negative 
            --> target interest rate drop
            """
            self.growth_rate = self.target_growth_rate + 0.01 * np.random.normal(-1, 1)
        
        elif self.target_interest_rate == self.interest_rate:
            """
            Neutral monetary policy 
            --> central bank achieve target growth rate.
            """
            self.growth_rate = self.target_growth_rate + 0.01 * np.random.normal(0, 1)

    def economic_cycle(self):
        """
        Adjust macroeconomic growth rate
        - If monetary policy is tightening then it means current inflation is too high --> positive economic shock
        - If monetary policy is teasing then it means current inflation is too low --> negative economic shock
        """

        if self.target_interest_rate > self.interest_rate:

            self.inflation_rate = self.inflation_rate + np.random.normal(1,1) * 0.01
      
        elif self.target_interest_rate < self.interest_rate:

            self.inflation_rate = self.inflation_rate + np.random.normal(-1,1) * 0.01
        

    def inflation_rate_converge(self, smoothing_param):
        """
        Helper function for calculating the converge of the inflation rate
        - In Tyler's rule model, we assume the observation inflation rate will converge to the target inflation rate.
        """
        self.inflation_rate = (1 - smoothing_param) * self.target_inflation_rate + smoothing_param * self.inflation_rate + np.random.normal(0,1) * 0.0001


        
        

        






