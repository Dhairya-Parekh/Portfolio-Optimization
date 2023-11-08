# ------------------- Imports -------------------
import gym
import numpy as np
import pandas as pd
# ------------------- Environment Parameters -------------------
FITTING_PERIOD = 50
HOLDING_PERIOD = 3
# ------------------- Environment -------------------
class MarketEnvironment(gym.Env):

    def __init__(self, type):
        if type == 'train':
            filepath = '../Data/New-Data/train.csv'
        elif type == 'test':
            filepath = '../Data/New-Data/test.csv'
        elif type == 'validation':
            filepath = '../Data/New-Data/validation.csv'
        else:
            raise ValueError(f"Invalid type: {type}")
        self.type = type
        closing_prices = pd.read_csv(filepath)
        print(f"This is {type} Environment with {len(closing_prices)} rows and {len(closing_prices.columns)} columns")
        closing_prices.sort_values(by=['Date'], inplace=True)
        closing_prices.dropna(inplace=True)
        closing_prices.reset_index(drop=True, inplace=True)
        # Returns are calculated as the percentage change in price not on Date column
        returns_df = closing_prices.drop(columns=['Date']).pct_change().dropna()
        dates = closing_prices['Date']
        self.dates = dates
        # Make sure length of returns_df - FITTING_PERIOD is integer multiple of HOLDING_PERIOD
        remove_sample_number = (len(returns_df) - FITTING_PERIOD) % HOLDING_PERIOD
        if remove_sample_number != 0:
            self.returns_df = returns_df.iloc[:-remove_sample_number]
        else:
            self.returns_df = returns_df
        print(f"ENV::Number of {type}ing examples: {(len(self.returns_df)-FITTING_PERIOD)//HOLDING_PERIOD} Hence, effective {type}ing length: {len(self.returns_df) - FITTING_PERIOD} days")
        self.num_assets = returns_df.shape[1]
        # The state is the [current_index, current_portfolio_value] the action is allocation of portfolio weights to each stock, and the reward is the return of the portfolio
        self.state = None
        self.portfolio_values_log = None
        self.dates_log = None
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=np.inf, shape=(self.num_assets,FITTING_PERIOD), dtype=np.float32)
        self.reset()

    def reset(self):
        self.state = [FITTING_PERIOD, 1.0]
        self.portfolio_values_log = np.array([1])
        self.dates_log = np.array([self.dates[FITTING_PERIOD-1]])
        observation = self.returns_df.iloc[:FITTING_PERIOD].values.T
        # print(f"Observation shape: {observation.shape}, index: {0} - {FITTING_PERIOD-1}")
        return observation
    
    def step(self, action):
        # Make sure the action is valid, i.e. the sum of weights is 1
        weights = action / np.sum(action)
        current_index = self.state[0]
        # Given action as portfolio weights, calculate the portfolio return for the holding period
        holding_returns = self.returns_df.iloc[current_index:current_index+HOLDING_PERIOD].values
        portfolio_values = (weights * (1 + holding_returns).cumprod(axis=0)).sum(axis=1)
        cumulative_return = portfolio_values[-1] - 1
        portfolio_values = self.state[1] * portfolio_values
        self.portfolio_values_log = np.append(self.portfolio_values_log, portfolio_values)
        # Extract dates from holding returns
        dates = self.dates[current_index:current_index+HOLDING_PERIOD]
        self.dates_log = np.append(self.dates_log, dates)
        current_index += HOLDING_PERIOD
        # Calculate the new state = the next date
        self.state = [current_index, portfolio_values[-1]]
        # Calculate the observation = the returns for the fitting period
        observation = self.returns_df.iloc[current_index-FITTING_PERIOD:current_index].values.T
        # print(f"Observation shape: {observation.shape}, index: {current_index-FITTING_PERIOD} - {current_index-1}")
        # Calculate the reward = portfolio return for the holding period
        reward = cumulative_return
        # Calculate the done flag = whether the new state is the last possible date i.e. adding the holding period to the current index of state exceeds the length of the returns dataframe
        done = self.state[0] + HOLDING_PERIOD > len(self.returns_df)
        # Calculate the info = None
        info = {}
        return observation, reward, done, info
    
    def get_data(self):
        return self.portfolio_values_log, self.dates_log