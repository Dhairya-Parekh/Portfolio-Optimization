import os
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.optimize as sco
import plotly.graph_objects as go
from tf_agents.environments import suite_gym, tf_py_environment
from Environment import MarketEnvironment, FITTING_PERIOD, HOLDING_PERIOD

# -------------------- Helper Functions ------------------ #
def get_returns_df():
    train_closing_prices = pd.read_csv('../Data/train.csv')
    test_closing_prices = pd.read_csv('../Data/test.csv')
    # Sort DataFrame by date
    train_closing_prices['Date'] = pd.to_datetime(train_closing_prices['Date'])
    train_closing_prices.sort_values(by='Date', inplace=True)
    train_closing_prices.reset_index(drop=True, inplace=True)
    test_closing_prices['Date'] = pd.to_datetime(test_closing_prices['Date'])
    test_closing_prices.sort_values(by='Date', inplace=True)
    test_closing_prices.reset_index(drop=True, inplace=True)
    # Drop rows with NaN values
    train_closing_prices.dropna(inplace=True)
    test_closing_prices.dropna(inplace=True)
    # Percent change in closing prices for each column except for the first one
    train_returns_df = pd.DataFrame()
    test_returns_df = pd.DataFrame()
    train_returns_df['Date'] = train_closing_prices['Date']
    test_returns_df['Date'] = test_closing_prices['Date']
    for column in train_closing_prices.columns[1:]:
        train_returns_df[column] = train_closing_prices[column].pct_change()
        test_returns_df[column] = test_closing_prices[column].pct_change()
    train_returns_df.dropna(inplace=True)
    test_returns_df.dropna(inplace=True)
    return train_returns_df, test_returns_df

def evaluate_portfolio(returns, weights):
    """
    Returns portfolio statistics
    
    Parameters
    ----------
    returns : NumPy array
        Returns for each instrument in the portfolio
    weights : NumPy array
        Weights for each instrument in the portfolio
    Returns
    -------
    dict
        Dictionary of portfolio statistics
    """
    # Portfolio value at each time step
    portfolio_values = ((1+returns).cumprod(axis=0) * weights).sum(axis=1)
    # Append 1 at the beginning to represent initial investment
    portfolio_values = np.append(1, portfolio_values)
    # Portfolio return at each time step (in %)
    portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
    # Calculate mean return
    mean_return = portfolio_returns.mean()
    # Calculate standard deviation of return
    sigma = portfolio_returns.std()
    # Calculate Sharpe ratio
    sharpe = mean_return / sigma
    # Calculate maximum drawdown
    drawdowns = (portfolio_values[:, np.newaxis] - portfolio_values) / portfolio_values[:, np.newaxis]
    np.fill_diagonal(drawdowns, 0)
    max_drawdown = np.min(drawdowns)
    # Cumulative return
    cumulative_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    # Return portfolio statistics
    return {
        'Mean Return': mean_return*100,
        'Volatility': sigma,
        'Sharpe': sharpe,
        'Portfolio Value': portfolio_values[1:],
        'Portfolio Returns': portfolio_returns,
        'Max Drawdown': max_drawdown,
        'Cumulative Return': cumulative_return*100
    }

def combine_portfolio_metrics(portfolio_metrics):
    """
    Combine portfolio metrics for each time step into a single dictionary
    
    Parameters
    ----------
    portfolio_metrics : list
        List of dictionaries of portfolio metrics
    Returns
    -------
    dict
        Dictionary of portfolio metrics
    """
    combined_metrics = {}
    for metric in portfolio_metrics[0].keys():
        combined_metrics[f"{metric} array"] = np.array([portfolio_metric[metric] for portfolio_metric in portfolio_metrics])

    # Cumulative portfolio value
    portfolio_values = np.array([portfolio_metric['Portfolio Value'] for portfolio_metric in portfolio_metrics])
    # Multiply all values of next row with the previous row's last value
    for i in range(1, len(portfolio_values)):
        portfolio_values[i] *= portfolio_values[i-1][-1]

    portfolio_values = portfolio_values.flatten()
     # Append 1 at the beginning to represent initial investment
    portfolio_values = np.append(1, portfolio_values)
    # Portfolio return at each time step (in %)
    portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
    # Calculate mean return
    mean_return = portfolio_returns.mean()
    # Calculate standard deviation of return
    sigma = portfolio_returns.std()
    # Calculate Sharpe ratio
    sharpe = mean_return / sigma
    # Calculate maximum drawdown
    drawdowns = (portfolio_values[:, np.newaxis] - portfolio_values) / portfolio_values[:, np.newaxis]
    np.fill_diagonal(drawdowns, 0)
    max_drawdown = np.min(drawdowns)
    # Cumulative return
    cumulative_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    combined_metrics['Mean Return'] = mean_return*100
    combined_metrics['Volatility'] = sigma
    combined_metrics['Sharpe'] = sharpe
    combined_metrics['Portfolio Value'] = portfolio_values[1:]
    combined_metrics['Portfolio Returns'] = portfolio_returns
    combined_metrics['Max Drawdown'] = max_drawdown
    combined_metrics['Cumulative Return'] = cumulative_return*100
    combined_metrics['Dates'] = combined_metrics['Dates array'].flatten()
    return combined_metrics

def plot_portfolio_statistics(results, metric, dataset='train'):
    """
    Function to plot portfolio statistics as a bar chart for the metric specified
    """
    fig = go.Figure()
    if metric in ['Mean Return', 'Volatility', 'Sharpe', 'Max Drawdown', 'Cumulative Return']:
        # x = list(results.keys())
        # y = [results[name][metric] for name in results.keys()]
        fig.add_trace(go.Bar(x=list(results.keys()), y=[results[name][metric] for name in results.keys()], name=metric))
        fig.update_layout(title=metric, xaxis_title='Strategy', yaxis_title=metric)
        pass
    elif metric in ['Portfolio Value', 'Portfolio Returns']:
        for name in results.keys():
            # x = results[name]['Dates']
            # y = results[name][metric]
            fig.add_trace(go.Scatter(x=results[name]['Dates'], y=results[name][metric], name=name))
    fig.update_layout(title=f"{metric} over time", xaxis_title='Date', yaxis_title=metric)
    # Make sure directory exists
    if not os.path.exists(f'../Images/{dataset}'):
        os.makedirs(f'../Images/{dataset}')
    # Save figure as png
    fig.write_image(f'../Images/{dataset}/{metric}.png')

def get_X_Y(df):
    """Split DataFrame into X and Y"""
    X = [df.iloc[i-FITTING_PERIOD:i, :] for i in range(FITTING_PERIOD, len(df), HOLDING_PERIOD)]
    Y = [df.iloc[i:i+HOLDING_PERIOD, :] for i in range(FITTING_PERIOD, len(df), HOLDING_PERIOD)]
    if len(Y[-1]) != HOLDING_PERIOD:
        X.pop()
        Y.pop()
    return X, Y

# -------------------- Strategies ------------------ #
class RandomWeights:
    def get_weights(self, returns):
        number_of_instruments = returns.shape[1]
        weights = np.random.random(number_of_instruments)
        return weights / np.sum(weights)

class UniformWeights:
    def get_weights(self, returns):
        number_of_instruments = returns.shape[1]
        return np.ones(number_of_instruments) / number_of_instruments

class MarkowitzWeights:
    def neg_sharpe(self, weights, mean_returns, cov_matrix):
        return -((mean_returns.T @ weights) / np.sqrt(weights.T @ cov_matrix @ weights))

    def get_weights(self, returns):
        number_of_instruments = returns.shape[1]
        mean_returns = returns.mean(axis=0)
        cov_matrix = np.cov(returns.T)
        results = sco.minimize(self.neg_sharpe, number_of_instruments*[1./number_of_instruments,], args=(mean_returns, cov_matrix), method='SLSQP', bounds=[(0, 1)]*number_of_instruments, constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        return results['x']

class RLWeights:
    def __init__(self):
        self.env = None
        policy_dir = os.path.join(os.getcwd(), '..', f'Policy_{FITTING_PERIOD}_{HOLDING_PERIOD}')
        self.policy = tf.saved_model.load(policy_dir)
    
    def set_env(self, type):
        gym.envs.registration.register(
            id=f'Eval{type}Env',
            entry_point=f'{__name__}:MarketEnvironment',
        )
        env = tf_py_environment.TFPyEnvironment(suite_gym.load(f'Eval{type}Env', gym_kwargs={'type': type}))
        self.env = env

    def get_weights(self, returns):
        if self.env is None:
            print("Error: Environment not set")
        time_step = self.env.current_time_step()
        action_step = self.policy.action(time_step)
        self.env.step(action_step.action)
        action = action_step.action.numpy()[0]
        print(action/np.sum(action))
        return action / np.sum(action)
        # number_of_instruments = returns.shape[1]
        # return np.ones(number_of_instruments) / number_of_instruments

# -------------------- Main ------------------ #
if __name__ == '__main__':
    strategies = {
        'Random' : RandomWeights(),
        'Uniform' : UniformWeights(),
        'Markowitz' : MarkowitzWeights(),
        'RL' : RLWeights()
    }
    # Get returns DataFrame
    train_returns_df, test_returns_df = get_returns_df()
    print("Training length:", len(train_returns_df))
    print("Testing length:", len(test_returns_df))
    # Split DataFrame into X and Y
    train_X, train_Y = get_X_Y(train_returns_df)
    test_X, test_Y = get_X_Y(test_returns_df)
    print("Number of training examples:", len(train_Y), "Hence effective training length:", len(train_Y)*HOLDING_PERIOD, "days")
    print("Number of testing examples:", len(test_Y), "Hence effective testing length:", len(test_Y)*HOLDING_PERIOD, "days")
    # Initialize results dictionary
    train_results, test_results = {}, {}
    # Iterate over strategies
    for name, strategy in strategies.items():
        # print(f"Evaluating {name} strategy")
        if name == 'RL':
            strategy.set_env('train')
        # Initialize list to store evaluated portfolio dictionary
        portfolio_metrics = []
        # Iterate over X and Y
        for X, Y in zip(train_X, train_Y):
            returns_X = X.iloc[:, 1:].values
            weights = strategy.get_weights(returns_X)
            returns_Y = Y.iloc[:, 1:].values
            # Evaluate portfolio
            portfolio_metric = evaluate_portfolio(returns_Y, weights)
            portfolio_metric['Dates'] = Y['Date'].values
            portfolio_metrics.append(portfolio_metric)
        # Store evaluated portfolio dictionary in results dictionary
        train_results[name] = combine_portfolio_metrics(portfolio_metrics)
        
        if name == 'RL':
            strategy.set_env('test')
        # Initialize list to store evaluated portfolio dictionary
        portfolio_metrics = []
        # Iterate over X and Y
        for X, Y in zip(test_X, test_Y):
            returns_X = X.iloc[:, 1:].values
            weights = strategy.get_weights(returns_X)
            # print(weights)
            returns_Y = Y.iloc[:, 1:].values
            # Evaluate portfolio
            portfolio_metric = evaluate_portfolio(returns_Y, weights)
            portfolio_metric['Dates'] = Y['Date'].values
            portfolio_metrics.append(portfolio_metric)
        # Store evaluated portfolio dictionary in results dictionary
        test_results[name] = combine_portfolio_metrics(portfolio_metrics)

    print("Evaluation complete, plotting results")
    # Plot portfolio statistics
    for metric  in ['Mean Return', 'Volatility', 'Sharpe', 'Max Drawdown', 'Cumulative Return', 'Portfolio Value', 'Portfolio Returns']:
        plot_portfolio_statistics(train_results, metric, dataset='train')
        plot_portfolio_statistics(test_results, metric, dataset='test')
