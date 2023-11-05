#%%
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.agents.ddpg import actor_network, critic_network, ddpg_agent
#%%
FITTING_PERIOD = 60
HOLDING_PERIOD = 1
# %%
class SampleEnvironment(gym.Env):

    def __init__(self, filepath, is_train=True):
        closing_prices = pd.read_csv(filepath, index_col=0)
        print(f"This is {'Train' if is_train else 'Test'} Environment with {len(closing_prices)} rows and {len(closing_prices.columns)} columns")
        closing_prices.sort_index(inplace=True)
        closing_prices.dropna(inplace=True)
        returns_df = closing_prices.pct_change().dropna()
        self.returns_df = returns_df
        self.num_assets = returns_df.shape[1]
        # The state is the [date, current_portfolio_value] the action is allocation of portfolio weights to each stock, and the reward is the return of the portfolio
        self.state = None
        self.portfolio_values = None
        self.dates = None
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=np.inf, shape=(self.num_assets,FITTING_PERIOD), dtype=np.float32)
        self.reset()
    
    def step(self, action):
        # Make sure the action is valid, i.e. the sum of weights is 1
        weights = action / np.sum(action)
        current_index = self.returns_df.index.get_loc(self.state[0])
        # Given action as portfolio weights, calculate the portfolio return for the holding period
        holding_returns = self.returns_df.iloc[current_index:current_index+HOLDING_PERIOD]
        portfolio_values = (weights * (1 + holding_returns).cumprod(axis=0)).sum(axis=1)
        cumulative_return = portfolio_values[-1] - 1
        portfolio_values = self.state[1] * portfolio_values
        self.portfolio_values = np.append(self.portfolio_values, portfolio_values)
        # Extract dates from holding returns
        dates = holding_returns.index
        self.dates = np.append(self.dates, dates)
        # Calculate the new state = the next date
        self.state = [dates[-1], portfolio_values[-1]]
        # [self.returns_df.index[self.returns_df.index.get_loc(self.state) + HOLDING_PERIOD], portfolio_values[-1]]
        # Calculate the observation = the returns for the fitting period
        observation = self.returns_df.iloc[current_index+1:current_index+1+FITTING_PERIOD].values.T
        print(f"Observation shape: {observation.shape}, index: {current_index+1} - {current_index+FITTING_PERIOD}")
        # Calculate the reward = portfolio return for the holding period
        reward = cumulative_return
        # Calculate the done flag = whether the new state is the last possible date i.e. adding the holding period to the current index of state exceeds the length of the returns dataframe
        done = self.returns_df.index.get_loc(self.state[0]) + HOLDING_PERIOD >= len(self.returns_df)
        # Calculate the info = None
        info = {}
        return observation, reward, done, info
        
    def reset(self):
        self.state = [self.returns_df.index[FITTING_PERIOD],1]
        self.portfolio_values = np.array([1])
        self.dates = [self.returns_df.index[FITTING_PERIOD-1]]
        observation = self.returns_df.iloc[:FITTING_PERIOD].values.T
        print(f"Observation shape: {observation.shape}")
        return observation
    
    def get_data(self):
        return self.portfolio_values, self.dates
    
# %%
gym.envs.registration.register(
    id='Env-v1',
    entry_point=f'{__name__}:SampleEnvironment',
)
train_env = tf_py_environment.TFPyEnvironment(suite_gym.load('Env-v1', gym_kwargs={'filepath': './Data/train.csv'}))
test_env = tf_py_environment.TFPyEnvironment(suite_gym.load('Env-v1', gym_kwargs={'filepath': './Data/test.csv', 'is_train': False}))
# %%
#---------------Parameters---------------#
batch_size = 64
data_collection_steps = 174

replay_buffer_max_length = 100000

actor_learning_rate = 1e-4
critic_learning_rate = 1e-3

#%%
#---------------Agent---------------#
actor_net = actor_network.ActorNetwork(
    train_env.time_step_spec().observation,
    train_env.action_spec(),
    fc_layer_params=(100, 50))
critic_net = critic_network.CriticNetwork(
    (train_env.time_step_spec().observation, train_env.action_spec()),
    observation_fc_layer_params=(100,),
    action_fc_layer_params=(100,),
    joint_fc_layer_params=(100,))

agent = ddpg_agent.DdpgAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
    )
    # ou_stddev=0.2,
    # ou_damping=0.15,
    # target_update_tau=0.05,
    # target_update_period=5,
    # dqda_clipping=None,
    # td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
    # gamma=0.95,
    # reward_scale_factor=1.0,
    # gradient_clipping=None,
    # debug_summaries=False,
    # summarize_grads_and_vars=False,
    # train_step_counter=tf.Variable(0))
agent.initialize()
#%%
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step,next_time_step)
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

#%%
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

collect_data(train_env, random_policy, replay_buffer, steps=data_collection_steps)

dataset = replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2, num_parallel_calls=12).prefetch(3)
# %%



























    # def __init__(self, filepath, is_train=True):
    #     closing_prices = pd.read_csv(filepath, index_col=0)
    #     print(f"This is {'Train' if is_train else 'Test'} Environment with {len(closing_prices)} rows and {len(closing_prices.columns)} columns")
    #     closing_prices.sort_index(inplace=True)
    #     closing_prices.dropna(inplace=True)
    #     returns_df = closing_prices.pct_change().dropna()
    #     self.returns_df = returns_df
    #     self.num_assets = returns_df.shape[1]
    #     # The state is the [date, current_portfolio_value] the action is allocation of portfolio weights to each stock, and the reward is the return of the portfolio
    #     self.state = None
    #     self.portfolio_values = None
    #     self.dates = None
    #     self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
    #     self.observation_space = gym.spaces.Box(low=-1.0, high=np.inf, shape=(self.num_assets,FITTING_PERIOD), dtype=np.float32)
    #     self.reset()
    
    # def step(self, action):
    #     # Make sure the action is valid, i.e. the sum of weights is 1
    #     weights = action / np.sum(action)
    #     current_index = self.returns_df.index.get_loc(self.state[0])
    #     # Given action as portfolio weights, calculate the portfolio return for the holding period
    #     holding_returns = self.returns_df.iloc[current_index:current_index+HOLDING_PERIOD]
    #     portfolio_values = (weights * (1 + holding_returns).cumprod(axis=0)).sum(axis=1)
    #     cumulative_return = portfolio_values[-1] - 1
    #     portfolio_values = self.state[1] * portfolio_values
    #     self.portfolio_values = np.append(self.portfolio_values, portfolio_values)
    #     # Extract dates from holding returns
    #     dates = holding_returns.index
    #     self.dates = np.append(self.dates, dates)
    #     # Calculate the new state = the next date
    #     self.state = [dates[-1], portfolio_values[-1]]
    #     # [self.returns_df.index[self.returns_df.index.get_loc(self.state) + HOLDING_PERIOD], portfolio_values[-1]]
    #     # Calculate the observation = the returns for the fitting period
    #     observation = self.returns_df.iloc[current_index+1:current_index+1+FITTING_PERIOD].values.T
    #     print(f"Observation shape: {observation.shape}, index: {current_index+1} - {current_index+FITTING_PERIOD}")
    #     # Calculate the reward = portfolio return for the holding period
    #     reward = cumulative_return
    #     # Calculate the done flag = whether the new state is the last possible date i.e. adding the holding period to the current index of state exceeds the length of the returns dataframe
    #     done = self.returns_df.index.get_loc(self.state[0]) + HOLDING_PERIOD >= len(self.returns_df)
    #     # Calculate the info = None
    #     info = {}
    #     return observation, reward, done, info
        
    # def reset(self):
    #     self.state = [self.returns_df.index[FITTING_PERIOD],1]
    #     self.portfolio_values = np.array([1])
    #     self.dates = [self.returns_df.index[FITTING_PERIOD-1]]
    #     observation = self.returns_df.iloc[:FITTING_PERIOD].values.T
    #     print(f"Observation shape: {observation.shape}")
    #     return observation
    
    # def get_data(self):
    #     return self.portfolio_values, self.dates