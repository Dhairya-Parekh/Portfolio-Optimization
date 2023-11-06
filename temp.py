#%%
# ------------------- Imports -------------------
import os
import gym
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.agents.ddpg import actor_network, critic_network, ddpg_agent
#%%
# ------------------- Environment Parameters -------------------
FITTING_PERIOD = 60
HOLDING_PERIOD = 1
# %%
# ------------------- Environment -------------------
class SampleEnvironment(gym.Env):

    def __init__(self, filepath, type):
        closing_prices = pd.read_csv(filepath)
        print(f"This is {type} Environment with {len(closing_prices)} rows and {len(closing_prices.columns)} columns")
        closing_prices.sort_values(by=['Date'], inplace=True)
        closing_prices.dropna(inplace=True)
        closing_prices.reset_index(drop=True, inplace=True)
        # Returns are calculated as the percentage change in price not on Date column
        returns_df = closing_prices.drop(columns=['Date']).pct_change().dropna()
        dates = closing_prices['Date']
        self.dates = dates
        self.returns_df = returns_df
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
        done = self.state[0] >= len(self.returns_df)
        # Calculate the info = None
        info = {}
        return observation, reward, done, info
    
    def get_data(self):
        return self.portfolio_values_log, self.dates_log

gym.envs.registration.register(
    id='Env-v1',
    entry_point=f'{__name__}:SampleEnvironment',
)
train_env = tf_py_environment.TFPyEnvironment(suite_gym.load('Env-v1', gym_kwargs={'filepath': './Data/train.csv', 'type': 'Training'}))
val_env = tf_py_environment.TFPyEnvironment(suite_gym.load('Env-v1', gym_kwargs={'filepath': './Data/validation.csv', 'type': 'Validation'}))
test_env = tf_py_environment.TFPyEnvironment(suite_gym.load('Env-v1', gym_kwargs={'filepath': './Data/test.csv', 'type': 'Test'}))
# %%
# ------------------- Data Related Parameters -------------------
# How long should training run?
num_iterations = 20
# How often should the program provide an update.
log_interval = 2
# How many initial random steps, before training start, to collect initial data.
initial_collect_steps = 100
# How many steps should we run each iteration to collect  data from.
collect_steps_per_iteration = 50
# How much data should we store for training examples.
replay_buffer_max_length = 100000
# Batch size for training.
batch_size = 64
# How many episodes should the program use for each evaluation.
num_eval_episodes = 1
# How often should an evaluation occur.
eval_interval = 5

# ------------------- Agent Related Parameters -------------------
actor_fc_layer_params = (256, 256)
critic_obs_fc_layer_params = (256,)
critic_action_fc_layer_params = (256,)
critic_joint_fc_layer_params = (256,)
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
target_update_tau = 0.05
target_update_period = 5
gamma = 0.99
reward_scale_factor = 1.0
gradient_clipping = None
# %%
# ------------------- Agent -------------------
actor_net = actor_network.ActorNetwork(
    train_env.time_step_spec().observation,
    train_env.action_spec(),
    fc_layer_params=actor_fc_layer_params)
critic_net = critic_network.CriticNetwork(
    (train_env.time_step_spec().observation, train_env.action_spec()),
    observation_fc_layer_params=critic_obs_fc_layer_params,
    action_fc_layer_params=critic_action_fc_layer_params,
    joint_fc_layer_params=critic_joint_fc_layer_params)

agent = ddpg_agent.DdpgAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
    ou_stddev=0.2,
    ou_damping=0.15,
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    dqda_clipping=None,
    td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping
    )
agent.initialize()
# %%
# ------------------- Helper Functions -------------------
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

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
# ------------------- Data Collection -------------------
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,batch_size=train_env.batch_size,max_length=replay_buffer_max_length)
collect_data(train_env, random_policy, replay_buffer, steps=initial_collect_steps)

dataset = replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2, num_parallel_calls=12).prefetch(3)

iterator = iter(dataset)
# %%
# ------------------- Training -------------------
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)
avg_return = compute_avg_return(val_env, agent.policy, num_eval_episodes)
returns = [avg_return]
# Start timer for training
start = time.time()
for _ in range(num_iterations):
    print(f"Iteration: {_}")
    collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss
    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print(f'step = {step}: loss = {train_loss}')
    if step % eval_interval == 0:
        avg_return = compute_avg_return(val_env, agent.policy, num_eval_episodes)
        print(f'step = {step}: Average Return = {avg_return}')
        returns.append(avg_return)
    # End timer for training
seconds = time.time() - start
print(f"Training time: {seconds}")
# %%
# ------------------- Save Model -------------------
policy_dir = os.path.join(os.getcwd(), 'SavedModel')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)