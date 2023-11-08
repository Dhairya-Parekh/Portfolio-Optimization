# ------------------- Imports -------------------
import os
import gym
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.utils import common
from Environment import MarketEnvironment, FITTING_PERIOD, HOLDING_PERIOD
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.agents.ddpg import actor_network, critic_network, ddpg_agent

gym.envs.registration.register(
    id='TrainingEnv',
    entry_point=f'{__name__}:MarketEnvironment',
)
train_env = tf_py_environment.TFPyEnvironment(suite_gym.load('TrainingEnv', gym_kwargs={'type': 'train'}))
val_env = tf_py_environment.TFPyEnvironment(suite_gym.load('TrainingEnv', gym_kwargs={'type': 'validation'}))

# ------------------- Data Related Parameters -------------------
# How long should training run?
num_iterations = 3000
# How often should the program provide an update.
log_interval = 20
# How many initial random steps, before training start, to collect initial data.
initial_collect_steps = 1000
# How many steps should we run each iteration to collect  data from.
collect_steps_per_iteration = 50
# How much data should we store for training examples.
replay_buffer_max_length = 100000
# Batch size for training.
batch_size = 64
# How many episodes should the program use for each evaluation.
num_eval_episodes = 1
# How often should an evaluation occur.
eval_interval = 50

# ------------------- Agent Related Parameters -------------------
actor_fc_layer_params = (400, 300)
critic_obs_fc_layer_params = (400,)
critic_action_fc_layer_params = (300,)
critic_joint_fc_layer_params = (300,)
actor_learning_rate = 1e-3
critic_learning_rate = 5e-3
target_update_tau = 0.05
target_update_period = 5
gamma = 0.99
reward_scale_factor = 1.0
gradient_clipping = None

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

# actor_net = actor_rnn_network.ActorRnnNetwork(
#     train_env.time_step_spec().observation,
#     train_env.action_spec(),
#     input_fc_layer_params=actor_fc_layer_params,
#     # lstm_size=(18,),
#     # output_fc_layer_params=(100,)
#     )
# critic_net = critic_rnn_network.CriticRnnNetwork(
#     (train_env.time_step_spec().observation, train_env.action_spec()),
#     observation_fc_layer_params=critic_obs_fc_layer_params,
#     action_fc_layer_params=critic_action_fc_layer_params,
#     joint_fc_layer_params=critic_joint_fc_layer_params,
#     lstm_size=(100,),
#     # output_fc_layer_params=(100,)
#     )


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
    # td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping
    )
agent.initialize()

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

# ------------------- Data Collection -------------------
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,batch_size=train_env.batch_size,max_length=replay_buffer_max_length)
collect_data(train_env, random_policy, replay_buffer, steps=initial_collect_steps)

dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

iterator = iter(dataset)

# ------------------- Training -------------------
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)
avg_return = compute_avg_return(val_env, agent.policy, num_eval_episodes)
returns = [avg_return]
# Start timer for training
start = time.time()
for _ in range(num_iterations):
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

# ------------------- Save Model -------------------
policy_dir = os.path.join(os.getcwd(),'..',f'Policy_{FITTING_PERIOD}_{HOLDING_PERIOD}')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)