import copy
import ctypes
import gfootball.env as football_env
import torch
import torch.multiprocessing as _mp

from a2c_ppo_acktr.base_factory import get_base
from a2c_ppo_acktr.model import Policy
from create_env import create_atari_mjc_env
from gym.spaces.discrete import Discrete

mp = _mp.get_context('spawn')
Value = mp.Value


def init_shared_var(action_space, observation_space, aug_obs_dim,
                    num_processes, num_agents, num_actors):
    manager = mp.Manager()
    shared_list = manager.list([False] * num_processes)
    done_list = manager.list([False] * num_processes)
    actions = torch.zeros(num_processes, num_agents, 1).long()
    action_log_probs = torch.zeros(num_processes, num_agents, 1)
    action_logits = torch.zeros(num_processes, num_agents, action_space.n)
    values = torch.zeros(num_processes, num_agents, 1)
    observations = torch.zeros(num_processes, *observation_space.shape)
    aug_observations = torch.zeros(num_processes, num_agents, aug_obs_dim)
    actions.share_memory_(), action_log_probs.share_memory_(
    ), values.share_memory_(), observations.share_memory_()
    aug_observations.share_memory_(), action_logits.share_memory_()
    step_dones = mp.Array(ctypes.c_int32, int(num_processes))
    act_in_progs = mp.Array(ctypes.c_int32, int(num_processes))
    model_updates = mp.Array(ctypes.c_int32, int(num_actors))
    please_load_model = Value('i', 0)
    please_load_model_actor = torch.zeros(int(num_actors)).long()
    all_episode_scores = manager.list()
    return shared_list, done_list, actions, action_log_probs, action_logits, values, observations, aug_observations, \
        step_dones, act_in_progs, model_updates, please_load_model, please_load_model_actor, all_episode_scores


def init_policies(observation_space, action_space, base_kwargs,
                  num_agents, base):
    actor_critics = [Policy(
        observation_space.shape[1:],
        action_space if num_agents == 1 else Discrete(action_space.nvec[0]),
        base=get_base(base),
        base_kwargs=base_kwargs) for _ in range(num_agents)]
    shared_cpu_actor_critics = [Policy(
        observation_space.shape[1:],
        action_space if num_agents == 1 else Discrete(action_space.nvec[0]),
        base=get_base(base),
        base_kwargs=base_kwargs).share_memory() for _ in range(num_agents)]
    shared_cpu_actor_critics_env_actor = [Policy(
        observation_space.shape[1:],
        action_space if num_agents == 1 else Discrete(action_space.nvec[0]),
        base=get_base(base),
        base_kwargs=base_kwargs).share_memory() for _ in range(num_agents)]
    pytorch_total_params = sum(
        p.numel() for p in actor_critics[0].parameters() if p.requires_grad)
    print('number of params ', pytorch_total_params)
    return actor_critics, shared_cpu_actor_critics, shared_cpu_actor_critics_env_actor


def get_policy_arg(hidden_size):
    base_kwargs = {'recurrent': False, 'hidden_size': hidden_size}
    aug_obs_dim = 0
    return base_kwargs, aug_obs_dim


def get_env_info(env_name, state, reward_experiment, num_left_agents,
                 num_right_agents, representation, render, seed, num_agents):
    is_football = '11' in env_name or 'academy' in env_name
    if is_football:
        env = football_env.create_environment(
            representation=representation,
            env_name=env_name,
            stacked=('stacked' in state),
            rewards=reward_experiment,
            logdir=None,
            render=render and (seed == 0),
            dump_frequency=50 if render and seed == 0 else 0)
    else:
        env = create_atari_mjc_env(env_name)
    if num_agents == 1:
        from a2c_ppo_acktr.envs import ObsUnsqueezeWrapper
        env = ObsUnsqueezeWrapper(env)
    env.reset()
    num_left_player = env.unwrapped._cached_observation[0]['left_team'].shape[0] if is_football else 1
    num_right_player = env.unwrapped._cached_observation[
        0]['right_team'].shape[0] if is_football else 0
    observation_space = copy.deepcopy(env.observation_space)
    action_space = copy.deepcopy(env.action_space)
    env.close()
    return num_left_player, num_right_player, observation_space, action_space
