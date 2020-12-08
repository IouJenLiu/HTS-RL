import copy
import os
import pickle

import numpy as np
import torch

from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.base_factory import get_base
from gym.spaces.discrete import Discrete


def actor(actor_rank, action_logits, values, observations, step_dones, act_in_progs, done_list,
          shared_cpu_actor_critics_env_actor, device, observation_space, action_space,
          please_load_model_actor, args, actor_lock):
    """

    Actor grabs observations from the observation buffer and perform forwarding. Then the actor sends the action logits
    and values to the action and values buffers.

    Args:
        actor_rank: actor's id.
        action_logits: A shared PyTorch tensor served as an action buffer.
        values: A shared PyTorch tensor served as a value buffer.
        observations: A shared PyTorch tensor served as an observation buffer.
        step_dones: A shared list to indicate environment processes finish one environment step.
        act_in_progs: A shared array to indicate the observation is being processed by an actor.
        done_list: A shared list that indicates if environment processes finish all steps.
        shared_cpu_actor_critics_env_actor: Shared models between actor and environment processes. Actor processes will
            load models from environment process 0.
        device: CPU/GPU device.
        observation_space: The OpenAI gym observation space of the environment.
        action_space: The OpenAI gym action space of the environment.
        please_load_model_actor: A shared array between actors and the environment process 0. When updated model is
            available. It is set to one. Once an actor finished loading the updated model, it sets its slots to zero.
        args: command line argument.
        actor_lock: A lock to prevent actors from grabbing data which is already being processed by other actors.

    Returns:
        None

    """

    if args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    base_kwargs = {'recurrent': args.recurrent_policy,
                   'hidden_size': args.hidden_size}
    act_in_progs_np = np.frombuffer(act_in_progs.get_obj(), dtype=np.int32)
    step_dones_np = np.frombuffer(step_dones.get_obj(), dtype=np.int32)
    step_dones_np = step_dones_np.reshape(args.num_processes)
    act_in_progs_np = act_in_progs_np.reshape(args.num_processes)
    models = [Policy(
        observation_space.shape[1:],
        action_space if args.num_agents == 1 else Discrete(
            action_space.nvec[0]),
        base=get_base(args.base),
        base_kwargs=base_kwargs).to(device) for _ in range(args.num_agents)]
    for agent_idx in range(args.num_agents):
        stat_dict = shared_cpu_actor_critics_env_actor[agent_idx].state_dict()
        models[agent_idx].load_state_dict(stat_dict)

    steps = 0
    while False in done_list:
        # polling
        if please_load_model_actor[actor_rank] == 1:
            for agent_idx in range(args.num_agents):
                stat_dict = shared_cpu_actor_critics_env_actor[agent_idx].state_dict(
                )
                models[agent_idx].load_state_dict(stat_dict)
            please_load_model_actor[actor_rank] = 0

        if args.cuda_deterministic:
            with actor_lock:
                step_done_not_prog_np = np.logical_and(
                    step_dones_np, act_in_progs_np == 0)
                ranks = np.where(step_done_not_prog_np == 1)[0]
                act_in_progs_np[ranks] = 1
            if ranks.size > 0:
                steps += 1
                for agent_idx in range(args.num_agents):
                    for env_rank in ranks:  # new code
                        with torch.no_grad():
                            obs = copy.deepcopy(
                                observations[env_rank:env_rank + 1].to(device))  # new
                            kargs = obs[:, agent_idx], None, None
                            value, action, action_log_prob, action_logit = models[agent_idx].act(
                                *kargs)
                        action_logits[env_rank,
                                      agent_idx] = action_logit.cpu()  # new
                        values[env_rank, agent_idx] = value.cpu()  # new
                        step_dones_np[env_rank] = 0  # new
                        act_in_progs_np[env_rank] = 0  # new
        else:
            with actor_lock:
                step_done_not_prog_np = np.logical_and(
                    step_dones_np, act_in_progs_np == 0)
                ranks = np.where(step_done_not_prog_np == 1)[0]
                act_in_progs_np[ranks] = 1

            if ranks.size > 0:
                obs = observations[ranks].clone().to(device)
                for agent_idx in range(args.num_agents):
                    with torch.no_grad():
                        kargs = obs[:, agent_idx], None, None
                        value, action, action_log_prob, action_logit = models[agent_idx].act(
                            *kargs)
                    action_logits[ranks, agent_idx] = action_logit.cpu()
                    values[ranks, agent_idx] = value.cpu()
                step_dones_np[ranks] = 0
                act_in_progs_np[ranks] = 0

    print('Done actor ', actor_rank)
