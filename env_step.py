import copy
import math
import os
import sys
import time

import gfootball.env as football_env
import numpy as np
import torch

from a2c_ppo_acktr.envs import EpisodeRewardScoreWrapper
from a2c_ppo_acktr.envs import GFNoopResetEnv
from a2c_ppo_acktr.storage_ma import RolloutStorageMA
from gym.spaces.discrete import Discrete
from torch.distributions import Categorical
from utils import dict2csv


def env_step(rank, args, action_logits, values, observations,
             rollout_storages, wait, done_list, step_dones, please_load_model, please_load_model_actor,
             shared_cpu_actor_critics, shared_cpu_actor_critics_env_actor, all_episode_scores, vgl_display):
    """

    Environment process grabs action logit from the action buffer and sample an action according to the action logit.
    Then it executes the action and send the next observation to the observation buffer. The transition tuples are
    stroed to data storage.

    Args:
        rank: environment process id.
        args: command line argument.
        action_logits: A shared PyTorch tensor served as an action buffer.
        values: A shared PyTorch tensor served as a value buffer.
        observations:  A shared PyTorch tensor served as an observation buffer.
        rollout_storages: A list of two rollout storage.
        wait: A shared list that indicates if environment processes are waiting for updated model.
        done_list: A shared list that indicates if environment processes finish all steps.
        step_dones: A shared list to indicate environment processes finish one environment step.
        please_load_model: A shared integer. Set to zero when finishing loading the update model from learner.
        please_load_model_actor: A shared array between actors and the environment process 0. When updated model is
            available. It is set to one.
        shared_cpu_actor_critics: A list of shared models. It contains the updated parameters.
        shared_cpu_actor_critics_env_actor: Shared models between actor and environment processes. Actor processes will
            load models from environment process 0.
        all_episode_scores: A shared list that collect all episode score from all environment processes

    Returns:
        None
    """

    os.environ['VGL_DISPLAY'] = vgl_display
    torch.manual_seed(args.seed + rank)

    env = football_env.create_environment(
        representation=args.representation,
        env_name=args.env_name,
        stacked=('stacked' in args.state),
        rewards=args.reward_experiment,
        logdir=args.log_dir,
        render=args.render and (args.seed == 0),
        dump_frequency=50 if args.render and args.seed == 0 else 0,
        other_config_options={'game_engine_random_seed': args.seed + rank})
    env = EpisodeRewardScoreWrapper(env,
                                    number_of_left_players_agent_controls=1,
                                    number_of_right_players_agent_controls=0)
    env.seed(args.seed + rank)
    if args.noop > 0:
        env = GFNoopResetEnv(env, noop_max=args.noop, seed=args.seed + rank)

    if args.num_agents == 1:
        from a2c_ppo_acktr.envs import ObsUnsqueezeWrapper
        env = ObsUnsqueezeWrapper(env)
    env = EpisodeRewardScoreWrapper(env,
                                    number_of_left_players_agent_controls=args.num_left_agents,
                                    number_of_right_players_agent_controls=args.num_right_agents)
    step_dones_np = np.frombuffer(step_dones.get_obj(), dtype=np.int32)
    step_dones_np = step_dones_np.reshape(args.num_processes)

    obs = env.reset()
    aug_feat_dim = 0

    # store the rollout by this process. After args.sync_every steps, batch copy to rollouts
    local_rollouts = RolloutStorageMA(args.sync_every, 1, env.observation_space.shape[1:],
                                      env.action_space if args.num_agents == 1 else Discrete(
                                          env.action_space.nvec[0]),
                                      recurrent_hidden_state_size=1, num_agents=args.num_agents,
                                      aug_size=aug_feat_dim)

    observations[rank] = torch.from_numpy(obs)
    step_dones_np[rank] = 1

    local_rollouts.obs[0].copy_(torch.from_numpy(obs).float().unsqueeze(0))
    num_steps = int(math.ceil(args.num_env_steps / args.num_processes))
    recurrent_hidden_states = torch.ones(1)
    print('Num of steps per environment', num_steps)
    sync_count = 0
    target_eval_step = 0

    if rank == 0:
        plot = {'steps': [], 'avg_scores': [], 'time_elapsed': [], 'fps': [], 'avg_rewards': [],
                'final_scores': [], 'final_rewards': [], 'fps_one_sync': []}
    scores = []
    episode_rewards = []
    start_sync = time.time()
    start_rollout = time.time()
    env_step_timer_start = time.time()
    if args.dump_traj_flag:
        prev_obs = copy.deepcopy(obs)
        dump_traj = {'action': [], 'obs': [], 'action_logit': [], 'v': []}
    for step in range(num_steps):
        # Observe reward and next observation
        while True:
            if step_dones_np[rank] == 0:
                break
        value_pred = values[rank].clone()
        dist = Categorical(logits=copy.deepcopy(action_logits[rank]))
        action = dist.sample()
        action_log_prob = dist.log_probs(action)
        obs, reward, done, infos = env.step(action.numpy().reshape(-1))
        if args.dump_traj_flag:
            dump_traj['action'].append(action)
            dump_traj['obs'].append(prev_obs)
            dump_traj['action_logit'].append(
                copy.deepcopy(action_logits[rank]))
            dump_traj['v'].append(value_pred)
        if done:
            if rank == 0:
                scores.append(infos['episode_score'])
                sys.stdout.flush()
            obs = env.reset()
            episode_rewards.append(
                np.sum(infos['episode_reward'][:args.num_left_agents]))
            all_episode_scores.append(infos['episode_score'])

        prev_obs = copy.deepcopy(obs)
        aug_obs = None
        obs = torch.from_numpy(obs)
        observations[rank] = obs

        masks = torch.FloatTensor([0.0]) if done else torch.FloatTensor([1.0])
        bad_masks = torch.FloatTensor(
            [0.0]) if 'bad_transition' in infos.keys() else torch.FloatTensor([1.0])
        reward = torch.FloatTensor([reward])

        local_rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob,
                              values[rank], reward.resize_(1, args.num_agents, 1), masks, bad_masks, aug_obs=aug_obs)

        if step % args.sync_every == 0 and step != 0:
            per_sync_time = time.time() - start_sync
            if sync_count == 19 and args.dump_traj_flag:
                import pickle
                with open('/tmp/traj_run{}_{}.pkl'.format(args.dump_run_id, rank), 'wb') as output:
                    pickle.dump(dump_traj, output, pickle.HIGHEST_PROTOCOL)
                    print("INFO: Dump trajectories for testing")
            # Copy local rollout to rollout_storage for training
            st_idx = sync_count % 2
            rollout_storages[st_idx].single_process_batch_insert(rank, local_rollouts.obs, local_rollouts.recurrent_hidden_states,
                                                                 local_rollouts.actions,
                                                                 local_rollouts.action_log_probs, local_rollouts.value_preds,
                                                                 local_rollouts.rewards, local_rollouts.masks, local_rollouts.bad_masks,
                                                                 aug_obs=local_rollouts.aug_obs)
            local_rollouts.after_update()
            sync_count += 1
            if rank == 0:
                print('Rollout time                 : {:.6f} s\n'
                      'Last {} episode average score: {:.6f} rew: {:.6f}'.
                      format(time.time() - start_rollout, 10, np.mean(scores[-10:]),
                             np.mean(episode_rewards[-args.num_processes:])))
                sys.stdout.flush()
            wait[rank] = True

            if rank == 0 and sync_count % 100 == 0:
                total_steps = sync_count * args.num_processes * args.sync_every
                target_eval_step += args.eval_freq
                plot['avg_scores'].append(np.mean(scores[-10:]))
                plot['final_scores'].append(np.mean(plot['avg_scores'][-10:]))
                plot['steps'].append(total_steps)
                time_elapsed = time.time() - env_step_timer_start
                plot['time_elapsed'].append(time_elapsed)
                plot['fps'].append(total_steps // time_elapsed)
                plot['fps_one_sync'].append(
                    args.num_processes * args.sync_every // per_sync_time)
                plot['avg_rewards'].append(np.mean(episode_rewards[-10:]))
                plot['final_rewards'].append(
                    np.mean(plot['avg_rewards'][-10:]))
                curve_file_path = os.path.join(
                    args.log_dir, args.exp_name, 'rank0_curve.csv')
                dict2csv(plot, curve_file_path)
                print('Wrote training curve to ', curve_file_path)
                sys.stdout.flush()

            while True:
                # Load the updated model for actor, ask actor to load,
                # wait for ack from actor, and send ack signal back to learner.
                if rank == 0 and please_load_model.value == 1:
                    for agent_idx in range(args.num_agents):
                        stat_dict = shared_cpu_actor_critics[agent_idx].state_dict(
                        )
                        shared_cpu_actor_critics_env_actor[agent_idx].load_state_dict(
                            stat_dict)
                    please_load_model_actor[:] = 1
                    while True:
                        if torch.all(please_load_model_actor == 0).item():
                            break
                    please_load_model.value = 0
                if wait[rank] == False:
                    break
            start_sync = time.time()
            start_rollout = time.time()
        step_dones_np[rank] = 1

    done_list[rank] = True
    print('Done env ', rank)
