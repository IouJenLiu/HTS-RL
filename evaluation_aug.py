import copy
import contextlib
import os
import sys


import gfootball.env as football_env
import numpy as np
import torch

from a2c_ppo_acktr.envs import EpisodeRewardScoreWrapper
from torch.distributions import Categorical
from utils import dict2csv


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_q(test_q, models, done_training, args):
    """

    Evaluation Processes

    Args:
        test_q: A shared queue to communicate with the learner process.
        models: Models for evaluation.
        done_training: A shared variable. Set to one when the learn finish its job.
        args: Command line argument.

    Returns:
        None
    """

    plot = {'steps': [], 'left_rewards': [], 'right_rewards': [], 'rewards': [], 'scores': [], 'final_reward': [], 'abs_reward': [],
            'final_score': [], 'abs_score': []}
    best_eval_score_mean = -100000000
    eval_count = 0
    env = football_env.create_environment(
        env_name=args.env_name, stacked=('stacked' in args.state),
        rewards=args.reward_experiment,
        logdir=os.path.join(args.log_dir, args.exp_name, 'trace_video'),
        render=False,
        dump_frequency=1,
        representation=args.representation,
        number_of_left_players_agent_controls=args.num_left_agents,
        write_full_episode_dumps=True,
        write_video=True,
        write_goal_dumps=True,
        other_config_options={'game_engine_random_seed': args.seed + 10})
    local_models = copy.deepcopy(models)
    for agent_idx in range(args.num_agents):
        stat_dict = models[agent_idx].state_dict()
        local_models[agent_idx].load_state_dict(stat_dict)

    if args.num_agents == 1:
        from a2c_ppo_acktr.envs import ObsUnsqueezeWrapper
        env = ObsUnsqueezeWrapper(env)
    env = EpisodeRewardScoreWrapper(env,
                                    number_of_left_players_agent_controls=args.num_left_agents,
                                    number_of_right_players_agent_controls=args.num_right_agents)
    while True:
        if not test_q.empty():
            print('INFO: Start to evaluate')
            test_q.get()
            for agent_idx in range(args.num_agents):
                stat_dict = models[agent_idx].state_dict()
                local_models[agent_idx].load_state_dict(stat_dict)
            eval_rewards, eval_left_rewards, eval_right_rewards = [], [], []
            eval_scores = []
            eval_count += 1
            with temp_seed(args.seed):
                for n_eval in range(args.num_eval_runs):
                    print('INFO: Eval # ', n_eval)
                    obs = env.reset()
                    obs = torch.from_numpy(obs).float()
                    while True:
                        actions = np.zeros(args.num_agents, dtype=int)
                        for agent_idx in range(args.num_agents):
                            with torch.no_grad():
                                kargs = obs[agent_idx:agent_idx+1], None, None
                                _, _, _, action_logit = local_models[agent_idx].act(
                                    *kargs)
                            dist = Categorical(logits=action_logit)
                            action = dist.sample()
                            actions[agent_idx] = int(action.item())
                        obs, reward, done, infos = env.step(
                            actions.reshape(-1))
                        obs = torch.from_numpy(obs).float()
                        if done:
                            eval_left_rewards.append(
                                np.sum(infos['episode_reward'][:args.num_left_agents]))
                            if args.num_right_agents > 0:
                                eval_right_rewards.append(
                                    np.sum(infos['episode_reward'][args.num_left_agents:]))
                            eval_scores.append(infos['episode_score'])
                            break
                if np.mean(eval_scores) > best_eval_score_mean:
                    best_eval_left_reward_mean, best_eval_left_reward_std = np.mean(
                        eval_left_rewards), np.std(eval_left_rewards)
                    best_eval_score_mean, best_eval_score_std = np.mean(
                        eval_scores), np.std(eval_scores)
                plot['steps'].append((eval_count - 1) * args.eval_every_step)
                plot['left_rewards'].append(np.mean(eval_left_rewards))
                if eval_right_rewards:
                    plot['right_rewards'].append(np.mean(eval_right_rewards))
                plot['scores'].append(np.mean(eval_scores))
                plot['final_reward'].append(
                    np.mean(plot['left_rewards'][-10:]))
                plot['final_score'].append(np.mean(plot['scores'][-10:]))
                plot['abs_score'].append(best_eval_score_mean)
                print(
                    "------------Eval Summary------------\n"
                    "Total num env steps: {}, {} eval runs\n"
                    "score avg/std        {:.6f}/{:.6f}\n"
                    "final reward avg/std {:.6f}/{:.6f}\n"
                    "final score avg/std  {:.6f}/{:.6f}\n"
                    "best reward avg/std  {:.6f}/{:.6f}\n"
                    "best score avg/std   {:.6f}/{:.6f}\n"
                    "------------------------------------\n".format(
                        plot['steps'][-1], args.num_eval_runs, np.mean(eval_scores), np.std(
                            eval_scores), np.mean(eval_left_rewards), np.std(eval_left_rewards),
                        np.mean(plot['scores'][-10:]), np.std(plot['scores'][-10:]
                                                              ), best_eval_left_reward_mean, best_eval_left_reward_std,
                        best_eval_score_mean, best_eval_score_std))
                curve_file_path = os.path.join(
                    args.log_dir, args.exp_name, 'train_curve.csv')
                dict2csv(plot, curve_file_path)
                print('INFO: Wrote training curve to ', curve_file_path)
                sys.stdout.flush()

        if done_training.value and test_q.empty():
            print('Finish Evaluation. Exit eval_q()')
            break
    print('Done Evaluation')
    env.close()
