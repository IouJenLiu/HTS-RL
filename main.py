
import random
import os

import numpy as np
import torch
import torch.multiprocessing as _mp

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.storage_ma import RolloutStorageMA
from actor import actor
from env_step import env_step
from evaluation_aug import eval_q
from gym.spaces.discrete import Discrete
from helper import get_env_info, get_policy_arg, init_policies, init_shared_var
from learner import learn


args = get_args()
torch.manual_seed(args.seed)


mp = _mp.get_context('spawn')
Value = mp.Value


def main():
    print("=================Arguments==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")
    # === Setup seed and log path === #
    torch.set_deterministic(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # === Get env info or init models and storage === #
    num_left_player, num_right_player, observation_space, action_space \
        = get_env_info(args.env_name, args.state, args.reward_experiment, args.num_left_agents,
                       args.num_right_agents, args.representation, args.render, args.seed, args.num_agents)

    # === Setup arguments for initializing policy === #
    base_kwargs, aug_obs_dim = get_policy_arg(args.hidden_size)

    # === Initialize Policy === #
    actor_critics, shared_cpu_actor_critics, shared_cpu_actor_critics_env_actor \
        = init_policies(observation_space, action_space, base_kwargs, args.num_agents, args.base)

    # === Initialize Agent === #
    agents = [algo.PPOAug(
        actor_critics[i],
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm) for i in range(args.num_agents)]

    # === Initialize storage === #
    # Two rollout storage, one for env process writing, one for learner process reading. Avoid deep memory copy.
    rollout_storages = [RolloutStorageMA(args.sync_every, args.num_processes, observation_space.shape[1:],
                                         action_space if args.num_agents == 1 else Discrete(
                                             action_space.nvec[0]),
                                         recurrent_hidden_state_size=1,
                                         num_agents=args.num_agents, aug_size=aug_obs_dim) for _ in range(2)]
    [rollout_storages[i].share_memory() for i in range(2)]

    # === Initzlize shared variables === #
    shared_list, done_list, actions, action_log_probs, action_logits, values, observations, aug_observations, \
        step_dones, act_in_progs, model_updates, please_load_model, please_load_model_actor, all_episode_scores \
        = init_shared_var(action_space, observation_space, aug_obs_dim, args.num_processes,
                          args.num_agents, args.num_actors)

    # === Launch Processes === #
    processes = []

    # eval process
    test_q = mp.Queue()
    done_training = Value('i', False)
    if args.eval_freq > 0:
        p = mp.Process(target=eval_q, args=(
            test_q, shared_cpu_actor_critics, done_training, args))
        p.start()
        processes.append(p)
        test_q.put(1)

    # learner process
    p = mp.Process(target=learn, args=(
        shared_list, done_list, rollout_storages, test_q, done_training,
        torch.device("cuda:{}".format(0) if args.cuda else "cpu"),
        agents, shared_cpu_actor_critics, please_load_model, args))
    p.start()
    processes.append(p)

    # env processes
    for rank in range(0, args.num_processes):
        vgl_display = ':0.{}'.format(rank % args.n_render_gpu)
        p = mp.Process(target=env_step,
                       args=(
                           rank, args, action_logits, values, observations,
                           rollout_storages, shared_list, done_list,
                           step_dones, please_load_model, please_load_model_actor, shared_cpu_actor_critics,
                           shared_cpu_actor_critics_env_actor, all_episode_scores, vgl_display))
        p.start()
        processes.append(p)

    # actor processes
    m = mp.Manager()
    actor_lock = m.Lock()
    for actor_rank in range(args.num_actors):
        if args.mul_gpu:
            actor_device = torch.device("cuda:{}".format(
                actor_rank % 3 + 1) if args.cuda else "cpu")
        else:
            actor_device = torch.device(
                "cuda:{}".format(1) if args.cuda else "cpu")
        p = mp.Process(target=actor,
                       args=(
                           actor_rank, action_logits, values, observations, step_dones, act_in_progs, done_list,
                           shared_cpu_actor_critics_env_actor, actor_device, observation_space, action_space,
                           please_load_model_actor, args, actor_lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    main()
    print("Finish main()")
