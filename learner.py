import os
import random
import sys
import time

import numpy as np
import torch

from a2c_ppo_acktr import utils
from collections import deque


def learn(shared_list, done_list, rollout_storages, test_q,
          done_training, device, agents,
          shared_cpu_actor_critics, please_load_model, args):
    """
    Learn grab data from a data storage and update the parameters.
    The updated parameters are loaded to
    shared_cpu_actor_critics for actors to load.

    Args:
        shared_list: A shared list that indicates if environment processes are waiting.
        done_list: A shared list that indicates if environment processes finish all steps.
        rollout_storages : A list of two rollout storage.
        test_q: A shared queue to communicate with the evaluation process.
        done_training: A shared variable. Set to one when the learn finish its job.
        device: CPU/GPU device.
        agents: A list of models. Used to update parameters.
        shared_cpu_actor_critics: A list of shared models. It contains the updated parameters.
        please_load_model: A shared integer. Set to one when the updated mode is ready for loading.
        args: Command line arguments.

    Returns:
        None
    """

    if args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    sync_count = 0
    target_eval_step = args.eval_freq

    start_sync_step = time.time()
    for agent_idx in range(args.num_agents):
        agents[agent_idx].actor_critic.to(device)
    left_agent_idx = [i for i in range(args.num_left_agents)]
    right_agent_idx = [i for i in range(args.num_left_agents, args.num_agents)]
    update_times = [0]
    num_updates = int(
        args.num_env_steps) // args.sync_every // args.num_processes
    fps_log = deque(maxlen=10)

    while True:
        if False not in shared_list:  # all env process waiting
            st_idx = sync_count % 2
            agents_to_train = left_agent_idx + right_agent_idx
            sync_count += 1
            # ask env process to load the updated model, and wait for acknowledgement
            please_load_model.value = 1
            while True:
                if please_load_model.value == 0:
                    break
            total_steps = sync_count * args.num_processes * args.sync_every
            fps = int((args.sync_every * args.num_processes) /
                      (time.time() - start_sync_step))
            fps_log.append(fps)

            print('---------------------\n'
                  'SYNC         : {}\n'
                  'Steps        : {}\n'
                  'Sync SPS     : {}\n'
                  'Average SPS  : {}\n'
                  'Sync time    : {:.6f}\n'
                  'Update time  : {:.6f}\n'
                  '---------------------'.format(
                      sync_count, total_steps, fps, np.mean(fps_log), time.time() - start_sync_step, update_times[-1]))
            sys.stdout.flush()
            start_sync_step = time.time()
            start_update = time.time()

            for i in range(len(shared_list)):
                shared_list[i] = False

            for agent_idx in agents_to_train:
                # update model
                if args.use_linear_lr_decay:
                    # decrease learning rate linearly
                    utils.update_linear_schedule(
                        agents[agent_idx].optimizer, sync_count, num_updates, args.lr)
                one_agent_rollout = rollout_storages[st_idx].get_one_agent_rollout(
                    agent_idx, is_aug=False)
                one_agent_rollout.to(device)
                with torch.no_grad():
                    input_obs = one_agent_rollout.obs[-1]
                    next_value = agents[agent_idx].actor_critic.get_value(
                        input_obs, one_agent_rollout.recurrent_hidden_states[-1],
                        one_agent_rollout.masks[-1]).detach()

                one_agent_rollout.compute_returns(next_value, args.use_gae,
                                                  args.gamma,
                                                  args.gae_lambda,
                                                  args.use_proper_time_limits)
                value_loss, action_loss, dist_entropy = agents[agent_idx].update(
                    one_agent_rollout)
                shared_cpu_actor_critics[agent_idx].load_state_dict(
                    agents[agent_idx].actor_critic.state_dict())
            update_times.append(time.time() - start_update)

            if args.eval_every_step > 0 and total_steps >= target_eval_step:
                target_eval_step += args.eval_every_step
                if not test_q.empty():
                    print(
                        "Wanring: eval slower than training, please decrease eval_freq")
                test_q.put(1)
                print('INFO: Sent model for evaluation')
                sys.stdout.flush()

        if False not in done_list:
            break

    done_training.value = True
    print('Done Learning')
