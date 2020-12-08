import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: ppo')
    parser.add_argument(
        '--lr', type=float, default=0.000343, help='learning rate (default: 0.000343)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.993,
        help='discount factor for rewards (default: 0.993)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.003,
        help='entropy term coefficient (default: 0.003)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.64,
        help='max norm of gradients (default: 0.64)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-actors',
        type=int,
        default=1,
        help='1')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=512,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=2,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=8,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.08,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=5e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='academy_empty_goal_close',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/fpsrl_cp_resutls',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')

    # gfootball arguments
    parser.add_argument('--state', type=str, default='extracted_stacked',
                        help="[ 'extracted', 'extracted_stacked'], Observation to be used for training.")
    parser.add_argument('--reward_experiment', type=str, default='scoring,checkpoints',
                        help=" ['scoring', 'scoring,checkpoints'], Reward to be used for training.")
    parser.add_argument('--dump_full_episodes', default=False, action='store_true',
                        help='If True, trace is dumped after every episode.')
    parser.add_argument('--dump_scores', default=False, action='store_true',
                        help='If True, sampled traces after scoring are dumped.')
    parser.add_argument('--dump_freq', type=int, default=50)
    parser.add_argument('--render', default=False, action='store_true',
                        help='If True, environment rendering is enabled.')
    parser.add_argument('--exp-name', default='test')
    parser.add_argument('--sync-every', type=int, default=512)
    parser.add_argument('--eval-freq', type=int, default=100000)
    parser.add_argument('--eval-every-step', type=int, default=100000)
    parser.add_argument('--num-eval-runs', type=int, default=10)
    parser.add_argument('--num-agents', type=int, default=1)
    parser.add_argument('--num-right-agents', type=int, default=0)
    parser.add_argument('--num-left-agents', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--base', type=str, default='None', help="['CNNBaseGfootball']")
    parser.add_argument('--representation', type=str, default='extracted', help='extracted | pixels_gray')
    parser.add_argument('--n_render_gpu', type=int, default=6)
    parser.add_argument('--noop', default=0, type=int)
    parser.add_argument('--mul-gpu', default=False, action='store_true')
    parser.add_argument('--dump_traj_flag', default=False, action='store_true')
    parser.add_argument('--dump_run_id', default=0, type=int)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.render = True if args.representation == 'pixels_gray' else False
    assert args.algo in ['ppo']
    return args
