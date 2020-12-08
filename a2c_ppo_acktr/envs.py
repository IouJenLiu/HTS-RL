import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from baselines.bench import monitor
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, NoopResetEnv
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
import time
try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass
import gfootball.env as football_env


def make_env(env_id, seed, rank, log_dir, allow_early_resets, state,
             reward_experiment, dump_scores, dump_full_episodes, render):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif 'academy' in env_id or '11' in env_id: # gfootball environments
            #env = football_env.create_environment(
            #          env_name=env_id, stacked=('stacked' in state),
            #          with_checkpoints=('with_checkpoints' in reward_experiment),
            #          logdir=log_dir,
            #          enable_goal_videos=dump_scores and (seed == 0),
            #          enable_full_episode_videos=dump_full_episodes and (seed == 0),
            #          render=render and (seed == 0),
            #          dump_frequency=50 if render and seed == 0 else 0,
            #          representation='extracted')
            env = football_env.create_environment(
                env_name=env_id, stacked=('stacked' in state),
                rewards=reward_experiment,
                logdir=log_dir,
                render=render and (seed == 0),
                dump_frequency=50 if render and seed == 0 else 0)
            env = EpisodeRewardScoreWrapper(env,
                                            number_of_left_players_agent_controls=1,
                                            number_of_right_players_agent_controls=0)

        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)



        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env, frame_stack=True)
        elif len(env.observation_space.shape) == 3 and 'academy' not in env_id and '11' not in env_id:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")


        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])
        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None,
                  state=None,
                  reward_experiment=None,
                  dump_scores=None,
                  dump_full_episodes=None,
                  render=None
                  ):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, state,
             reward_experiment, dump_scores, dump_full_episodes, render)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 1, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types
        self.total_step_wait_time = 0
        self.start = 0
        self.steps = 0

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        #if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
        #    actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)
        self.start = time.time()
        self.steps += 1

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        self.total_step_wait_time += time.time() - self.start
        if self.steps == 400:
            print('avg step wait time', self.total_step_wait_time / self.steps)
        into_to_device = time.time()
        obs = torch.from_numpy(obs).float().to(self.device)
        #print('[envs] to_device {}'.format(time.time() - into_to_device))
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class EpisodeRewardScoreWrapper(gym.Wrapper):
    def __init__(self, env, number_of_left_players_agent_controls, number_of_right_players_agent_controls):
        super().__init__(env)
        self.n_left_agent = number_of_left_players_agent_controls
        self.n_right_agent = number_of_right_players_agent_controls

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        #if 'score_reward' in info:
        #    if info['score_reward'] == 0:
        #        rew[self.n_left_agent:-1] -= sum(rew[:self.n_left_agent])
        #    if done and info['score_reward'] == 0:
        #        rew[self.n_left_agent:] += 1
        #
        self.episode_score += info['score_reward']
        self.episode_reward += rew
        #self.episode_reward[self.n_left_agent:] -= sum(rew[:self.n_left_agent])

        if done:
            info['bad_transition'] = True
            info['episode_reward'] = self.episode_reward
            info['episode_score'] = self.episode_score
        return obs, rew, done, info

    def reset(self, **kwargs):
        self.episode_reward = 0
        self.episode_score = 0
        return self.env.reset(**kwargs)


class ObsUnsqueezeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space.shape = (1,) + self.observation_space.shape
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs[np.newaxis], np.array([rew]), done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs[np.newaxis]

class GFNoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30, seed=0):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        self.np_random = np.random.RandomState(seed)
    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)