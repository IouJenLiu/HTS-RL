import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

torch.manual_seed(0)
class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[-1], **base_kwargs)
        self.policy = nn.Linear(base_kwargs['hidden_size'], action_space.n)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        logits = self.policy(actor_features)
        return value, None, None, logits


    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        logits = self.policy(actor_features)
        dist_log = F.log_softmax(logits, dim=-1)
        dist = F.softmax(logits, dim=-1)
        action_log_probs = torch.gather(dist_log, 1, action)

        dist_entropy = torch.mean(dist * dist_log)
        return value, action_log_probs, dist_entropy, rnn_hxs


class PolicyShareBase(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, num_agents=1):
        super(PolicyShareBase, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[2], **base_kwargs)
        num_outputs = action_space.n
        self.dists = nn.ModuleList([Categorical(self.base.output_size, num_outputs) for _ in range(num_agents)])
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.critic_linears = nn.ModuleList([init_(nn.Linear(self.base.output_size, 1)) for _ in range(num_agents)])
        self.num_agents = num_agents


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        bz = int(inputs[0].size()[0] / self.num_agents)
        dists = [self.dists[i](actor_features[bz * i: bz * (i + 1)]) for i in range(self.num_agents)]
        if deterministic:
            actions = torch.cat([dists[i].mode() for i in range(self.num_agents)], dim=0)
        else:
            actions = torch.cat([dists[i].sample() for i in range(self.num_agents)], dim=0)

        action_log_probs = torch.cat([dists[i].log_probs(actions[i]) for i in range(self.num_agents)], dim=0)
        values = torch.cat([self.critic_linears[i](actor_features[bz * i: bz * (i + 1)])
                            for i in range(self.num_agents)], dim=0)

        return values, actions, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        """
        :param inputs: (obs: [(num_agent * bz), *obs_shape], aug_obs: [(num_agent * bz), aug_obs_dim])
        :param rnn_hxs: None
        :param masks:
        :return:
        """
        bz = int(inputs[0].size()[0] / self.num_agents)
        actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        values = [self.critic_linears[i](actor_features[bz * i: bz * (i + 1)]) for i in range(self.num_agents)]
        return torch.cat(values, dim=0)

    def evaluate_actions(self, inputs, rnn_hxs, masks, actions):
        assert inputs[0].size()[0] % self.num_agents == 0
        bz = int(inputs[0].size()[0] / self.num_agents)
        actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        torch.set_printoptions(profile="full")

        torch.set_printoptions(profile="default")
        values = [self.critic_linears[i](actor_features[bz * i: bz * (i + 1)]) for i in range(self.num_agents)]
        dists = [self.dists[i](actor_features[bz * i: bz * (i + 1)]) for i in range(self.num_agents)]
        action_log_probs = [dists[i].log_probs(actions[bz * i: bz * (i + 1)]) for i in range(self.num_agents)]
        dist_entropys = [dists[i].entropy().mean() for i in range(self.num_agents)]
        return torch.cat(values, dim=0), torch.cat(action_log_probs, dim=0), dist_entropys[0], rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=2))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))
        self.fc4 = init_(nn.Linear(32 * 19 * 13, hidden_size))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        #x = self.main(inputs / 255.0)
        inputs = inputs.transpose(1, 3)
        x = F.relu(self.conv1(inputs / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(-1, 32 * 19 * 13)))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class CNNAtariBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNAtariBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 64, 3, stride=1))
        self.fc4 = init_(nn.Linear(3136, hidden_size))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        #x = self.main(inputs / 255.0)
        inputs = inputs.transpose(1, 3)
        x = F.relu(self.conv1(inputs / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(-1, 3136)))

        return self.critic_linear(x), x, rnn_hxs

class CNNBaseSmall(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBaseSmall, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 16, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 8, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 2, stride=1))
        self.fc4 = init_(nn.Linear(32 * 6 * 3, hidden_size))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        #x = self.main(inputs / 255.0)
        #print('inputs', inputs.size())
        inputs = inputs.transpose(1, 3)
        x = F.relu(self.conv1(inputs / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(-1, 32 * 6 * 3)))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class CNNBaseGfootball(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBaseGfootball, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 64, 3, stride=1))
        self.fc4 = init_(nn.Linear(2560, hidden_size))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        inputs = inputs.transpose(1, 3)
        x = F.relu(self.conv1(inputs / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(-1, 64 * 8 * 5)))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
