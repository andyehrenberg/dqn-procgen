import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from level_replay.algo.dqn import ImpalaCNN
from level_replay.algo.policy import DQNAgent


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    target.data.copy_(target.data * (1.0 - tau) + source.data * tau)


def update(optimizer, loss, model, norm_clip=None):
    optimizer.zero_grad()
    loss.backward()
    if norm_clip is not None:
        clip_grad_norm_(model.parameters(), norm_clip)
    optimizer.step()


class ErrorModel(nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.features = ImpalaCNN(env.observation_space.shape[0])
        self.conv_output_size = 2048
        self.fc1 = nn.Linear(self.conv_output_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, args.num_actions)
        self.apply(initialize_weights_xavier)

    def forward(self, input, action):
        x = self.features(input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        error = x.gather(1, action)

        return error


class DisCor(DQNAgent):
    def __init__(self, args, env):
        super().__init__(args, env)
        args.error_lr = 2.5e-4

        self.online_error = ErrorModel(args, env).to(self.device)
        self.target_error = copy.deepcopy(self.online_error)
        self.target_error.load_state_dict(self.online_error.state_dict())

        for param in self.target_error.parameters():
            param.requires_grad = False

        self.error_optimizer = Adam(self.online_error.parameters(), lr=args.error_lr)

        self.tau = torch.tensor(args.tau_init, device=args.device, requires_grad=False)
        self.target_update_coef = 0.005

    def update_targets(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.target_error.load_state_dict(self.online_error.state_dict())

    def learn(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, _ = replay_buffer.sample()

        weights = self.calc_importance_weights(next_state, not_done)

        for idx, seed in enumerate(seeds):
            s = seed.cpu().numpy()[0]
            self.seed_weights[s] = self.seed_weights.get(s, 0) + weights[idx].cpu().numpy()[0]

        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(next_state).gather(
                1, next_action
            )

        current_Q = self.Q(state).gather(1, action)

        q_loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()
        update(self.Q_optimizer, q_loss, self.Q, self.norm_clip)

        current_error = self.calc_online_error(state, action)
        target_error = self.calc_target_error(next_state, not_done, current_Q, target_Q)

        print("Curr: ", current_error.mean())
        print("Target: ", target_error.mean())

        error_loss = self.calc_error_loss(current_error, target_error)

        update(self.error_optimizer, error_loss, self.online_error)

        self.iterations += 1
        if self.iterations % self.target_update_frequency == 0:
            self.update_targets()

        return q_loss, error_loss

    def calc_importance_weights(self, next_state, not_done):
        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1).reshape(-1, 1)
            next_error = self.target_error(next_state, next_action)

        x = -(not_done) * self.gamma * next_error / self.tau

        weight = F.softmax(x, dim=0)

        return weight

    def calc_online_error(self, state, action):
        error = self.online_error(state, action)
        return error

    def calc_target_error(self, next_state, not_done, curr_q, target_q):
        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1).reshape(-1, 1)
            next_error = self.target_error(next_state, next_action)

            target_error = (curr_q - target_q).abs() + not_done * self.gamma * next_error

        return target_error

    def calc_error_loss(self, curr_error, target_error):
        error_loss = torch.mean((curr_error - target_error).pow(2))

        soft_update(self.tau, curr_error.detach().mean(), self.target_update_coef)

        return error_loss
