import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from level_replay.algo.dqn import DQN, RainbowDQN, SimpleDQN
from torch.nn.utils import clip_grad_norm_


class Rainbow(object):
    def __init__(self, args):
        self.device = args.device
        self.action_space = args.num_actions
        self.atoms = args.atoms
        self.V_min = args.V_min
        self.V_max = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(
            device=args.device
        )  # Support (range) of z
        self.delta_z = float(args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.norm_clip = args.norm_clip
        self.PER = args.PER and not args.ERE
        self.gamma = args.gamma

        self.Q = RainbowDQN(args, self.action_space).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, args.optimizer)(
            self.Q.parameters(), **args.optimizer_parameters
        )

        for param in self.Q_target.parameters():
            param.requires_grad = False

        self.alpha = args.alpha
        self.min_priority = args.min_priority

        self.maybe_update_target = (
            self.polyak_target_update if args.polyak_target_update else self.copy_target_update
        )
        self.target_update_frequency = args.target_update
        self.tau = args.tau

        self.state_shape = (-1,) + args.state_dim
        self.eval_eps = args.eval_eps
        self.num_actions = args.num_actions

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        with torch.no_grad():
            q = self.Q(state)
            action = q.argmax(1).reshape(-1, 1)
            return action, q.max(1)[0]

    def get_value(self, state):
        with torch.no_grad():
            q = self.Q(state)
            return q.max(1)[0]

    def train(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        log_prob = self.Q.dist(state, log=True)
        log_prob_a = log_prob[range(self.batch_size), action]

        with torch.no_grad():
            next_prob = self.Q.dist(next_state)
            next_dist = self.support.expand_as(next_prob) * next_prob
            argmax_idx = next_dist.sum(-1).argmax(1)

            self.Q_target.reset_noise()

            next_prob = self.Q_target.dist(next_state)
            next_prob_a = next_prob[range(self.batch_size), argmax_idx]

            Tz = reward.unsqueeze(1) + not_done * (self.gamma ** self.n) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.V_min, max=self.V_max)

            b = (Tz - self.V_min) / self.delta_z
            lower, upper = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            lower[(upper > 0) * (lower == upper)] -= 1
            upper[(lower < (self.atoms - 1)) * (lower == upper)] += 1

            m = state.new_zeros(self.batch_size, self.atoms)
            offset = (
                torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size)
                .unsqueeze(1)
                .expand(self.batch_size, self.atoms)
                .to(action)
            )
            m.view(-1).index_add_(0, (lower + offset).view(-1), (next_prob_a * (upper.float() - b)).view(-1))
            m.view(-1).index_add_(0, (upper + offset).view(-1), (next_prob_a * (b - lower.float())).view(-1))

        KL = -torch.sum(m * log_prob_a, 1)

        self.Q_optimizer.zero_grad()
        loss = (weights * KL).mean()
        loss.backward()
        clip_grad_norm_(self.Q.parameters(), self.norm_clip)
        self.Q_optimizer.step()
        grad_magnitude = list(self.Q.named_parameters())[-2][1].grad.clone().norm()

        self.iterations += 1
        self.maybe_update_target()

        if self.PER:
            priority = KL.clamp(min=self.min_priority).cpu().data.numpy().flatten()
            replay_buffer.update_priority(ind, priority)

        return loss, grad_magnitude

    def huber(self, x):
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def reset_noise(self):
        self.Q.reset_noise()

    def save(self, filename):
        torch.save(self.iterations, filename + "iterations")
        torch.save(self.Q.state_dict(), f"{filename}Q_{self.iterations}")
        torch.save(self.Q_optimizer.state_dict(), filename + "optimizer")

    def load(self, filename):
        self.iterations = torch.load(filename + "iterations")
        self.Q.load_state_dict(torch.load(f"{filename}Q_{self.iterations}"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "optimizer"))


class DDQN(object):
    def __init__(self, args):
        self.device = args.device
        self.action_space = args.num_actions
        self.batch_size = args.batch_size
        self.norm_clip = args.norm_clip
        self.gamma = args.gamma

        if args.simple_dqn:
            self.Q = SimpleDQN(args, self.action_space).to(self.device)
        else:
            self.Q = DQN(args, self.action_space).to(self.device)

        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, args.optimizer)(
            self.Q.parameters(), **args.optimizer_parameters
        )
        for param in self.Q_target.parameters():
            param.requires_grad = False

        self.PER = args.PER and not args.ERE
        self.n_step = args.multi_step

        self.alpha = args.alpha
        self.min_priority = args.min_priority

        # Target update rule
        self.maybe_update_target = (
            self.polyak_target_update if args.polyak_target_update else self.copy_target_update
        )
        self.target_update_frequency = args.target_update
        self.tau = args.tau

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + args.state_dim
        self.eval_eps = args.eval_eps
        self.num_actions = args.num_actions

        # For seed bar chart
        self.seed_weights = {i: 0 for i in range(args.start_level, args.start_level + args.num_train_seeds)}

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        with torch.no_grad():
            q = self.Q(state)
            action = q.argmax(1).reshape(-1, 1)
            return action, q.max(1)[0]

    def get_value(self, state):
        with torch.no_grad():
            q = self.Q(state)
            value = q.max(1)[0]
            return value

    def train(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        for idx, seed in enumerate(seeds):
            s = seed.cpu().numpy()[0]
            if self.PER:
                self.seed_weights[s] = self.seed_weights.get(s, 0) + weights[idx].cpu().numpy()[0]
            else:
                self.seed_weights[s] = self.seed_weights.get(s, 0) + 1

        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(next_state).gather(
                1, next_action
            )

        current_Q = self.Q(state).gather(1, action)

        loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()

        self.Q_optimizer.zero_grad()
        loss.backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.Q.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        grad_magnitude = list(self.Q.named_parameters())[-2][1].grad.clone().norm()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        if self.PER:
            priority = ((current_Q - target_Q).abs() + 1e-10).cpu().data.numpy().flatten()
            replay_buffer.update_priority(ind, priority)

        return loss, grad_magnitude

    def train_with_online_target(self, replay_buffer, online):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        for idx, seed in enumerate(seeds):
            s = seed.cpu().numpy()[0]
            if self.PER:
                self.seed_weights[s] = self.seed_weights.get(s, 0) + weights[idx].cpu().numpy()[0]
            else:
                self.seed_weights[s] = self.seed_weights.get(s, 0) + 1

        with torch.no_grad():
            target_Q = reward + not_done * (self.gamma ** self.n_step) * online.get_value(next_state, 0, 0)

        current_Q = self.Q(state).gather(1, action)

        loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()

        self.Q_optimizer.zero_grad()
        loss.backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.Q.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        grad_magnitude = list(self.Q.named_parameters())[-2][1].grad.clone().norm()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        if self.PER:
            priority = ((current_Q - target_Q).abs() + 1e-10).cpu().data.numpy().flatten()
            replay_buffer.update_priority(ind, priority)

        return loss, grad_magnitude

    def huber(self, x):
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.iterations, filename + "iterations")
        torch.save(self.Q.state_dict(), f"{filename}Q_{self.iterations}")
        torch.save(self.Q_optimizer.state_dict(), filename + "optimizer")

    def load(self, filename):
        self.iterations = torch.load(filename + "iterations")
        self.Q.load_state_dict(torch.load(f"{filename}Q_{self.iterations}"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "optimizer"))


class Conv_Q(nn.Module):
    def __init__(self, frames, num_actions):
        super(Conv_Q, self).__init__()
        self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.l1 = nn.Linear(3136, 512)
        self.l2 = nn.Linear(512, num_actions)

    def forward(self, state):
        q = F.relu(self.c1(state))
        q = F.relu(self.c2(q))
        q = F.relu(self.c3(q))
        q = F.relu(self.l1(q.reshape(-1, 3136)))
        return self.l2(q)


class AtariAgent(object):
    # Doesn't use IMPALA features
    def __init__(self, args):
        self.device = args.device
        self.action_space = args.num_actions
        self.batch_size = args.batch_size
        self.norm_clip = args.norm_clip
        self.gamma = args.gamma

        self.Q = Conv_Q(4, self.action_space).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, args.optimizer)(
            self.Q.parameters(), **args.optimizer_parameters
        )

        self.PER = args.PER
        self.n_step = args.multi_step

        self.alpha = args.alpha
        self.min_priority = args.min_priority

        # Target update rule
        self.maybe_update_target = (
            self.polyak_target_update if args.polyak_target_update else self.copy_target_update
        )
        self.target_update_frequency = args.target_update
        self.tau = args.tau

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + args.state_dim
        self.eval_eps = args.eval_eps
        self.num_actions = args.num_actions

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        with torch.no_grad():
            q = self.Q(state)
            action = q.argmax(1).reshape(-1, 1)
            return action, None

    def train(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(next_state).gather(
                1, next_action
            )

        current_Q = self.Q(state).gather(1, action)

        loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()

        self.Q_optimizer.zero_grad()
        loss.backward()  # Backpropagate importance-weighted minibatch loss
        grad_magnitude = list(self.Q.named_parameters())[-2][1].grad.clone().norm()
        # clip_grad_norm_(self.Q.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        if self.PER:
            priority = ((current_Q - target_Q).abs() + 1e-10).pow(0.6).cpu().data.numpy().flatten()
            replay_buffer.update_priority(ind, priority)

        return loss, grad_magnitude

    def huber(self, x):
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.iterations, filename + "iterations")
        torch.save(self.Q.state_dict(), f"{filename}Q_{self.iterations}")
        torch.save(self.Q_optimizer.state_dict(), filename + "optimizer")

    def load(self, filename):
        self.iterations = torch.load(filename + "iterations")
        self.Q.load_state_dict(torch.load(f"{filename}Q_{self.iterations}"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "optimizer"))
