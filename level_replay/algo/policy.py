import copy
import torch
import torch.nn.functional as F
from level_replay.algo.dqn import DQN, SimpleDQN, Conv_Q, ATCDQN, ATCEncoder, ATCContrast
from torch.nn.utils import clip_grad_norm_

import numpy as np


class DQNAgent(object):
    def __init__(self, args, env):
        self.device = args.device
        self.action_space = env.action_space.n
        self.batch_size = args.batch_size
        self.norm_clip = args.norm_clip
        self.gamma = args.gamma

        if args.simple_dqn:
            self.Q = SimpleDQN(args, env).to(self.device)
        else:
            self.Q = DQN(args, env).to(self.device)

        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, args.optimizer)(
            self.Q.parameters(), **args.optimizer_parameters
        )
        for param in self.Q_target.parameters():
            param.requires_grad = False

        self.PER = args.PER
        self.n_step = args.multi_step

        self.min_priority = args.min_priority

        # Target update rule
        self.maybe_update_target = (
            self.polyak_target_update if args.polyak_target_update else self.copy_target_update
        )
        self.target_update_frequency = int(args.target_update // args.num_processes)
        self.tau = args.tau

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + env.observation_space.shape
        self.eval_eps = args.eval_eps

        # For seed bar chart
        self.track_seed_weights = args.track_seed_weights
        if self.track_seed_weights:
            self.seed_weights = {
                i: 0 for i in range(args.start_level, args.start_level + args.num_train_seeds)
            }

        if self.Q.c51:
            self.loss = self._loss_c51
        elif self.Q.qrdqn:
            if args.drq:
                self.loss = self._loss_qrdqn_aug
            else:
                self.loss = self._loss_qrdqn
            self.kappa = 1.0
            self.cumulative_density = np.array((2 * np.arange(self.Q.atoms) + 1) / (2.0 * self.Q.atoms))
            self.cumulative_density = torch.from_numpy(self.cumulative_density).to(self.device)
        elif args.drq or args.autodrq:
            self.loss = self._loss_aug
        else:
            self.loss = self._loss

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eps=0.1, eval=False):
        with torch.no_grad():
            q = self.Q(state)
            action = q.argmax(1).reshape(-1, 1)
            max_q = q.max(1)[0]
            mean_q = q.mean(1)
            v = (1 - eps) * max_q + eps * mean_q
            return action, v

    def get_value(self, state, eps=0.1):
        with torch.no_grad():
            q = self.Q(state)
            max_q = q.max(1)[0]
            mean_q = q.mean(1)
            v = (1 - eps) * max_q + eps * mean_q
            return v

    def advantage(self, state, eps):
        q = self.Q(state)
        max_q = q.max(1)[0]
        mean_q = q.mean(1)
        v = (1 - eps) * max_q + eps * mean_q
        return q - v.repeat(q.shape[1]).reshape(-1, q.shape[1])

    def train(self, replay_buffer):
        ind, loss, priority = self.loss(replay_buffer)

        self.Q_optimizer.zero_grad()
        loss.backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.Q.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        grad_magnitude = list(self.Q.named_parameters())[-2][1].grad.clone().norm()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        if self.PER:
            replay_buffer.update_priority(ind, priority)

        return loss, grad_magnitude

    def _loss(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(next_state).gather(
                1, next_action
            )

        current_Q = self.Q(state).gather(1, action)

        loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()
        priority = (current_Q - target_Q).abs().clamp(min=self.min_priority).cpu().data.numpy().flatten()

        return ind, loss, priority

    def _loss_c51(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        log_prob = self.Q.dist(state, log=True)
        log_prob_a = log_prob[range(self.batch_size), action]

        with torch.no_grad():
            next_prob = self.Q.dist(next_state)
            next_dist = self.Q.support.expand_as(next_prob) * next_prob
            argmax_idx = next_dist.sum(-1).argmax(1)

            if self.Q_target.noisy_layers:
                self.Q_target.reset_noise()

            next_prob = self.Q_target.dist(next_state)
            next_prob_a = next_prob[range(self.batch_size), argmax_idx]

            Tz = reward.unsqueeze(1) + not_done * (self.gamma ** self.n_step) * self.Q.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.Q.V_min, max=self.Q.V_max)

            b = (Tz - self.Q.V_min) / self.Q.delta_z
            lower, upper = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            lower[(upper > 0) * (lower == upper)] -= 1
            upper[(lower < (self.Q.atoms - 1)) * (lower == upper)] += 1

            m = state.new_zeros(self.batch_size, self.Q.atoms)
            offset = (
                torch.linspace(0, ((self.batch_size - 1) * self.Q.atoms), self.batch_size)
                .unsqueeze(1)
                .expand(self.batch_size, self.Q.atoms)
                .to(action)
            )
            m.view(-1).index_add_(0, (lower + offset).view(-1), (next_prob_a * (upper.float() - b)).view(-1))
            m.view(-1).index_add_(0, (upper + offset).view(-1), (next_prob_a * (b - lower.float())).view(-1))

        KL = -torch.sum(m * log_prob_a, 1)

        self.Q_optimizer.zero_grad()
        loss = (weights * KL).mean()
        priority = KL.clamp(min=self.min_priority).cpu().data.numpy().flatten()

        return ind, loss, priority

    def _loss_qrdqn(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        with torch.no_grad():
            next_quantiles = self.Q_target.quantiles(next_state)
            next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_greedy_actions = next_greedy_actions.expand(self.batch_size, self.Q.atoms, 1)
            next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
            target_quantiles = reward + not_done * self.gamma ** self.n_step * next_quantiles

        current_quantiles = self.Q.quantiles(state)
        actions = action[..., None].long().expand(self.batch_size, self.Q.atoms, 1)
        current_quantiles = torch.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

        loss, td = self.quantile_huber_loss(current_quantiles, target_quantiles, weights)

        priority = (
            td.abs()
            .clamp(min=self.min_priority)
            .detach()
            .sum(dim=1)
            .mean(dim=1, keepdim=True)
            .cpu()
            .numpy()
            .flatten()
        )

        return ind, loss, priority

    def _loss_qrdqn_aug(self, replay_buffer):
        (
            state,
            action,
            next_state,
            reward,
            not_done,
            seeds,
            ind,
            weights,
            state_aug,
            next_state_aug,
        ) = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        with torch.no_grad():
            next_quantiles = self.Q_target.quantiles(next_state)
            next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_greedy_actions = next_greedy_actions.expand(self.batch_size, self.Q.atoms, 1)
            next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
            target_quantiles = reward + not_done * self.gamma ** self.n_step * next_quantiles

            next_quantiles_aug = self.Q_target.quantiles(next_state_aug)
            next_greedy_actions_aug = next_quantiles_aug.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_greedy_actions_aug = next_greedy_actions_aug.expand(self.batch_size, self.Q.atoms, 1)
            next_quantiles_aug = next_quantiles_aug.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
            target_quantiles_aug = reward + not_done * self.gamma ** self.n_step * next_quantiles_aug

            target_quantiles = (target_quantiles + target_quantiles_aug) / 2

        current_quantiles = self.Q.quantiles(state)
        current_quantiles_aug = self.Q.quantiles(state_aug)
        actions = action[..., None].long().expand(self.batch_size, self.Q.atoms, 1)
        current_quantiles = torch.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)
        current_quantiles_aug = torch.gather(current_quantiles_aug, dim=2, index=actions).squeeze(dim=2)

        loss, td = self.quantile_huber_loss(current_quantiles, target_quantiles, weights)
        loss += self.quantile_huber_loss(current_quantiles_aug, target_quantiles_aug, weights)[0]

        priority = (
            td.abs()
            .clamp(min=self.min_priority)
            .detach()
            .sum(dim=1)
            .mean(dim=1, keepdim=True)
            .cpu()
            .numpy()
            .flatten()
        )

        return ind, loss, priority

    def _loss_aug(self, replay_buffer):
        (
            state,
            action,
            next_state,
            reward,
            not_done,
            seeds,
            ind,
            weights,
            state_aug,
            next_state_aug,
        ) = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(next_state).gather(
                1, next_action
            )

            next_action_aug = self.Q(next_state_aug).argmax(1).reshape(-1, 1)
            target_Q_aug = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(
                next_state_aug
            ).gather(1, next_action_aug)

            target_Q = (target_Q + target_Q_aug) / 2

        current_Q = self.Q(state).gather(1, action)
        current_Q_aug = self.Q(state_aug).gather(1, action)

        loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()
        loss += (weights * F.smooth_l1_loss(current_Q_aug, target_Q, reduction="none")).mean()
        priority = (current_Q - target_Q).abs().clamp(min=self.min_priority).cpu().data.numpy().flatten()

        return ind, loss, priority

    def update_seed_weights(self, seeds, weights):
        if self.track_seed_weights:
            for idx, seed in enumerate(seeds):
                s = seed.cpu().numpy()[0]
                if type(weights) != int and len(weights) > 1:
                    self.seed_weights[s] = self.seed_weights.get(s, 0) + weights[idx].cpu().numpy()[0]
                else:
                    self.seed_weights[s] = self.seed_weights.get(s, 0) + 1
        else:
            pass

    def train_with_online_target(self, replay_buffer, online):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

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

    def huber(self, td_errors, kappa=1.0):
        return torch.where(
            td_errors.abs() <= kappa, 0.5 * td_errors.pow(2), kappa * (td_errors.abs() - 0.5 * kappa)
        )

    def quantile_huber_loss(self, current_quantiles, target_quantiles, weights):
        n_quantiles = current_quantiles.shape[-1]
        cum_prob = (
            torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5
        ) / n_quantiles
        cum_prob = cum_prob.view(1, -1, 1)

        td = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
        huber_loss = self.huber(td)
        loss = torch.abs(cum_prob - (td.detach() < 0).float()) * huber_loss
        loss = (loss.sum(dim=-2) * weights).mean()

        return loss, td

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


class ATCAgent(object):
    def __init__(self, args, env):
        self.device = args.device
        self.action_space = env.action_space.n
        self.batch_size = args.batch_size
        self.norm_clip = args.norm_clip
        self.gamma = args.gamma

        self.Q = ATCDQN(args, env).to(self.device)

        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, args.optimizer)(
            self.Q.parameters(), **args.optimizer_parameters
        )
        for param in self.Q_target.parameters():
            param.requires_grad = False

        self.encoder = ATCEncoder(env).to(self.device)
        self.contrast = ATCContrast().to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.ul_optimizer = getattr(torch.optim, args.optimizer)(
            self.ul_parameters(), **args.optimizer_parameters
        )
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.PER = args.PER
        self.n_step = args.multi_step

        self.min_priority = args.min_priority

        # Target update rule
        self.maybe_update_target = (
            self.polyak_target_update if args.polyak_target_update else self.copy_target_update
        )
        self.target_update_frequency = int(args.target_update // args.num_processes)
        self.tau = args.tau

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + env.observation_space.shape
        self.eval_eps = args.eval_eps

        # For seed bar chart
        self.track_seed_weights = False
        if self.track_seed_weights:
            self.seed_weights = {
                i: 0 for i in range(args.start_level, args.start_level + args.num_train_seeds)
            }

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eps=0.1, eval=False):
        with torch.no_grad():
            features = self.encoder.encode(state)
            q = self.Q(features)
            action = q.argmax(1).reshape(-1, 1)
            max_q = q.max(1)[0]
            mean_q = q.mean(1)
            v = (1 - eps) * max_q + eps * mean_q
            return action, v

    def get_value(self, state, eps=0.1):
        with torch.no_grad():
            features = self.encoder.encode(state)
            q = self.Q(features)
            max_q = q.max(1)[0]
            mean_q = q.mean(1)
            v = (1 - eps) * max_q + eps * mean_q
            return v

    def advantage(self, state, eps):
        features = self.encoder.encode(state)
        q = self.Q(features)
        max_q = q.max(1)[0]
        mean_q = q.mean(1)
        v = (1 - eps) * max_q + eps * mean_q
        return q - v.repeat(q.shape[1]).reshape(-1, q.shape[1])

    def train(self, replay_buffer):
        ind, rl_loss, ul_loss, accuracy = self.loss(replay_buffer)

        self.Q_optimizer.zero_grad()
        rl_loss.backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.Q.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.Q_optimizer.step()

        self.ul_optimizer.zero_grad()
        ul_loss.backward()
        clip_grad_norm_(self.ul_parameters(), self.norm_clip)
        self.ul_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()
        self.atc_target_update()

        return rl_loss, accuracy

    def loss(self, replay_buffer):
        (
            state,
            action,
            next_state,
            reward,
            not_done,
            seeds,
            ind,
            weights,
            state_aug,
            next_state_aug,
        ) = replay_buffer.sample_atc()

        with torch.no_grad():
            next_action = self.Q(self.encoder.encode(next_state)).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(
                self.target_encoder.encode(next_state)
            ).gather(1, next_action)

        current_Q = self.Q(self.encoder.encode(state)).gather(1, action)

        rl_loss = F.smooth_l1_loss(current_Q, target_Q, reduction="none").mean()

        anchor = self.encoder(state_aug)
        with torch.no_grad():
            positive = self.target_encoder(next_state_aug)

        logits = self.contrast(anchor, positive)
        labels = torch.arange(state.shape[0], dtype=torch.long, device=self.device)

        ul_loss = F.cross_entropy(logits, labels)

        correct = torch.argmax(logits.detach(), dim=1) == labels
        accuracy = torch.mean(correct.float())

        return ind, rl_loss, ul_loss, accuracy

    def ul_parameters(self):
        yield from self.encoder.parameters()
        yield from self.contrast.parameters()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def atc_target_update(self):
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)


class AtariAgent(object):
    def __init__(self, args, env):
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

        self.min_priority = args.min_priority

        # Target update rule
        self.maybe_update_target = (
            self.polyak_target_update if args.polyak_target_update else self.copy_target_update
        )
        self.target_update_frequency = args.target_update
        self.tau = args.tau

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + env.observation_space.shape
        self.eval_eps = args.eval_eps

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
