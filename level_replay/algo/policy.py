import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from level_replay.algo.dqn import DQN
from torch.nn.utils import clip_grad_norm_

class LAP_DDQN(object):
    def __init__(self, args):

        self.device = args.device
        self.action_space = args.num_actions
        self.atoms = args.atoms
        self.V_min = args.V_min
        self.V_max = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.norm_clip = args.norm_clip

        self.Q = DQN(args, self.action_space).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, args.optimizer)(self.Q.parameters(), **args.optimizer_parameters)

        self.discount = args.discount

        # LAP hyper-parameters
        self.alpha = args.alpha
        self.min_priority = args.min_priority

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if args.polyak_target_update else self.copy_target_update
        self.target_update_frequency = args.target_update_frequency
        self.tau = args.tau

        # Decay for eps
        self.initial_eps = args.initial_eps
        self.end_eps = args.end_eps
        self.slope = (self.end_eps - self.initial_eps) / args.eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + args.state_dim
        self.eval_eps = 0.001
        self.num_actions = 15

        # Number of training iterations
        self.iterations = 0


    def select_action(self, state, eval=False):
        with torch.no_grad():
            q = self.Q(state)
            action = (q * self.support).sum(2).argmax(1).reshape(-1, 1)
            return action, torch.log(q)


    def train(self, replay_buffer):
        state, action, next_state, reward, done, ind, weights = replay_buffer.sample()

        log_p1s = self.Q(state, log=True)
        log_p1s_a = log_p1s.gather(1, action.unsqueeze(1).expand(self.batch_size, 1, self.atoms)).squeeze(1)

        with torch.no_grad():
            next_Q = self.Q(next_state)
            next_action = (next_Q * self.support).sum(2).argmax(1).reshape(-1, 1)
            pns = self.Q_target(next_state)
            pns_a = pns.gather(1, next_action.unsqueeze(1).expand(self.batch_size, 1, self.atoms)).squeeze(1)
            target_Q = (
                reward.expand(-1, self.atoms) + done * (self.discount ** self.n) * 
                pns_a
            )

        current_Q = self.Q(state).gather(1, action.unsqueeze(1).expand(self.batch_size, 1, self.atoms)).squeeze(1)
        
        target_Q = target_Q.clamp(min=self.V_min, max=self.V_max)  # Clamp between supported values
        # Compute L2 projection of Tz onto fixed support z
        b = (target_Q - self.V_min) / self.delta_z 
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        # Distribute probability of Tz
        m = state.new_zeros(self.batch_size, self.atoms)
        offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(action)
        m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_p1s_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.Q.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.Q.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        priority = loss.clamp(min=self.min_priority).pow(self.alpha).cpu().data.numpy().flatten()
        replay_buffer.update_priority(ind, priority)


    def huber(self, x):
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()


    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def copy_target_update(self):
        #if self.iterations % self.target_update_frequency == 0:
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

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            _, action_log_dist = self.Q(state.unsqueeze(0), log = True)
            return (action_log_dist * self.support).sum(2).max(1)[0].item()
