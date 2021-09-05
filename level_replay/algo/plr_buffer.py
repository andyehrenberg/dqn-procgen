import numpy as np
import torch
from dataclasses import dataclass

from level_replay.algo.buffer import Buffer


@dataclass
class LevelBufferConfig:
    batch_size = 32
    memory_capacity = 15000
    device: str
    seeds: list
    ptr = 0
    size = 0
    PER = False


class PLRBuffer:
    def __init__(self, args, env):
        self.device = args.device
        seeds = args.seeds
        self.obs_space = env.observation_space.shape
        self.action_space = env.action_space.n
        self.num_actors = args.num_processes
        self.strategy = "value_l1"
        self.replay_schedule = "proportionate"
        self.score_transform = "rank"
        self.staleness_transform = "power"
        self.temperature = 0.1
        self.eps = 0.05
        self.rho = 1.0
        self.nu = 0.5
        self.alpha = 1.0
        self.staleness_coef = 0.1
        self.staleness_temperature = 1.0

        self.num_seeds_in_update = 64
        self.batch_size_per_seed = 32

        self._init_seed_index(seeds)

        self.unseen_seed_weights = np.array([1.0] * len(self.seeds))
        self.seed_scores = np.array([0.0] * len(self.seeds), dtype=np.float)
        self.partial_seed_scores = np.zeros((self.num_actors, len(self.seeds)), dtype=np.float)
        self.partial_seed_steps = np.zeros((self.num_actors, len(self.seeds)), dtype=np.int64)
        self.seed_staleness = np.array([0.0] * len(self.seeds), dtype=np.float)

        self.next_seed_index = 0

        buffer_config = LevelBufferConfig(self.device, self.seeds)
        buffer_config.batch_size = self.batch_size_per_seed

        self.buffers = {seed: Buffer(buffer_config, env) for seed in self.seeds}
        self.valid_buffers = np.array([0.0] * len(self.seeds), dtype=np.float)

    def add(self, state, action, next_state, reward, done, seed):
        self.buffers[seed.item()].add(state, action, next_state, reward, done, seed)
        if self.buffers[seed.item()].size > self.batch_size_per_seed:
            self.valid_buffers[self.seed2index[seed.item()]] = 1.0

    def _get_weights(self, ind):
        seeds = self.seeds[ind]
        weights = []
        for seed in seeds:
            weight = 1.0 if self.unseen_seed_weights[seed][0] != 0.0 else self.seed_scores[seed][0]
            weights.append(weight)

        weights = torch.FloatTensor(weights).to(self.device).reshape(-1, 1)

        return weights

    def seed_range(self):
        return (int(min(self.seeds)), int(max(self.seeds)))

    def _init_seed_index(self, seeds):
        self.seeds = np.array(seeds, dtype=np.int64)
        self.seed2index = {seed: i for i, seed in enumerate(seeds)}

    def update_with_rollouts(self, rollouts):
        score_function = self._average_value_l1

        self._update_with_rollouts(rollouts, score_function)

    def update_seed_score(self, actor_index, seed_idx, score, num_steps):
        score = self._partial_update_seed_score(actor_index, seed_idx, score, num_steps, done=True)

        # self.unseen_seed_weights[seed_idx] = 0.0  # No longer unseen

        old_score = self.seed_scores[seed_idx]
        self.seed_scores[seed_idx] = (1 - self.alpha) * old_score + self.alpha * score

    def _partial_update_seed_score(self, actor_index, seed_idx, score, num_steps, done=False):
        partial_score = self.partial_seed_scores[actor_index][seed_idx]
        partial_num_steps = self.partial_seed_steps[actor_index][seed_idx]

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score) * num_steps / float(running_num_steps)

        if done:
            self.partial_seed_scores[actor_index][seed_idx] = 0.0  # zero partial score, partial num_steps
            self.partial_seed_steps[actor_index][seed_idx] = 0
        else:
            self.partial_seed_scores[actor_index][seed_idx] = merged_score
            self.partial_seed_steps[actor_index][seed_idx] = running_num_steps

        return merged_score

    def _average_value_l1(self, **kwargs):
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]

        advantages = returns - value_preds

        return advantages.abs().mean().item()

    def _update_with_rollouts(self, rollouts, score_function):
        level_seeds = rollouts.level_seeds
        done = ~(rollouts.masks > 0)
        total_steps, num_actors = rollouts.rewards.shape[:2]

        for actor_index in range(num_actors):
            done_steps = torch.nonzero(done[:, actor_index], as_tuple=False)[:total_steps, 0]
            start_t = 0

            for t in done_steps:
                if not start_t < total_steps:
                    break

                if t == 0:  # if t is 0, then this done step caused a full update of previous seed last cycle
                    continue

                seed_t = level_seeds[start_t, actor_index].item()
                seed_idx_t = self.seed2index[seed_t]

                score_function_kwargs = {}

                score_function_kwargs["returns"] = rollouts.returns[start_t:t, actor_index]
                score_function_kwargs["rewards"] = rollouts.rewards[start_t:t, actor_index]
                score_function_kwargs["value_preds"] = rollouts.value_preds[start_t:t, actor_index]

                score1 = score_function(**score_function_kwargs)
                num_steps = len(rollouts.rewards[start_t:t, actor_index])
                self.update_seed_score(actor_index, seed_idx_t, score1, num_steps)

                start_t = t.item()

            if start_t < total_steps:
                seed_t = level_seeds[start_t, actor_index].item()
                seed_idx_t = self.seed2index[seed_t]

                score_function_kwargs = {}

                score_function_kwargs["returns"] = rollouts.returns[start_t:, actor_index]
                score_function_kwargs["rewards"] = rollouts.rewards[start_t:, actor_index]
                score_function_kwargs["value_preds"] = rollouts.value_preds[start_t:, actor_index]

                score2 = score_function(**score_function_kwargs)
                num_steps = len(rollouts.rewards[start_t:, actor_index])
                self._partial_update_seed_score(actor_index, seed_idx_t, score2, num_steps)

    def after_update(self):
        # Reset partial updates, since weights have changed, and thus logits are now stale
        for actor_index in range(self.partial_seed_scores.shape[0]):
            for seed_idx in range(self.partial_seed_scores.shape[1]):
                if self.partial_seed_scores[actor_index][seed_idx] != 0:
                    self.update_seed_score(actor_index, seed_idx, 0, 0)
        self.partial_seed_scores.fill(0)
        self.partial_seed_steps.fill(0)

    def _update_staleness(self, selected_idx):
        self.unseen_seed_weights[selected_idx] = 0.0  # has been updated on

        if self.staleness_coef > 0:
            self.seed_staleness = self.seed_staleness + 1
            self.seed_staleness[selected_idx] = 0

    def _sample_replay_level(self):
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=np.float) / len(sample_weights)

        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        return int(seed)

    def _sample_unseen_level(self):
        w = self.unseen_seed_weights * self.valid_buffers
        sample_weights = w / w.sum()

        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        return int(seed)

    def sample(self):
        sub_batch = int(self.batch_size_per_seed)
        batch_size = self.num_seeds_in_update * self.batch_size_per_seed
        state = torch.empty((batch_size,) + self.obs_space, dtype=torch.float, device=self.device)
        action = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        next_state = torch.empty((batch_size,) + self.obs_space, dtype=torch.float, device=self.device)
        reward = torch.empty((batch_size, 1), dtype=torch.float, device=self.device)
        not_done = torch.empty((batch_size, 1), dtype=torch.float, device=self.device)
        seeds = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        for i in range(self.num_seeds_in_update):
            seed = self._sample_seed()
            state_, action_, next_state_, reward_, not_done_, seeds_, _, _ = self.buffers[seed].sample()
            state[i * sub_batch : (i + 1) * sub_batch] = state_
            action[i * sub_batch : (i + 1) * sub_batch] = action_
            next_state[i * sub_batch : (i + 1) * sub_batch] = next_state_
            reward[i * sub_batch : (i + 1) * sub_batch] = reward_
            not_done[i * sub_batch : (i + 1) * sub_batch] = not_done_
            seeds[i * sub_batch : (i + 1) * sub_batch] = seeds_

        return state, action, next_state, reward, not_done, seeds, 0, 1

    def _sample_seed(self):
        strategy = self.strategy

        if strategy == "random":
            seed_idx = np.random.choice(range((len(self.seeds))))
            seed = self.seeds[seed_idx]
            return int(seed)

        if strategy == "sequential":
            seed_idx = self.next_seed_index
            self.next_seed_index = (self.next_seed_index + 1) % len(self.seeds)
            seed = self.seeds[seed_idx]
            return int(seed)

        num_unseen = (self.unseen_seed_weights > 0).sum()
        proportion_seen = (len(self.seeds) - num_unseen) / len(self.seeds)
        w = self.unseen_seed_weights * self.valid_buffers

        if self.replay_schedule == "fixed":
            if w.sum() == 0:
                return self._sample_replay_level()
            if proportion_seen >= self.rho:
                # Sample replay level with fixed prob = 1 - nu OR if all levels seen
                if np.random.rand() > self.nu or not proportion_seen < 1.0:
                    return self._sample_replay_level()

            # Otherwise, sample a new level
            return self._sample_unseen_level()

        else:  # Default to proportionate schedule
            if w.sum() == 0:
                return self._sample_replay_level()
            if (proportion_seen >= self.rho and np.random.rand() < proportion_seen) or w.sum() == 0:
                return self._sample_replay_level()
            else:
                return self._sample_unseen_level()

    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.seed_scores)
        weights = weights * (1 - self.unseen_seed_weights)  # zero out unseen levels

        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(
                self.staleness_transform, self.staleness_temperature, self.seed_staleness
            )
            staleness_weights = staleness_weights * (1 - self.unseen_seed_weights)
            z = np.sum(staleness_weights)
            if z > 0:
                staleness_weights /= z

            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * staleness_weights

        return weights

    def _score_transform(self, transform, temperature, scores):
        if transform == "constant":
            weights = np.ones_like(scores)
        if transform == "max":
            weights = np.zeros_like(scores)
            scores = scores[:]
            scores[self.unseen_seed_weights > 0] = -float("inf")  # only argmax over seen levels
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
            weights[argmax] = 1.0
        elif transform == "eps_greedy":
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1.0 - self.eps
            weights += self.eps / len(self.seeds)
        elif transform == "rank":
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / ranks ** (1.0 / temperature)
        elif transform == "power":
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1.0 / temperature)
        elif transform == "softmax":
            weights = np.exp(np.array(scores) / temperature)

        return weights
