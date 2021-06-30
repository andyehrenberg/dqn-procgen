import numpy as np
import torch
import math
import random
import operator

from level_replay.algo.binary_heap import BinaryHeap


class AbstractBuffer:
    def __init__(self, state_dim, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, *state_dim), dtype=np.uint8)
        self.action = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.seeds = np.zeros((self.max_size, 1), dtype=np.uint8)

    def add(self, state, action, next_state, reward, done, seeds):
        pass

    def sample(self):
        pass

    def update_priority(self, ind, priority):
        pass


class Buffer(AbstractBuffer):
    def __init__(self, state_dim, batch_size, buffer_size, device, prioritized, num_updates, args):
        super(Buffer, self).__init__(state_dim, batch_size, buffer_size, device)
        self.prioritized = prioritized

        if self.prioritized:
            self.tree = SumTree(self.max_size)
            self.max_priority = 1.0
            self.beta = args.beta
            self.beta_stepper = (1 - self.beta) / float(num_updates)
            self.alpha = args.alpha

    def add(self, state, action, next_state, reward, done, seeds):
        n_transitions = state.shape[0] if len(state.shape) == 4 else 1
        end = (self.ptr + n_transitions) % self.max_size
        if "cuda" in self.device.type:
            state = (state * 255).cpu().numpy().astype(np.uint8)
            action = action.cpu().numpy().astype(np.uint8)
            next_state = (next_state * 255).cpu().numpy().astype(np.uint8)
            # We leave reward as numpy throughout
            # reward = reward.cpu().numpy()
            seeds = seeds.cpu().numpy().astype(np.uint8)
        else:
            state = (state * 255).numpy().astype(np.uint8)
            action = action.numpy().astype(np.uint8)
            next_state = (next_state * 255).numpy().astype(np.uint8)
            seeds = seeds.numpy().astype(np.uint8)

        not_done = (1 - done).reshape(-1, 1)

        if self.ptr + n_transitions > self.max_size:
            self.state[self.ptr :] = state[: n_transitions - end]
            self.state[:end] = state[n_transitions - end :]

            self.action[self.ptr :] = action[: n_transitions - end]
            self.action[:end] = action[n_transitions - end :]

            self.next_state[self.ptr :] = next_state[: n_transitions - end]
            self.next_state[:end] = next_state[n_transitions - end :]

            self.reward[self.ptr :] = reward[: n_transitions - end]
            self.reward[:end] = reward[n_transitions - end :]

            self.not_done[self.ptr :] = not_done[: n_transitions - end]
            self.not_done[:end] = not_done[n_transitions - end :]
            self.seeds[self.ptr :] = seeds[: n_transitions - end]
            self.seeds[:end] = seeds[n_transitions - end :]
        else:
            self.state[self.ptr : self.ptr + n_transitions] = state
            self.action[self.ptr : self.ptr + n_transitions] = action
            self.next_state[self.ptr : self.ptr + n_transitions] = next_state
            self.reward[self.ptr : self.ptr + n_transitions] = reward
            self.not_done[self.ptr : self.ptr + n_transitions] = not_done
            self.seeds[self.ptr : self.ptr + n_transitions] = seeds

        if self.prioritized:
            for index in [i % self.max_size for i in range(self.ptr, self.ptr + n_transitions)]:
                self.tree.set(index, self.max_priority)

        self.ptr = end
        self.size = min(self.size + n_transitions, self.max_size)

    def sample(self):
        ind = (
            self.tree.sample(self.batch_size)
            if self.prioritized
            else np.random.randint(0, self.size, size=self.batch_size)
        )

        if self.prioritized:
            weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
            weights = torch.FloatTensor(weights / weights.max()).to(self.device).reshape(-1, 1)
            self.beta = min(self.beta + self.beta_stepper, 1)
        else:
            weights = torch.FloatTensor([1]).to(self.device)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device) / 255.0,
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device),
            ind,
            weights,
        )

    def update_priority(self, ind, priority):
        priority = np.power(priority, self.alpha)
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)


class RankBuffer(AbstractBuffer):
    def __init__(self, state_dim, batch_size, buffer_size, device, prioritized, num_updates, args):
        super(RankBuffer, self).__init__(state_dim, batch_size, buffer_size, device)

        self.beta = args.beta
        self.beta_stepper = (1 - self.beta) / float(num_updates)
        self.priority_queue = BinaryHeap(self.max_size)
        self.max_priority = 1.0
        self.alpha = args.alpha

        self.build_distribution()

    def build_distribution(self):
        pdf = list(map(lambda x: math.pow(x, -self.max_priority), range(1, self.max_size + 1)))
        pdf_sum = math.fsum(pdf)
        self.power_law_distribution = list(map(lambda x: x / pdf_sum, pdf))

    def add(self, state, action, next_state, reward, done, seeds):
        n_transitions = state.shape[0] if len(state.shape) == 4 else 1
        end = (self.ptr + n_transitions) % self.max_size
        if "cuda" in self.device.type:
            state = (state * 255).cpu().numpy().astype(np.uint8)
            action = action.cpu().numpy().astype(np.uint8)
            next_state = (next_state * 255).cpu().numpy().astype(np.uint8)
            # reward = reward.cpu().numpy()
            seeds = seeds.cpu().numpy().astype(np.uint8)
        else:
            state = (state * 255).numpy().astype(np.uint8)
            action = action.numpy().astype(np.uint8)
            next_state = (next_state * 255).numpy().astype(np.uint8)
            seeds = seeds.numpy().astype(np.uint8)

        not_done = (1 - done).reshape(-1, 1)

        if self.ptr + n_transitions > self.max_size:
            self.state[self.ptr :] = state[: n_transitions - end]
            self.state[:end] = state[n_transitions - end :]

            self.action[self.ptr :] = action[: n_transitions - end]
            self.action[:end] = action[n_transitions - end :]

            self.next_state[self.ptr :] = next_state[: n_transitions - end]
            self.next_state[:end] = next_state[n_transitions - end :]

            self.reward[self.ptr :] = reward[: n_transitions - end]
            self.reward[:end] = reward[n_transitions - end :]

            self.not_done[self.ptr :] = not_done[: n_transitions - end]
            self.not_done[:end] = not_done[n_transitions - end :]
            self.seeds[self.ptr :] = seeds[: n_transitions - end]
            self.seeds[:end] = seeds[n_transitions - end :]
        else:
            self.state[self.ptr : self.ptr + n_transitions] = state
            self.action[self.ptr : self.ptr + n_transitions] = action
            self.next_state[self.ptr : self.ptr + n_transitions] = next_state
            self.reward[self.ptr : self.ptr + n_transitions] = reward
            self.not_done[self.ptr : self.ptr + n_transitions] = not_done
            self.seeds[self.ptr : self.ptr + n_transitions] = seeds

        priority = self.priority_queue.get_max_priority()
        for index in [i % self.max_size for i in range(self.ptr, self.ptr + n_transitions)]:
            self.priority_queue.update(priority, index)

        self.ptr = end
        self.size = min(self.size + n_transitions, self.max_size)

    def sample(self):
        ind, weights = self.select(self.batch_size)
        weights = torch.FloatTensor(weights).to(self.device).reshape(-1, 1)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device) / 255.0,
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device),
            ind,
            weights,
        )

    def rebalance(self):
        self.priority_queue.balance_tree()

    def update_priority(self, indices, delta):
        delta = np.power(delta, self.alpha)
        for i in range(len(indices)):
            self.priority_queue.update(math.fabs(delta[i]), indices[i])

    def select(self, batch_size):
        distribution = self.power_law_distribution
        rank_list = []
        for _ in range(batch_size):
            index = random.randint(1, self.priority_queue.size)
            rank_list.append(index)

        alpha_pow = [distribution[v - 1] for v in rank_list]
        w = np.power(np.array(alpha_pow) * self.size, -self.beta)
        w_max = max(w)
        w = np.divide(w, w_max)
        rank_e_id = 0
        self.beta = min(self.beta + self.beta_stepper, 1)
        rank_e_id = self.priority_queue.priority_to_experience(rank_list)
        return rank_e_id, w


class ReplayBuffer(object):
    def __init__(self, args):
        self.max_size = int(args.memory_capacity)
        self.state = np.zeros((self.max_size, *args.state_dim), dtype=np.uint8)
        self.action = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.seeds = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.batch_size = args.batch_size
        self.size = 0
        self.ptr = 0
        self.device = args.device

    def __len__(self):
        return self.size

    def _encode_transition(self, state, action, next_state, done, seed):
        state = (state * 255).cpu().numpy().astype(np.uint8)
        action = action.cpu().numpy().astype(np.uint8)
        next_state = (next_state * 255).cpu().numpy().astype(np.uint8)
        seed = seed.cpu().numpy().astype(np.uint8)
        not_done = (1 - done).reshape(-1, 1)

        return state, action, next_state, seed, not_done

    def add(self, state, action, next_state, reward, done, seed):
        state, action, next_state, seed, not_done = self._encode_transition(
            state, action, next_state, done, seed
        )

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = not_done
        self.seeds[self.ptr] = seed

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _encode_sample(self, idxes):
        state = torch.FloatTensor(self.state[idxes]).to(self.device) / 255.0
        action = torch.LongTensor(self.action[idxes]).to(self.device)
        next_state = torch.FloatTensor(self.next_state[idxes]).to(self.device) / 255.0
        reward = torch.FloatTensor(self.reward[idxes]).to(self.device)
        not_done = torch.FloatTensor(self.not_done[idxes]).to(self.device)
        seed = torch.LongTensor(self.seeds[idxes]).to(self.device)

        return state, action, next_state, reward, not_done, seed, idxes

    def sample(self):
        idxes = [random.randint(0, self.size - 1) for _ in range(self.batch_size)]
        return tuple(self._encode_sample(idxes) + [1])


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, args):
        super(PrioritizedReplayBuffer, self).__init__(args)
        num_updates = (args.T_max // args.num_processes - args.start_timesteps) // args.train_freq
        self._alpha = args.alpha
        self._beta = args.beta
        self.beta_stepper = (1 - self._beta) / float(num_updates)

        it_capacity = 1
        while it_capacity < self.max_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self.ptr
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self):
        idxes = self._sample_proportional(self.batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.size) ** (-self._beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.size) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = torch.FloatTensor(np.array(weights)).to(self.device).reshape(-1, 1)
        encoded_sample = self._encode_sample(idxes)

        self._beta = min(self._beta + self.beta_stepper, 1)

        return tuple(list(encoded_sample) + [weights])

    def update_priority(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.size
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class AutoEREBuffer:
    def __init__(self, state_dim, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.device = device

        self.ptr, self.size, self.max_size = 0, 0, int(buffer_size)
        self.eta_init = 0.995

        self.state = np.zeros((self.max_size, *state_dim), dtype=np.uint8)
        self.action = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.seeds = np.zeros((self.max_size, 1), dtype=np.uint8)

        self.epoch_performance_buf = np.zeros(0, dtype=np.float32)
        self.current_epoch = 0
        self.max_history_improvement = 1e-5

    def add(self, state, action, next_state, reward, done, seeds):
        n_transitions = state.shape[0] if len(state.shape) == 4 else 1
        end = (self.ptr + n_transitions) % self.max_size
        if "cuda" in self.device.type:
            state = (state * 255).cpu().numpy().astype(np.uint8)
            action = action.cpu().numpy().astype(np.uint8)
            next_state = (next_state * 255).cpu().numpy().astype(np.uint8)
            seeds = seeds.cpu().numpy().astype(np.uint8)
        else:
            state = (state * 255).numpy().astype(np.uint8)
            action = action.numpy().astype(np.uint8)
            next_state = (next_state * 255).numpy().astype(np.uint8)
            seeds = seeds.numpy().astype(np.uint8)

        not_done = (1 - done).reshape(-1, 1)

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = not_done
        self.seeds[self.ptr] = seeds

        self.ptr = end
        self.size = min(self.size + n_transitions, self.max_size)

    def store_epoch_performance(self, test_score):
        self.epoch_performance_buf = np.append(self.epoch_performance_buf, np.array([test_score]))
        self.current_epoch += 1

    def get_auto_eta(self, eta_init=0.995, baseline_epoch=100, ave_size=20):
        if self.current_epoch < baseline_epoch:  # if don't have enough data
            return self.eta_init
        # if have enough data already
        baseline_improvement = (
            self.epoch_performance_buf[baseline_epoch - ave_size : baseline_epoch].mean()
            - self.epoch_performance_buf[0:ave_size].mean()
        )
        current_performance = self.epoch_performance_buf[
            self.current_epoch - ave_size : self.current_epoch
        ].mean()
        previous_performance = self.epoch_performance_buf[
            self.current_epoch - 100 : self.current_epoch - 100 + ave_size
        ].mean()
        recent_improvement = current_performance - previous_performance
        interpolation = recent_improvement / baseline_improvement
        if interpolation < 0:
            interpolation = 0
        auto_eta = self.eta_init * interpolation + 1 * (1 - interpolation)
        return auto_eta

    def get_auto_eta_with_recent_baseline(self, eta_init=0.994, compare_interval=200):
        half_compare_interval = int(compare_interval / 2)
        onethird_compare_interval = int(compare_interval / 3)
        if (
            self.current_epoch < half_compare_interval + onethird_compare_interval
        ):  # if don't have enough data
            return self.eta_init

        baseline_start = self.current_epoch - compare_interval  #
        if baseline_start < 0:
            baseline_start = 0

        current_performance = self.epoch_performance_buf[
            self.current_epoch - onethird_compare_interval : self.current_epoch
        ].mean()
        previous_performance_istart = (
            self.current_epoch - half_compare_interval - int(onethird_compare_interval / 2)
        )
        previous_performance_iend = (
            self.current_epoch - half_compare_interval + int(onethird_compare_interval / 2)
        )
        previous_performance = self.epoch_performance_buf[
            previous_performance_istart:previous_performance_iend
        ].mean()
        baseline_performance = self.epoch_performance_buf[
            baseline_start : baseline_start + onethird_compare_interval
        ].mean()

        recent_improvement = current_performance - previous_performance
        older_improvement = previous_performance - baseline_performance

        if older_improvement == 0:
            interpolation = 1
        else:
            interpolation = recent_improvement / older_improvement

        if interpolation < 0:
            interpolation = 0
        if interpolation > 1:
            interpolation = 1
        auto_eta = self.eta_init * interpolation + 1 * (1 - interpolation)
        return auto_eta

    def get_auto_eta_max_history(self, baseline_epoch=100, ave_size=20):
        if self.current_epoch < baseline_epoch:  # if don't have enough data
            return self.eta_init
        # if have enough data already
        current_performance = self.epoch_performance_buf[
            self.current_epoch - ave_size : self.current_epoch
        ].mean()
        previous_performance = self.epoch_performance_buf[
            self.current_epoch - 100 : self.current_epoch - 100 + ave_size
        ].mean()
        recent_improvement = current_performance - previous_performance
        if recent_improvement > self.max_history_improvement:
            self.max_history_improvement = recent_improvement

        interpolation = recent_improvement / self.max_history_improvement
        # clip to range (0, 1)
        interpolation = np.clip(interpolation, a_min=0, a_max=1)
        auto_eta = self.eta_init * interpolation + 0.999 * (1 - interpolation)
        return auto_eta

    def sample_uniform_batch(self, batch_size=32):
        idxes = np.random.randint(0, self.size, size=self.batch_size)

        state = torch.FloatTensor(self.state[idxes]).to(self.device) / 255.0
        action = torch.LongTensor(self.action[idxes]).to(self.device)
        next_state = torch.FloatTensor(self.next_state[idxes]).to(self.device) / 255.0
        reward = torch.FloatTensor(self.reward[idxes]).to(self.device)
        not_done = torch.FloatTensor(self.not_done[idxes]).to(self.device)
        seed = torch.LongTensor(self.seeds[idxes]).to(self.device)

        return state, action, next_state, reward, not_done, seed, idxes, 1.0

    def sample_priority_only_batch(self, c_k):
        recent_data_size = self.batch_size
        max_index = min(int(c_k), self.size)
        recent_relative_idxs = -np.random.randint(0, max_index, size=recent_data_size)
        recent_idxs = (self.ptr - 1 + recent_relative_idxs) % self.size

        state = torch.FloatTensor(self.state[recent_idxs]).to(self.device) / 255.0
        action = torch.LongTensor(self.action[recent_idxs]).to(self.device)
        next_state = torch.FloatTensor(self.next_state[recent_idxs]).to(self.device) / 255.0
        reward = torch.FloatTensor(self.reward[recent_idxs]).to(self.device)
        not_done = torch.FloatTensor(self.not_done[recent_idxs]).to(self.device)
        seed = torch.LongTensor(self.seeds[recent_idxs]).to(self.device)

        return state, action, next_state, reward, not_done, seed, recent_idxs, 1.0


class AtariBuffer(AbstractBuffer):
    def __init__(self, state_dim, batch_size, buffer_size, device, prioritized, num_updates, args):
        super(AtariBuffer, self).__init__(state_dim, batch_size, buffer_size, device)
        self.prioritized = prioritized

        if self.prioritized:
            self.tree = SumTree(self.max_size)
            self.max_priority = 1.0
            self.beta = 0.4
            self.beta_stepper = (1 - self.beta) / float(num_updates)

    def add(self, state, action, next_state, reward, done, seeds):
        end = (self.ptr + 1) % self.max_size
        if "cuda" in self.device.type:
            state = (state * 255).cpu().numpy().astype(np.uint8)
            action = action.cpu().numpy().astype(np.uint8)
            next_state = (next_state * 255).cpu().numpy().astype(np.uint8)
            # reward = reward.cpu().numpy()
            # seeds = seeds.cpu().numpy().astype(np.uint8)
        else:
            state = (state * 255).numpy().astype(np.uint8)
            action = action.numpy().astype(np.uint8)
            next_state = (next_state * 255).numpy().astype(np.uint8)
            # seeds = seeds.numpy().astype(np.uint8)

        not_done = (1 - done).reshape(-1, 1)

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = not_done
        self.seeds[self.ptr] = seeds

        self.ptr = end
        self.size = min(self.size + 1, self.max_size)

        if self.prioritized:
            self.tree.set(self.ptr, self.max_priority)

    def sample(self):
        ind = (
            self.tree.sample(self.batch_size)
            if self.prioritized
            else np.random.randint(0, self.size, size=self.batch_size)
        )

        if self.prioritized:
            weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
            weights = torch.FloatTensor(weights / weights.max()).to(self.device).reshape(-1, 1)
            self.beta = min(self.beta + self.beta_stepper, 1)
        else:
            weights = torch.FloatTensor([1]).to(self.device)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device) / 255.0,
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device),
            ind,
            weights,
        )

    def update_priority(self, ind, priority):
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity=capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity=capacity, operation=min, neutral_element=float("inf"))

    def min(self, start=0, end=None):
        return super(MinSegmentTree, self).reduce(start, end)


# class PLRBuffer:
# def __init__(self, state_dim, batch_size, buffer_size, device, prioritized, seeds, num_updates):
# self.buffers = {
#    i: Buffer(state_dim, batch_size, buffer_size, device, prioritized, num_updates) for i in seeds
# }

# def add(self, state, action, next_state, reward, done, seeds):
# n_transitions = state.shape[0]
# end = (self.ptr + n_transitions) % self.max_size
# for i in seeds.unique():
# s = state[torch.where(seeds == i)[0]]
# a = action[torch.where(seeds == i)[0]]
# n_s = next_state[torch.where(seeds == i)[0]]
# r = reward[torch.where(seeds == i)[0]]
# d = done[torch.where(seeds == i)[0]]
# seed = seeds[torch.where(seeds == i)[0]]
# self.buffers[i].add(s, a, n_s, r, d, seed)

# def sample(self, seed):
# self.buffers[seed].sample()


class SumTree(object):
    def __init__(self, max_size):
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index
    def sample(self, batch_size):
        query_value = np.random.uniform(0, self.nodes[0][0], size=batch_size)
        node_index = np.zeros(batch_size, dtype=int)

        for nodes in self.nodes[1:]:
            node_index *= 2
            left_sum = nodes[node_index]

            is_greater = np.greater(query_value, left_sum)
            # If query_value > left_sum -> go right (+1), else go left (+0)
            node_index += is_greater
            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            query_value -= left_sum * is_greater

        return node_index

    def set(self, node_index, new_priority):
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    def batch_set(self, node_index, new_priority):
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2


def make_buffer(args, num_updates, atari=False):
    if args.seg_tree_buffer:
        return PrioritizedReplayBuffer(args)
    if args.rank_based_PER:
        return RankBuffer(
            args.state_dim, args.batch_size, args.memory_capacity, args.device, args.PER, num_updates, args
        )
    if not atari:
        return Buffer(
            args.state_dim, args.batch_size, args.memory_capacity, args.device, args.PER, num_updates, args
        )
    return AtariBuffer(
        args.state_dim, args.batch_size, args.memory_capacity, args.device, args.PER, num_updates, args
    )
