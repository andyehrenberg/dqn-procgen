import numpy as np
import torch
import math
import random

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


class SegmentTree:
    def __init__(self, size, device, args):
        self.device = device
        self.index = 0
        self.max_size = int(size)
        self.full = False  # Used to track actual capacity
        self.tree_start = (
            2 ** (self.max_size - 1).bit_length() - 1
        )  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.max_size,), dtype=np.float32)
        self.state = np.zeros((self.max_size, *args.state_dim), dtype=np.uint8)
        self.action = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.seeds = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, state, action, next_state, reward, done, seeds, value):
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

        self.state[self.index] = state
        self.action[self.index] = action
        self.next_state[self.index] = next_state
        self.reward[self.index] = reward
        self.not_done[self.index] = not_done
        self.seeds[self.index] = seeds

        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.max_size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)  # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(
            np.int32
        )  # Classify which values are in left or right branches
        successor_indices = children_indices[
            successor_choices, np.arange(indices.size)
        ]  # Use classification to index into the indices matrix
        successor_values = (
            values - successor_choices * left_children_values
        )  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_index):
        return (
            torch.FloatTensor(self.state[data_index % self.max_size]).to(self.device) / 255.0,
            torch.LongTensor(self.action[data_index % self.max_size]).to(self.device),
            torch.FloatTensor(self.next_state[data_index % self.max_size]).to(self.device) / 255.0,
            torch.FloatTensor(self.reward[data_index % self.max_size]).to(self.device),
            torch.FloatTensor(self.not_done[data_index % self.max_size]).to(self.device),
            torch.LongTensor(self.seeds[data_index % self.max_size]).to(self.device),
            data_index % self.max_size,
        )

    def total(self):
        return self.sum_tree[0]


class ReplayMemory:
    def __init__(self, args):
        self.device = args.device
        self.capacity = args.memory_capacity
        self.discount = args.discount
        self.n = args.multi_step
        self.batch_size = args.batch_size
        self.priority_weight = (
            args.beta
        )  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.alpha
        self.t = 0  # Internal episode timestep counter
        self.transitions = SegmentTree(
            self.capacity, self.device, args
        )  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and terminal at time t + 1
    def add(self, state, action, next_state, reward, done, seeds):
        self.transitions.append(
            state, action, next_state, reward, done, seeds, self.transitions.max
        )  # Store new transition with maximum priority
        self.t = 0 if done else self.t + 1  # Start new episodes with t = 0

    # Returns the transitions with blank states where appropriate
    def _get_transitions(self, idxs):
        transition_idxs = idxs
        transitions = self.transitions.get(transition_idxs)
        return transitions

    # Returns a valid sample from each segment
    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = (
            p_total / batch_size
        )  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            samples = (
                np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
            )  # Uniformly sample from within all segments
            probs, idxs, tree_idxs = self.transitions.find(
                samples
            )  # Retrieve samples from tree with un-normalised probability
            if (
                np.all((self.transitions.index - idxs) % self.capacity > self.n)
                and np.all((idxs - self.transitions.index) % self.capacity >= 1)
                and np.all(probs != 0)
            ):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
        state, action, next_state, reward, not_done, seeds, ind = self._get_transitions(idxs)
        return probs, idxs, tree_idxs, state, action, next_state, reward, not_done, seeds, ind

    def sample(self):
        p_total = (
            self.transitions.total()
        )  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        (
            probs,
            idxs,
            tree_idxs,
            state,
            action,
            next_state,
            reward,
            not_done,
            seeds,
            ind,
        ) = self._get_samples_from_segments(
            self.batch_size, p_total
        )  # Get batch of valid samples
        probs = probs / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(
            weights / weights.max(), dtype=torch.float32, device=self.device
        )  # Normalise by max importance-sampling weight from batch
        return state, action, next_state, reward, not_done, seeds, ind, weights

    def update_priority(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)


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


def make_buffer(args, num_updates, atari=False, seg_tree_buffer=False):
    if args.seg_tree_buffer:
        return ReplayMemory(args)
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
