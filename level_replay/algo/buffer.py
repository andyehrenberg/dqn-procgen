import numpy as np
import torch

class AbstractBuffer():
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
    def __init__(self, state_dim, batch_size, buffer_size, device, prioritized, num_updates):
        super(Buffer, self).__init__(state_dim, batch_size, buffer_size, device)
        self.prioritized = prioritized

        if self.prioritized:
            self.tree = SumTree(self.max_size)
            self.max_priority = 1.0
            self.beta = 0.4
            self.beta_stepper = (1 - 0.4)/float(num_updates)

    def add(self, state, action, next_state, reward, done, seeds):
        n_transitions = state.shape[0] if len(state.shape) == 4 else 1
        end = (self.ptr + n_transitions) % self.max_size
        if 'cuda' in self.device.type:
            state = (state*255).cpu().numpy().astype(np.uint8)
            action = action.cpu().numpy().astype(np.uint8)
            next_state = (next_state*255).cpu().numpy().astype(np.uint8)
            reward = reward.cpu().numpy()
            seeds = seeds.cpu().numpy().astype(np.uint8)
        else:
            state = (state*255).numpy().astype(np.uint8)
            action = action.numpy().astype(np.uint8)
            next_state = (next_state*255).numpy().astype(np.uint8)
            seeds = seeds.numpy().astype(np.uint8)

        not_done = (1 - done).reshape(-1, 1)

        if self.ptr + n_transitions > self.max_size:
            self.state[self.ptr:] = state[:n_transitions - end]
            self.state[:end] = state[n_transitions - end:]

            self.action[self.ptr:] = action[:n_transitions - end]
            self.action[:end] = action[n_transitions - end:]

            self.next_state[self.ptr:] = next_state[:n_transitions - end]
            self.next_state[:end] = next_state[n_transitions - end:]

            self.reward[self.ptr:] = reward[:n_transitions - end]
            self.reward[:end] = reward[n_transitions - end:]

            self.not_done[self.ptr:] = not_done[:n_transitions - end]
            self.not_done[:end] = not_done[n_transitions - end:]
            self.seeds[self.ptr:] = seeds[:n_transitions - end]
            self.seeds[:end] = seeds[n_transitions - end:]
        else:
            self.state[self.ptr:self.ptr+n_transitions] = state
            self.action[self.ptr:self.ptr+n_transitions] = action
            self.next_state[self.ptr:self.ptr+n_transitions] = next_state
            self.reward[self.ptr:self.ptr+n_transitions] = reward
            self.not_done[self.ptr:self.ptr+n_transitions] = not_done
            self.seeds[self.ptr:self.ptr+n_transitions] = seeds

        if self.prioritized:
            self.tree.set(self.ptr, self.max_priority)

        self.ptr = end
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        ind = self.tree.sample(self.batch_size) if self.prioritized \
            else np.random.randint(0, self.size, size=self.batch_size)

        batch = (
            torch.FloatTensor(self.state[ind]).to(self.device)/255.,
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device)/255.,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device)
        )

        if self.prioritized:
            weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
            weights /= weights.max()
            self.beta = min(self.beta + self.beta_stepper, 1)
            batch += (ind, torch.FloatTensor(weights).to(self.device).reshape(-1, 1))
        else:
            batch += (ind, torch.FloatTensor([1]).to(self.device))

        return batch

    def update_priority(self, ind, priority):
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)

class PLRBuffer:
    def __init__(self, state_dim, batch_size, buffer_size, device, prioritized, seeds, num_updates):
        self.buffers = {i: Buffer(state_dim, batch_size, buffer_size, device, prioritized, num_updates) for i in seeds}

    def add(self, state, action, next_state, reward, done, seeds):
        n_transitions = state.shape[0]
        #end = (self.ptr + n_transitions) % self.max_size
        for i in seeds.unique():
            s = state[torch.where(seeds == i)[0]]
            a = action[torch.where(seeds == i)[0]]
            n_s = next_state[torch.where(seeds == i)[0]]
            r = reward[torch.where(seeds == i)[0]]
            d = done[torch.where(seeds == i)[0]]
            seed = seeds[torch.where(seeds == i)[0]]
            self.buffers[i].add(s, a, n_s, r, d, seed)

    def sample(self, seed):
        self.buffers[seed].sample()

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

def make_buffer(args, num_updates):
    replay_buffer = Buffer(
        args.state_dim,
        args.batch_size,
        args.memory_capacity,
        args.device,
        args.PER,
        num_updates
    )
    return replay_buffer
