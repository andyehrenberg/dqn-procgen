import os
from collections import deque
from typing import List

import numpy as np
import torch
import wandb

from level_replay import utils
from level_replay.algo.buffer import make_buffer
from level_replay.algo.policy import DQNAgent
from level_replay.atari_args import parser

os.environ["OMP_NUM_THREADS"] = "1"

last_checkpoint_time = None


def train(args, seeds):
    global last_checkpoint_time
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    if "cuda" in args.device.type:
        print("Using CUDA\n")
    args.optimizer_parameters = {"lr": args.learning_rate, "eps": args.adam_eps}
    args.seeds = seeds

    args.sge_job_id = int(os.environ.get("JOB_ID", -1))
    args.sge_task_id = int(os.environ.get("SGE_TASK_ID", -1))

    torch.set_num_threads(1)
    utils.seed(args.seed)

    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project=args.wandb_project,
        entity="andyehrenberg",
        config=vars(args),
        tags=["ddqn", "procgen"] + (args.wandb_tags.split(",") if args.wandb_tags else []),
        group=args.wandb_group,
    )
    wandb.run.name = (
        f"dqn-{args.env_name}"
        + f"{'-PER' if args.PER else ''}"
        + f"{'-dueling' if args.dueling else ''}"
        + f"{'-CQL' if args.cql else ''}"
        + f"{'-qrdqn' if args.qrdqn else ''}"
        + f"{'-c51' if args.c51 else ''}"
        + f"{'-noisylayers' if args.noisy_layers else ''}"
    )
    wandb.run.save()

    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3,
    }

    envs = AtariVecEnv("BreakoutNoFrameskip-v0", seeds, args.num_processes, args.device, atari_preprocessing)

    agent = DQNAgent(args, envs)

    state = envs.reset()

    replay_buffer = make_buffer(args, envs, atari=True)

    episode_reward = 0

    state_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]
    reward_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]
    action_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]

    num_steps = int(args.T_max // args.num_processes)

    epsilon_start = args.initial_eps
    epsilon_final = args.end_eps
    epsilon_decay = args.eps_decay_period

    def epsilon(t):
        return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
            -1.0 * (t - args.start_timesteps) / epsilon_decay
        )

    episode_reward = 0
    episode_num = 0

    for t in range(num_steps):
        if t < args.start_timesteps:
            action = (
                torch.LongTensor([envs.action_space.sample() for _ in range(args.num_processes)])
                .reshape(-1, 1)
                .to(args.device)
            )
            value = agent.get_value(state)
        else:
            cur_epsilon = epsilon(t)
            action, value = agent.select_action(state)
            for i in range(args.num_processes):
                if np.random.uniform() < cur_epsilon:
                    action[i] = torch.LongTensor([envs.action_space.sample()]).to(args.device)

        # Perform action and log results
        next_state, reward, done, infos, levels = envs.step(action)

        for i, info in enumerate(infos):
            state_deque[i].append(state[i])
            reward_deque[i].append(reward[i])
            action_deque[i].append(action[i])
            if len(state_deque[i]) == args.multi_step or done[i]:
                n_reward = multi_step_reward(reward_deque[i], args.gamma)
                n_state = state_deque[i][0]
                n_action = action_deque[i][0]
                replay_buffer.add(
                    n_state,
                    n_action,
                    next_state[i],
                    n_reward,
                    np.uint8(done[i]),
                    levels[i],
                )
                if done[i]:
                    reward_deque_i = list(reward_deque[i])
                    for j in range(1, len(reward_deque_i)):
                        n_reward = multi_step_reward(reward_deque_i[j:], args.gamma)
                        n_state = state_deque[i][j]
                        n_action = action_deque[i][j]
                        replay_buffer.add(
                            n_state,
                            n_action,
                            next_state[i],
                            n_reward,
                            np.uint8(done[i]),
                            levels[i],
                        )
                    episode_reward = info["return"]
                    episode_num += 1
                    wandb.log({"Train Episode Returns": episode_reward}, step=t * args.num_processes)
                    state_deque[i].clear()
                    reward_deque[i].clear()
                    action_deque[i].clear()

        state = next_state

        # Train agent after collecting sufficient data
        if t % args.train_freq == 0 and t >= args.start_timesteps:
            loss, grad_magnitude = agent.train(replay_buffer)
            wandb.log({"Value Loss": loss, "Gradient magnitude": grad_magnitude}, step=t * args.num_processes)

        if t % 10000 == 0:
            effective_rank = agent.Q.effective_rank()
            wandb.log({"Effective Rank of DQN": effective_rank}, step=t * args.num_processes)

        if (t >= args.start_timesteps and (t + 1) % args.eval_freq == 0) or t == num_steps - 1:
            train_eval_episode_rewards = eval_policy(args, agent, envs, atari_preprocessing, True)
            test_eval_episode_rewards = eval_policy(args, agent, envs, atari_preprocessing, False)
            wandb.log(
                {
                    "Train Evaluation Returns": np.mean(train_eval_episode_rewards),
                    "Test Evaluation Returns": np.mean(test_eval_episode_rewards),
                },
                step=t * args.num_processes,
            )


def eval_policy(args, policy, envs, atari_preprocessing, train_seeds=True, num_episodes=10, num_processes=1):
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3,
    }
    if train_seeds:
        eval_env = AtariVecEnv(envs.env_name, envs.seeds, num_processes, envs.device, atari_preprocessing)
    else:
        eval_env = AtariVecEnv(
            envs.env_name, envs.test_seeds, num_processes, envs.device, atari_preprocessing
        )

    eval_episode_rewards: List[float] = []
    state = eval_env.reset()

    while len(eval_episode_rewards) < num_episodes:
        if np.random.uniform() < args.eval_eps:
            action = (
                torch.LongTensor([eval_env.action_space.sample() for _ in range(num_processes)])
                .reshape(-1, 1)
                .to(args.device)
            )
        else:
            with torch.no_grad():
                action, _ = policy.select_action(state, eval=True)
        state, _, done, infos, _ = eval_env.step(action)
        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"])

    avg_reward = sum(eval_episode_rewards) / len(eval_episode_rewards)

    print("---------------------------------------")
    print(f"Evaluation over {num_episodes} episodes: {avg_reward}")
    print("---------------------------------------")
    return eval_episode_rewards


def multi_step_reward(rewards, gamma):
    ret = 0.0
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret


class AtariVecEnv:
    def __init__(self, env_name, seeds, num_processes, device, atari_preprocessing):
        self.atari_preprocessing = atari_preprocessing
        self.replay_action_probs = [(i / 1000.0) / 2 for i in range(1000)]
        np.random.shuffle(self.replay_action_probs)
        self.seeds = seeds
        self.test_seeds = [self.seeds[-1] + i for i in range(1, 1001 - len(self.seeds))]
        self.num_processes = num_processes
        self.env_name = env_name
        self.level_seeds = torch.LongTensor(
            [np.random.choice(self.seeds) for _ in range(self.num_processes)]
        ).unsqueeze(-1)
        self.envs = [
            utils.make_env(
                self.env_name,
                self.atari_preprocessing,
                self.replay_action_probs[int(self.level_seeds[i].item())],
            )[0]
            for i in range(self.num_processes)
        ]
        self.device = device
        self.observation_space = np.zeros(
            (self.atari_preprocessing["state_history"],) + self.envs[0].observation_space.shape
        )
        self.n_frames = self.atari_preprocessing["state_history"]
        self.action_space = self.envs[0].action_space

    def reset(self):
        self.returns = [0 for i in range(self.num_processes)]
        next_states = torch.zeros(
            (self.num_processes, self.n_frames, 84, 84), dtype=torch.float32, device=self.device
        )
        for idx, env in enumerate(self.envs):
            next_state = env.reset()
            next_state = (torch.FloatTensor(next_state) / 255.0).to(self.device)
            next_states[idx, :, :, :] = next_state

        return next_states

    def step(self, actions):
        next_states = torch.zeros(
            (self.num_processes, self.n_frames, 84, 84), dtype=torch.float32, device=self.device
        )
        rewards = torch.zeros((self.num_processes, 1), dtype=torch.float32, device=self.device)
        dones = []
        infos: List[dict] = [{} for _ in range(self.num_processes)]
        for idx, env in enumerate(self.envs):
            next_state, reward, done, info = env.step(actions[idx])
            self.returns[idx] += reward
            rewards[idx, :] = info[0]
            dones.append(done)
            if done:
                infos[idx]["return"] = self.returns[idx]
                self.returns[idx] = 0
                self.level_seeds[idx] = np.random.choice(self.seeds)
                self.envs[idx] = utils.make_env(
                    self.env_name, self.atari_preprocessing, self.replay_action_probs[self.level_seeds[idx]]
                )[0]
                next_state = self.envs[idx].reset()
                next_state = (torch.FloatTensor(next_state) / 255.0).to(self.device)
                next_states[idx, :, :, :] = next_state
            else:
                next_state = (torch.FloatTensor(next_state) / 255.0).to(self.device)
                next_states[idx, :, :, :] = next_state

        return next_states, rewards, dones, infos, self.level_seeds


def generate_seeds(num_seeds, base_seed=0):
    return [base_seed + i for i in range(num_seeds)]


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    train_seeds = generate_seeds(args.num_train_seeds)

    train(args, train_seeds)
