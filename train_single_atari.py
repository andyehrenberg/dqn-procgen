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


def train(args):
    args.num_processes = 1
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    if "cuda" in args.device.type:
        print("Using CUDA\n")
    args.optimizer_parameters = {"lr": args.learning_rate, "eps": args.adam_eps}
    args.seeds = None

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

    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3,
    }

    env, state_dim, num_actions = utils.make_env(args.env_name, atari_preprocessing)

    replay_buffer = make_buffer(args, env, atari=True)

    agent = DQNAgent(args, env)

    state_deque: deque = deque(maxlen=args.multi_step)
    reward_deque: deque = deque(maxlen=args.multi_step)
    action_deque: deque = deque(maxlen=args.multi_step)

    num_steps = int(args.T_max)

    epsilon_start = 1.0
    epsilon_final = args.end_eps
    epsilon_decay = args.eps_decay_period

    def epsilon(t):
        return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
            -1.0 * (t - args.start_timesteps) / epsilon_decay
        )

    state, done = env.reset(), False
    state = (torch.FloatTensor(state) / 255.0).to(args.device)

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(num_steps):
        episode_timesteps += 1
        if t < args.start_timesteps or np.random.uniform() < epsilon(t):
            action = torch.LongTensor([env.action_space.sample()]).reshape(-1, 1).to(args.device)
        else:
            action, _ = agent.select_action(state.unsqueeze(0))

        # Perform action and log results
        next_state, reward, done, info = env.step(action)

        state_deque.append(state)
        reward_deque.append(reward)
        action_deque.append(action)
        if len(state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(reward_deque, args.gamma)
            n_state = state_deque[0]
            n_action = action_deque[0]
            replay_buffer.add(n_state, n_action, next_state, n_reward, np.uint8(done), np.array([0]))
            if done:
                reward_deque_i = list(reward_deque)
                for j in range(1, len(reward_deque_i)):
                    n_reward = multi_step_reward(reward_deque_i[j:], args.gamma)
                    n_state = state_deque[j]
                    n_action = action_deque[j]
                    replay_buffer.add(
                        n_state,
                        n_action,
                        next_state,
                        n_reward,
                        np.uint8(done),
                        np.array([0]),
                    )

        state = next_state

        if done:
            wandb.log({"Train Episode Returns": episode_reward}, step=t)
            state, done = env.reset(), False
            state = (torch.FloatTensor(state) / 255.0).to(args.device)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Train agent after collecting sufficient data
        if (t + 1) % args.train_freq == 0 and t >= args.start_timesteps:
            loss, grad_magnitude = agent.train(replay_buffer)
            wandb.log({"Value Loss": loss, "Gradient magnitude": grad_magnitude}, step=t)

        if t % 10000 == 0:
            effective_rank = agent.Q.effective_rank()
            wandb.log({"Effective Rank of DQN": effective_rank}, step=t)

        if (t >= args.start_timesteps and (t + 1) % args.eval_freq == 0) or t == num_steps - 1:
            eval_episode_rewards = eval_policy(args, agent)
            wandb.log({"Evaluation Returns": np.mean(eval_episode_rewards)}, step=t)


def eval_policy(args, policy, num_episodes=10):
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3,
    }
    eval_env, _, _ = utils.make_env(args.env_name, atari_preprocessing, record_runs=True)
    eval_env.seed(args.seed + 100)

    eval_episode_rewards: List[float] = []
    state, done = eval_env.reset(), False
    state = (torch.FloatTensor(state) / 255.0).to(args.device)

    episode_returns = 0

    while len(eval_episode_rewards) < num_episodes:
        if np.random.uniform() < args.eval_eps:
            action = torch.LongTensor([eval_env.action_space.sample()]).reshape(-1, 1).to(args.device)
        else:
            with torch.no_grad():
                action, _ = policy.select_action(state.unsqueeze(0), eval=True)
        state, reward, done, info = eval_env.step(action)
        state = (torch.FloatTensor(state) / 255.0).to(args.device)
        episode_returns += reward
        if done:
            eval_episode_rewards.append(episode_returns)
            episode_returns = 0
            state, done = eval_env.reset(), False
            state = (torch.FloatTensor(state) / 255.0).to(args.device)

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


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    train(args)
