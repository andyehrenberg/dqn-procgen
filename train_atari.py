import logging
import os
from collections import deque
from typing import List

import numpy as np
import torch
import wandb

from level_replay import utils
from level_replay.algo.buffer import make_buffer
from level_replay.algo.policy import AtariAgent
from level_replay.atari_args import parser

os.environ["OMP_NUM_THREADS"] = "1"

last_checkpoint_time = None


def train(args):
    global last_checkpoint_time
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    if "cuda" in args.device.type:
        print("Using CUDA\n")
    args.optimizer_parameters = {"lr": args.learning_rate, "eps": args.adam_eps}

    torch.set_num_threads(1)
    utils.seed(args.seed)

    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project="atari",
        entity="andyehrenberg",
        config=vars(args),
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

    args.num_actions = env.action_space.n
    agent = AtariAgent(args)

    num_updates = (args.T_max - args.start_timesteps) // args.train_freq

    replay_buffer = make_buffer(args, num_updates, atari=True)

    episode_reward = 0

    state_deque: deque = deque(maxlen=args.multi_step)
    reward_deque: deque = deque(maxlen=args.multi_step)
    action_deque: deque = deque(maxlen=args.multi_step)

    num_steps = int(args.T_max)

    loss, grad_magnitude = None, None

    epsilon_start = args.initial_eps
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
        action = None
        if t < args.start_timesteps or np.random.uniform() < epsilon(t):
            action = torch.LongTensor([env.action_space.sample()]).reshape(-1, 1).to(args.device)
        else:
            action, _ = agent.select_action(state.unsqueeze(0))

        # Perform action and log results
        next_state, reward, done, info = env.step(action)
        next_state = (torch.FloatTensor(next_state) / 255.0).to(args.device)
        episode_reward += reward

        reward = info[0]

        state_deque.append(state)
        reward_deque.append(reward)
        action_deque.append(action)
        if len(state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(reward_deque, args.gamma)
            n_state = state_deque[0]
            n_action = action_deque[0]
            replay_buffer.add(n_state, n_action, next_state, n_reward, np.uint8(done), np.array([0]))

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

        if (t >= args.start_timesteps and (t + 1) % args.eval_freq == 0) or t == num_steps - 1:
            eval_episode_rewards = eval_policy(args, agent)
            wandb.log({"Evaluation Returns": np.mean(eval_episode_rewards)}, step=t)


def eval_policy(args, policy, num_episodes=10):
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": True,
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

    for video in eval_env.get_videos():
        wandb.log({"evaluation_behavior": video})

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

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    train(args)
