import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import sys
import time
from collections import deque
import timeit
import logging
from baselines.logger import HumanOutputFormat

from level_replay import utils
from level_replay.algo.policy import Rainbow, DDQN
from level_replay.algo.buffer import make_buffer
from level_replay.model import model_for_env_name
from level_replay.file_writer import FileWriter
from level_replay.envs import make_lr_venv
from level_replay.atari_args import parser

from tqdm import trange
import wandb

os.environ["OMP_NUM_THREADS"] = "1"

last_checkpoint_time = None

def train(args, seeds):
    global last_checkpoint_time
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    if 'cuda' in args.device.type:
        print('Using CUDA\n')
    args.optimizer_parameters = {'lr': args.learning_rate,  'eps': args.adam_eps}

    env_name = args.env_name

    torch.set_num_threads(1)
    utils.seed(args.seed)

    wandb.init(settings=wandb.Settings(start_method="fork"), project="test", entity="andyehrenberg", config=vars(args))

    atari_preprocessing = {
		"frame_skip": 4,
		"frame_size": 84,
		"state_history": 4,
		"done_on_life_loss": False,
		"reward_clipping": True,
		"max_episode_timesteps": 27e3
	}

    env, state_dim, num_actions = utils.make_env(args.env_name, atari_preprocessing)

    num_updates = (args.T_max // args.num_processes - args.start_timesteps) // args.train_freq

    replay_buffer = make_buffer(args, num_updates)

    agent = DDQN(args)

    state = envs.reset()

    episode_rewards = deque(maxlen=10)

    episode_reward = 0

    state_deque = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]
    reward_deque = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]
    action_deque = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]

    num_steps = int(
        args.T_max // args.num_processes
    )

    timer = timeit.default_timer
    update_start_time = timer()

    loss, grad_magnitude = None, None

    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 2500

    epsilon = lambda t: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * (t - args.start_timesteps) / epsilon_decay)

    for t in trange(num_steps):
        action = None
        if t < args.start_timesteps or np.random.uniform() < epsilon(t):
            action = torch.LongTensor([envs.action_space.sample() for _ in range(args.num_processes)]).reshape(-1, 1).to(args.device)
        else:
            action, _ = agent.select_action(state)

        # Perform action and log results
        next_state, reward, done, infos = envs.step(action)

        for i, info in enumerate(infos):
            if 'bad_transition' in info.keys():
                print("Bad transition")
            if 'episode' in info.keys():
                episode_reward = info['episode']['r']
                episode_rewards.append(episode_reward)
                if args.wandb:
                    wandb.log({"Train Episode Returns": episode_reward}, step=t*64)
                state_deque[i].clear()
                reward_deque[i].clear()
                action_deque[i].clear()
            if level_sampler:
                level_seeds[i][0] = info['level_seed']

        for i in range(args.num_processes):
            state_deque[i].append(state[i])
            reward_deque[i].append(reward[i])
            action_deque[i].append(action[i])
            if len(state_deque[i]) == args.multi_step or done[i]:
                n_reward = multi_step_reward(reward_deque[i], args.gamma)
                n_state = state_deque[i][0]
                n_action = action_deque[i][0]
                replay_buffer.add(n_state, n_action, next_state[i], n_reward, np.uint8(done[i]), 0)

        state = next_state

        # Train agent after collecting sufficient data
        if (t + 1) % args.train_freq == 0 and t >= args.start_timesteps:
            loss, grad_magnitude = agent.train(replay_buffer)
            if args.wandb:
                wandb.log({"Value Loss": loss, "Gradient magnitude": grad_magnitude}, step=t*64)

        if (t >= args.start_timesteps and (t + 1) % args.eval_freq == 0) or t == num_steps - 1:
            eval_episode_rewards = eval_policy(args, agent, args.num_test_seeds)
            train_eval_episode_rewards = eval_policy(args, agent, args.num_test_seeds, start_level=0, num_levels=args.num_train_seeds, seeds=seeds)

            wandb.log({
            "Test Evaluation Returns": np.mean(eval_episode_rewards), "Train Evaluation Returns": np.mean(train_eval_episode_rewards)
            }, step=t*64)

def eval_policy(args, policy, num_episodes, num_processes=1):
    if level_sampler:
        start_level = level_sampler.seed_range()[0]
        num_levels = 1

    env = ...

    eval_episode_rewards = []
    if level_sampler:
        state, _ = eval_envs.reset()
    else:
        state = eval_envs.reset()
    while len(eval_episode_rewards) < num_episodes:
        action = None
        if np.random.uniform() < 0.05:
            action = torch.LongTensor([eval_envs.action_space.sample() for _ in range(num_processes)]).reshape(-1, 1).to(args.device)
        else:
            with torch.no_grad():
                action, q = policy.select_action(state, eval=True)
        state, _, done, infos = eval_envs.step(action)
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                if progressbar:
                    progressbar.update(1)

    eval_envs.close()
    if progressbar:
        progressbar.close()

    avg_reward = sum(eval_episode_rewards)/len(eval_episode_rewards)

    print("---------------------------------------")
    print(f"Evaluation over {num_episodes} episodes: {avg_reward}")
    print("---------------------------------------")
    return eval_episode_rewards

def multi_step_reward(rewards, gamma):
    ret = 0.
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

    train(args, train_seeds)
