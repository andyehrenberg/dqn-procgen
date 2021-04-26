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

def train(args):
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

    args.num_actions = env.action_space.n
    agent = DDQN(args)

    num_updates = (args.T_max - args.start_timesteps) // args.train_freq

    replay_buffer = make_buffer(args, num_updates)

    episode_rewards = deque(maxlen=10)

    episode_reward = 0

    state_deque = deque(maxlen=args.multi_step)
    reward_deque = deque(maxlen=args.multi_step)
    action_deque = deque(maxlen=args.multi_step)

    num_steps = int(args.T_max)

    timer = timeit.default_timer
    update_start_time = timer()

    loss, grad_magnitude = None, None

    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 2500

    epsilon = lambda t: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * (t - args.start_timesteps) / epsilon_decay)

    state, done = env.reset(), False
    state = (torch.FloatTensor(state)/255.).to(args.device)

    episode_start = True
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in trange(num_steps):
        episode_timesteps += 1
        action = None
        if t < args.start_timesteps or np.random.uniform() < epsilon(t):
            action = torch.LongTensor([env.action_space.sample()]).reshape(-1, 1).to(args.device)
        else:
            action, _ = agent.select_action(state.unsqueeze(0))

        # Perform action and log results
        next_state, reward, done, info = env.step(action)
        next_state = (torch.FloatTensor(next_state)/255.).to(args.device)
        episode_reward += reward

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

        reward = info[0]

        state_deque.append(state)
        reward_deque.append(reward)
        action_deque.append(action)
        if len(state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(reward_deque, args.gamma)
            n_state = state_deque[0]
            n_action = action_deque[0]
            replay_buffer.add(n_state, n_action, next_state, n_reward, np.uint8(done), torch.Tensor([0]))

        state = next_state
        episode_start = False

        if done:
            wandb.log({"Train Episode Returns": episode_reward})
            state, done = env.reset(), False
            state = (torch.FloatTensor(state)/255.).to(args.device)
            episode_start = True
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Train agent after collecting sufficient data
        if (t + 1) % args.train_freq == 0 and t >= args.start_timesteps:
            loss, grad_magnitude = agent.train(replay_buffer)
            if args.wandb:
                wandb.log({"Value Loss": loss, "Gradient magnitude": grad_magnitude}, step=t)

        if (t >= args.start_timesteps and (t + 1) % args.eval_freq == 0) or t == num_steps - 1:
            eval_episode_rewards = eval_policy(args, agent, args.num_test_seeds)

            wandb.log({"Evaluation Returns": np.mean(eval_episode_rewards)}, step=t)

def eval_policy(args, policy, num_episodes=10):
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": True,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }
    eval_env, state_dim, num_actions = utils.make_env(args.env_name, atari_preprocessing)

    eval_episode_rewards = []
    state, done = eval_env.reset(), False
    state = (torch.FloatTensor(state)/255.).to(args.device)

    episode_returns = 0

    while len(eval_episode_rewards) < num_episodes:
        action = None
        if np.random.uniform() < args.eval_eps:
            action = torch.LongTensor([eval_env.action_space.sample()]).reshape(-1, 1).to(args.device)
        else:
            with torch.no_grad():
                action, _ = policy.select_action(state, eval=True)
        state, reward, done, info = eval_env.step(action)
        episode_returns += reward
        if done:
            eval_episode_rewards.append(episode_returns)
            episode_returns = 0
            state, done = eval_env.reset(), False
            state = (torch.FloatTensor(state)/255.).to(args.device)

    eval_env.close()

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

    train(args)
