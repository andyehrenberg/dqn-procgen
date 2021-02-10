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

import numpy as np
import torch
from baselines.logger import HumanOutputFormat

from level_replay import utils
from level_replay.algo.dqn import RainbowDQN, DQN
from level_replay.algo.policy import Rainbow, DDQN
from level_replay.algo.buffer import Buffer, PrioritizedBuffer
from level_replay.model import model_for_env_name
from level_replay.storage import RolloutStorage
from level_replay.file_writer import FileWriter
from level_replay.envs import make_lr_venv
from level_replay.dqn_args import parser
from test import evaluate
from tqdm import trange
import wandb

os.environ["OMP_NUM_THREADS"] = "1"

last_checkpoint_time = None

def train(args, seeds):
    global last_checkpoint_time
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if 'cuda' in device.type:
        print('Using CUDA\n')
    args.optimizer_parameters = {'lr': args.learning_rate,  'eps': args.adam_eps}

    torch.set_num_threads(1)

    utils.seed(args.seed)

    # Configure logging
    #if args.xpid is None:
    #    args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    #log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))
    #plogger = FileWriter(
    #    xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir,
    #    seeds=seeds,
    #)
    #stdout_logger = HumanOutputFormat(sys.stdout)

    #checkpointpath = os.path.expandvars(
        #os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, "model.tar"))
    #)

    # Configure actor envs
    wandb.init(project="test", entity="andyehrenberg", config=vars(args))
    start_level = 0

    num_levels = 1
    level_sampler_args = dict(
        num_actors=args.num_processes,
        strategy=args.level_replay_strategy,
        replay_schedule=args.level_replay_schedule,
        score_transform=args.level_replay_score_transform,
        temperature=args.level_replay_temperature,
        eps=args.level_replay_eps,
        rho=args.level_replay_rho,
        nu=args.level_replay_nu,
        alpha=args.level_replay_alpha,
        staleness_coef=args.staleness_coef,
        staleness_transform=args.staleness_transform,
        staleness_temperature=args.staleness_temperature
    )
    envs, level_sampler = make_lr_venv(
        num_envs=args.num_processes, env_name=args.env_name,
        seeds=seeds, device=args.device,
        num_levels=num_levels, start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler_args=level_sampler_args)

    prioritized = False#True
    replay_buffer = Buffer(
        args.state_dim,
        args.batch_size,
        args.memory_capacity,
        args.device, prioritized
    )

    agent = DDQN(args)
    wandb.watch(agent)

    #def checkpoint():
    #    if args.disable_checkpoint:
    #        return
    #    logging.info("Saving checkpoint to %s", checkpointpath)
    #    torch.save(
    #        {
    #            "model_state_dict": model.online_net.state_dict(),
    #            #"optimizer_state_dict": agent.optimizer.state_dict(),
    #            "args": vars(args),
    #        },
    #        checkpointpath,
    #    )

    level_seeds = torch.zeros(args.num_processes)
    if level_sampler:
        state, level_seeds = envs.reset()
    else:
        state = envs.reset()
    level_seeds.unsqueeze(-1)
    episode_rewards = deque(maxlen=10)

    episode_start = True
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    #state_deque = deque(maxlen=args.multi_step)
    #reward_deque = deque(maxlen=args.multi_step)
    #action_deque = deque(maxlen=args.multi_step)
    #num_updates = int(
        #args.num_env_steps) // args.num_steps // args.num_processes

    timer = timeit.default_timer
    update_start_time = timer()

    #losses = []
    for t in trange(int(args.T_max)):
        episode_timesteps += 1

        #if args.train_behavioral:
        if t < args.start_timesteps or np.random.uniform() < 0.01:
            action = torch.LongTensor([envs.action_space.sample()]).reshape(-1, 1)
        else:
            action, _ = agent.select_action(state)

        # Perform action and log results
        next_state, reward, done, infos = envs.step(action)
        #print(action, reward, done, infos)
        #state_deque.append(state)
        #reward_deque.append(reward)
        #action_deque.append(action)

        #if done:#len(state_deque) == args.multi_step or done:
            #n_reward = multi_step_reward(reward_deque, args.gamma)
            #n_state = state_deque[0]
            #n_action = action_deque[0]
        replay_buffer.add(state, action, next_state, reward, np.float32(done), done, episode_start)

        state = next_state
        episode_start = False

        # Only consider "done" if episode terminates due to failure condition
        #done_float = float(done) if episode_timesteps < envs._max_episode_steps else 0

        # For atari, info[0] = clipped reward, info[1] = done_float
        for i, info in enumerate(infos):
            if 'bad_transition' in info.keys():
                print("Bad transition")
            #print(info)
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
            if level_sampler:
                level_seeds[i] = info['level_seed']

        # Train agent after collecting sufficient data
        if  (t + 1) % args.train_freq == 0 and t >= args.start_timesteps:
            loss = agent.train(replay_buffer)
            wandb.log({"Value Loss": loss})
            #losses.append(loss)

        if t >= args.start_timesteps and t % args.target_update == 0:
            agent.copy_target_update()

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            #print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward}")
            # Reset environment
            state, level_seeds = envs.reset()
            done = False
            episode_start = True
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if t >= args.start_timesteps and (t + 1) % args.eval_freq == 0:
            #evaluations.append(eval_policy(agent, seeds, start_level, level_sampler_args, num_levels))
            evaluation = eval_policy(agent, seeds, start_level, level_sampler_args, num_levels)
            wandb.log({"Evaluation Returns": evaluation})
            #np.save(f"./results/log.npy", evaluations)


def generate_seeds(num_seeds, base_seed=0):
    return [base_seed + i for i in range(num_seeds)]


def load_seeds(seed_path):
    seed_path = os.path.expandvars(os.path.expanduser(seed_path))
    seeds = open(seed_path).readlines()
    return [int(s) for s in seeds]

def eval_policy(policy, seeds, start_level, level_sampler_args, num_levels, eval_episodes=10):
    #seeds = generate_seeds(args.num_train_seeds)
    eval_envs, level_sampler = make_lr_venv(
        num_envs=args.num_processes, env_name=args.env_name,
        seeds=seeds, device=args.device,
        num_levels=num_levels, start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler_args=level_sampler_args)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, level_seeds = eval_envs.reset()
        done = False
        while not done:
            action, q = policy.select_action(state, eval=True)
            state, _, done, infos = eval_envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    avg_reward += info['episode']['r']

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward}")
    print("---------------------------------------")
    return avg_reward

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret


if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    if args.seed_path:
        train_seeds = load_seeds(args.seed_path)
    else:
        train_seeds = generate_seeds(args.num_train_seeds)

    train(args, train_seeds)
