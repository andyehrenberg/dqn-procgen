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
from level_replay.algo.buffer import make_buffer
from level_replay.model import model_for_env_name
from level_replay.storage import RolloutStorage
from level_replay.file_writer import FileWriter
from level_replay.envs import make_lr_venv
from level_replay.dqn_args import parser
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

    torch.set_num_threads(1)

    utils.seed(args.seed)

    # Configure logging
    if not args.wandb:
        if args.xpid is None:
            args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))
        plogger = FileWriter(
            xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir,
            seeds=seeds,
        )
        stdout_logger = HumanOutputFormat(sys.stdout)

        #checkpointpath = os.path.expandvars(
            #os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, "model.tar"))
        #)

    else:
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

    replay_buffer = make_buffer(args)

    agent = DDQN(args)
    '''
    def checkpoint():
        if args.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.online_net.state_dict(),
                #"optimizer_state_dict": agent.optimizer.state_dict(),
                "args": vars(args),
            },
            checkpointpath,
        )
    '''

    level_seeds = torch.zeros(args.num_processes)
    if level_sampler:
        state, level_seeds = envs.reset()
    else:
        state = envs.reset()
    level_seeds = level_seeds.unsqueeze(-1)
    episode_rewards = deque(maxlen=10)

    episode_start = True
    episode_reward = 0
    episode_num = 0

    #state_deque = deque(maxlen=args.multi_step)
    #reward_deque = deque(maxlen=args.multi_step)
    #action_deque = deque(maxlen=args.multi_step)

    num_steps = int(
        args.T_max // args.num_processes
    )

    timer = timeit.default_timer
    update_start_time = timer()

    #losses = []
    loss, grad_magnitude = None, None

    for t in trange(num_steps):
        if t < args.start_timesteps or np.random.uniform() < 0.05:
            action = torch.LongTensor([envs.action_space.sample() for _ in range(args.num_processes)]).reshape(-1, 1).to(args.device)
        else:
            action, _ = agent.select_action(state)

        # Perform action and log results
        next_state, reward, done, infos = envs.step(action)
        #Uncomment out if using multi step returns
        #state_deque.append(state)
        #reward_deque.append(reward)
        #action_deque.append(action)

        #if len(state_deque) == args.multi_step or done:
            #n_reward = multi_step_reward(reward_deque, args.gamma)
            #n_state = state_deque[0]
            #n_action = action_deque[0]

        # For atari, info[0] = clipped reward, info[1] = done_float
        for i, info in enumerate(infos):
            if 'bad_transition' in info.keys():
                print("Bad transition")
            if 'episode' in info.keys():
                episode_reward = info['episode']['r']
                episode_rewards.append(episode_reward)
                if args.wandb:
                    wandb.log({"Train Episode Returns": episode_reward}, step=t)
            if level_sampler:
                level_seeds[i][0] = info['level_seed']

        replay_buffer.add(state, action, next_state, reward, np.float32(done), level_seeds)

        state = next_state
        episode_start = False

        # Train agent after collecting sufficient data
        if (t + 1) % args.train_freq == 0 and t >= args.start_timesteps:
            loss, grad_magnitude = agent.train(replay_buffer)
            if args.wandb:
                wandb.log({"Value Loss": loss, "Gradient magnitude": grad_magnitude}, step=t)
            #losses.append(loss)

        if (t >= args.start_timesteps and (t + 1) % args.eval_freq == 0) or t == num_steps - 1:
            if not args.wandb:
                logging.info(f"\nUpdate {t//args.train_freq} done, {t} steps\n  ")
                logging.info(f"\nEvaluating on {args.num_test_seeds} test levels...\n  ")
            eval_episode_rewards = eval_policy(args, agent, args.num_test_seeds)
            if not args.wandb:
                logging.info(f"\nEvaluating on {args.num_test_seeds} train levels...\n  ")
            train_eval_episode_rewards = eval_policy(args, agent, args.num_test_seeds, start_level=0, num_levels=args.num_train_seeds, seeds=seeds)

            if args.wandb:
                wandb.log({
                "Test Evaluation Returns": np.mean(eval_episode_rewards), "Train Evaluation Returns": np.mean(train_eval_episode_rewards)
                }, step=t)

            else:
                stats = {
                    "step": t,
                    "value_loss": loss,
                    "grad_magnitude": grad_magnitude,
                    "train:mean_episode_return": np.mean(episode_rewards),
                    "train:median_episode_return": np.median(episode_rewards),
                    "test:mean_episode_return": np.mean(eval_episode_rewards),
                    "test:median_episode_return": np.median(eval_episode_rewards),
                    "train_eval:mean_episode_return": np.mean(train_eval_episode_rewards),
                    "train_eval:median_episode_return": np.median(train_eval_episode_rewards)
                }

                if t == num_updates - 1:
                    logging.info(f"\nLast update: Evaluating on {args.num_test_seeds} test levels...\n  ")
                    final_eval_episode_rewards = eval_policy(args, agent, args.final_num_test_seeds)

                    mean_final_eval_episode_rewards = np.mean(final_eval_episode_rewards)
                    median_final_eval_episide_rewards = np.median(final_eval_episode_rewards)

                    plogger.log_final_test_eval({
                        'num_test_seeds': args.final_num_test_seeds,
                        'mean_episode_return': mean_final_eval_episode_rewards,
                        'median_episode_return': median_final_eval_episide_rewards
                    })

                plogger.log(stats)

def generate_seeds(num_seeds, base_seed=0):
    return [base_seed + i for i in range(num_seeds)]

def load_seeds(seed_path):
    seed_path = os.path.expandvars(os.path.expanduser(seed_path))
    seeds = open(seed_path).readlines()
    return [int(s) for s in seeds]

def eval_policy(args, policy, num_episodes, num_processes=1, deterministic=False,
    start_level=0, num_levels=0, seeds=None, level_sampler=None, progressbar=None):
    if level_sampler:
        start_level = level_sampler.seed_range()[0]
        num_levels = 1

    eval_envs, level_sampler = make_lr_venv(
        num_envs=num_processes, env_name=args.env_name,
        seeds=seeds, device=args.device,
        num_levels=num_levels, start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler=level_sampler)

    eval_episode_rewards = []
    if level_sampler:
        state, _ = eval_envs.reset()
    else:
        state = eval_envs.reset()
    while len(eval_episode_rewards) < num_episodes:
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

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    if args.seed_path:
        train_seeds = load_seeds(args.seed_path)
    else:
        train_seeds = generate_seeds(args.num_train_seeds)

    train(args, train_seeds)
