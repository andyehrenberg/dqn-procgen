from collections import deque
from typing import List

import numpy as np
import torch

from level_replay.envs import make_dqn_lr_venv
from level_replay.algo.buffer import RolloutStorage


def warm_up(replay_buffer, agent, args):
    num_levels = 1
    level_sampler_args = dict(
        num_actors=len(args.seeds),
        strategy="random",
        replay_schedule=args.level_replay_schedule,
        score_transform=args.level_replay_score_transform,
        temperature=args.level_replay_temperature,
        eps=args.level_replay_eps,
        rho=args.level_replay_rho,
        nu=args.level_replay_nu,
        alpha=args.level_replay_alpha,
        staleness_coef=args.staleness_coef,
        staleness_transform=args.staleness_transform,
        staleness_temperature=args.staleness_temperature,
    )
    envs, level_sampler = make_dqn_lr_venv(
        num_envs=len(args.seeds),
        env_name=args.env_name,
        seeds=args.seeds,
        device=args.device,
        num_levels=num_levels,
        start_level=args.start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        use_sequential_levels=args.use_sequential_levels,
        level_sampler_args=level_sampler_args,
    )

    rollouts = RolloutStorage(
        args.num_steps, len(args.seeds), envs.observation_space.shape, envs.action_space
    )

    level_seeds = torch.zeros(len(args.seeds))
    if level_sampler:
        state, level_seeds = envs.reset()
    else:
        state = envs.reset()
    level_seeds = level_seeds.unsqueeze(-1)

    rollouts.obs[0].copy_(state)
    rollouts.to(args.device)

    state_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(len(args.seeds))]
    reward_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(len(args.seeds))]
    action_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(len(args.seeds))]

    num_steps = int(100000 / len(args.seeds))

    for _ in range(num_steps):
        action = (
            torch.LongTensor([envs.action_space.sample() for _ in range(len(args.seeds))])
            .reshape(-1, 1)
            .to(args.device)
        )
        value = agent.get_value(state)

        # Perform action and log results
        next_state, reward, done, infos = envs.step(action)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

        for i, info in enumerate(infos):
            if "bad_transition" in info.keys():
                print("Bad transition")
            if level_sampler:
                level_seed = info["level_seed"]
                if level_seeds[i][0] != level_seed:
                    level_seeds[i][0] = level_seed
            state_deque[i].append(state[i])
            reward_deque[i].append(reward[i])
            action_deque[i].append(action[i])
            if len(state_deque[i]) == args.multi_step or done[i]:
                n_reward = multi_step_reward(reward_deque[i], args.gamma)
                n_state = state_deque[i][0]
                n_action = action_deque[i][0]
                replay_buffer.add(
                    n_state, n_action, next_state[i], n_reward, np.uint8(done[i]), level_seeds[i]
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
                            level_seeds[i],
                        )
            if "episode" in info.keys():
                state_deque[i].clear()
                reward_deque[i].clear()
                action_deque[i].clear()

        rollouts.insert(next_state, action, value.unsqueeze(1), torch.Tensor(reward), masks, level_seeds)

        state = next_state

        if (rollouts.step + 1) == rollouts.num_steps:
            obs_id = rollouts.obs[-1]
            next_value = agent.get_value(obs_id).unsqueeze(1).detach()

            rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

            replay_buffer.update_with_rollouts(rollouts)

            rollouts.after_update()

            replay_buffer.after_update()

    envs.close()

    return num_steps


def multi_step_reward(rewards, gamma):
    ret = 0.0
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret
