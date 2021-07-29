import os

import numpy as np
import torch
import wandb

from level_replay import utils
from level_replay.upside_down_args import parser
from level_replay.algo import upside_down
from level_replay.utils import ppo_normalise_reward


def train(args, seeds):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    if "cuda" in args.device.type:
        print("Using CUDA\n")
    args.seeds = seeds
    args.no_ret_normalization = True

    torch.set_num_threads(1)

    utils.seed(args.seed)

    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project="off-policy-procgen",
        entity="ucl-dark",
        config=vars(args),
        tags=["ddqn", "procgen"] + (args.wandb_tags.split(",") if args.wandb_tags else []),
        group=args.wandb_group,
    )

    print("Warming up memory, creating policy")
    mem, env_steps, policy = upside_down.warm_up(args)

    it = 0

    while env_steps < args.T_max:
        it += 1
        loss = upside_down.update(policy, mem, args.n_updates_per_iter)
        wandb.log({"Loss": loss}, step=env_steps)
        commands = upside_down.sample_commands(mem, args.last_few)
        env_steps = upside_down.generate_episodes(
            policy,
            mem,
            commands,
            env_steps,
            args,
        )
        if (it % args.eval_freq) == 0:
            commands = upside_down.sample_commands(mem, args.last_few)
            mean_train_rewards = np.mean(
                upside_down.evaluate(
                    args, policy, commands, start_level=0, num_levels=args.num_train_seeds, seeds=args.seeds
                )
            )
            mean_test_rewards = np.mean(upside_down.evaluate(args, policy, commands))
            wandb.log(
                {
                    "Test Evaluation Returns": mean_test_rewards,
                    "Train Evaluation Returns": mean_train_rewards,
                    "Generalization Gap:": mean_train_rewards - mean_test_rewards,
                    "Test Evaluation Returns (normalised)": ppo_normalise_reward(
                        mean_test_rewards, args.env_name
                    ),
                    "Train Evaluation Returns (normalised)": ppo_normalise_reward(
                        mean_train_rewards, args.env_name
                    ),
                },
                step=env_steps,
            )
            print("---------------------------------------")
            print(f"Evaluation on train episodes at iteration {it}: {mean_train_rewards}")
            print("---------------------------------------")

    commands = upside_down.sample_commands(mem, args.last_few)
    mean_train_rewards = np.mean(
        upside_down.evaluate(
            args,
            policy,
            commands,
            start_level=0,
            num_levels=args.num_train_seeds,
            seeds=args.seeds,
            num_episodes=100,
        )
    )
    mean_test_rewards = np.mean(upside_down.evaluate(args, policy, commands, num_episodes=100))
    wandb.log(
        {
            "Final Test Evaluation Returns": mean_test_rewards,
            "Final Train Evaluation Returns": mean_train_rewards,
            "Final Generalization Gap:": mean_train_rewards - mean_test_rewards,
            "Final Test Evaluation Returns (normalised)": ppo_normalise_reward(
                mean_test_rewards, args.env_name
            ),
            "Final Train Evaluation Returns (normalised)": ppo_normalise_reward(
                mean_train_rewards, args.env_name
            ),
        },
        step=env_steps,
    )

    if args.save_model:
        print(f"Saving model to {args.model_path}")
        if "models" not in os.listdir():
            os.mkdir("models")
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "args": vars(args),
            },
            args.model_path,
        )
        wandb.save(args.model_path)


def generate_seeds(num_seeds, base_seed=0):
    return [base_seed + i for i in range(num_seeds)]


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    train_seeds = generate_seeds(args.num_train_seeds)

    train(args, train_seeds)
