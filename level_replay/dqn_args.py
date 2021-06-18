import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description="DQN")

# Training parameters
parser.add_argument(
    "--start_timesteps", type=int, default=2000, help="Timesteps until using DQN to take actions"
)
parser.add_argument("--train_freq", type=int, default=64, help="Number of steps between DQN updates")
parser.add_argument("--eval_freq", type=int, default=1000, help="Evaluation frequency")
parser.add_argument("--T_max", type=int, default=50e6, help="Total environment steps")
parser.add_argument("--max_episode_length", type=int, default=108e3, help="Max timesteps in one episode")

# Model parameters
parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="learning rate")
parser.add_argument("--num_actions", type=int, default=15, help="Action space dimension")
parser.add_argument("--state_dim", type=tuple, default=(3, 64, 64))  # type: ignore
parser.add_argument("--no_cuda", type=lambda x: bool(strtobool(x)), default=False, help="disables gpu")
parser.add_argument("--discount", type=float, default=0.99, help="Discount factor on future returns")
parser.add_argument("--adam_eps", type=float, default=1.5e-4)
parser.add_argument("--optimizer", default="Adam", help="Optimizer to use")
parser.add_argument(
    "--polyak_target_update",
    type=lambda x: bool(strtobool(x)),
    default=False,
    help="Whether to use polyak update to target network",
)
parser.add_argument("--target_update", type=int, default=2000, help="How often to update target network")
parser.add_argument("--tau", type=float, default=0.005, help="tau")
parser.add_argument("--initial_eps", type=float, default=1, help="intial epsilon")
parser.add_argument("--end_eps", type=float, default=0.001, help="end epsilon")
parser.add_argument("--eps_decay_period", type=int, default=25e4)
parser.add_argument("--eval_eps", type=float, default=0.05)
parser.add_argument("--min_priority", type=float, default=1e-1)
parser.add_argument("--V_min", type=float, default=0)
parser.add_argument("--V_max", type=float, default=100)
parser.add_argument("--batch_size", type=int, default=512, help="Batch Size")
parser.add_argument("--norm_clip", type=float, default=10)
parser.add_argument("--atoms", type=int, default=51, help="Number of atoms for distributional RL")
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--noisy_std", type=float, default=0.5, help="Standard deviation for noisy layers")
parser.add_argument("--model", default=None)
parser.add_argument("--history_length", type=int, default=4)
parser.add_argument("--multi_step", type=int, default=1, help="Number of steps for multi step rewards")
parser.add_argument("--priority_weight", type=float, default=0.4)
parser.add_argument("--priority_exponent", type=float, default=0.5)
parser.add_argument("--t", type=int, default=0)
parser.add_argument("--no_ret_normalization", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--eps", type=float, default=1e-05)
parser.add_argument("--alpha", type=float, default=0.6)
parser.add_argument("--clip_param", type=float, default=0.2)
parser.add_argument("--disable_checkpoint", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--final_num_test_seeds", type=int, default=1000)
parser.add_argument("--full_train_distribution", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--PER", type=lambda x: bool(strtobool(x)), default=True, help="Whether to use PER")

# Environment parameters
parser.add_argument("--num_processes", type=int, default=64)
parser.add_argument("--env_name", default="starpilot")
parser.add_argument("--distribution_mode", default="easy")
parser.add_argument("--paint_vel_info", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--start_level", type=int, default=0)
parser.add_argument("--render", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--checkpoint_interval", type=int, default=0)
parser.add_argument("--memory_capacity", type=float, default=1e6)
parser.add_argument("--reward_clip", type=float, default=1)

# Level Replay parameters
parser.add_argument("--level_replay_alpha", type=float, default=1.0)
parser.add_argument("--level_replay_eps", type=float, default=0.05)
parser.add_argument("--level_replay_nu", type=float, default=0.5)
parser.add_argument("--level_replay_rho", type=float, default=1.0)
parser.add_argument("--level_replay_schedule", default="proportionate")
parser.add_argument("--level_replay_score_transform", default="rank")
parser.add_argument("--level_replay_strategy", default="random")
parser.add_argument("--level_replay_temperature", type=float, default=0.1)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--num_env_steps", type=float, default=25000000.0)
parser.add_argument("--num_mini_batch", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=256)
parser.add_argument("--num_test_seeds", type=int, default=10)
parser.add_argument("--num_train_seeds", type=int, default=200)
parser.add_argument("--ppo_epoch", type=int, default=3)
parser.add_argument("--save_interval", type=int, default=60)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--seed_path", default=None)
parser.add_argument("--staleness_coef", type=float, default=0.1)
parser.add_argument("--staleness_temperature", type=float, default=1.0)
parser.add_argument("--staleness_transform", default="power")
parser.add_argument("--value_loss_coef", type=float, default=0.5)
parser.add_argument("--weight_log_interval", type=int, default=1)

# Logging
parser.add_argument(
    "--wandb",
    type=lambda x: bool(strtobool(x)),
    default=True,
    help="Whether to log with wandb or save results locally",
)
parser.add_argument(
    "--wandb-tags",
    type=str,
    default="",
    help="Additional tags for this wandb run",
)
parser.add_argument(
    "--wandb-group",
    type=str,
    default="",
    help="Wandb group for this run",
)
parser.add_argument("--log_dir", default="~/PLEXR/logs/")
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--xpid", default="latest")
parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--save_model", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--model_path", default="models/model.tar")
