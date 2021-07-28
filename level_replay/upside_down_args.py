import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description="DQN")

# Training parameters
parser.add_argument("--eval_freq", type=int, default=50, help="Evaluation frequency")
parser.add_argument("--T_max", type=int, default=25e6, help="Total environment steps")
parser.add_argument("--max_episode_length", type=int, default=108e3, help="Max timesteps in one episode")

# Model parameters
parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="learning rate")
parser.add_argument("--state_dim", type=tuple, default=(3, 64, 64))  # type: ignore
parser.add_argument("--no_cuda", type=lambda x: bool(strtobool(x)), default=False, help="disables gpu")
parser.add_argument("--adam_eps", type=float, default=1.5e-4)
parser.add_argument("--batch_size", type=int, default=1024, help="Batch Size")
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--no_ret_normalization", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--eps", type=float, default=1e-05)
parser.add_argument("--clip_param", type=float, default=0.2)
parser.add_argument("--disable_checkpoint", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--final_num_test_seeds", type=int, default=1000)
parser.add_argument("--full_train_distribution", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--replay_size", type=float, default=1e3)
parser.add_argument("--horizon_scale", type=float, default=0.02)
parser.add_argument("--return_scale", type=float, default=0.02)
parser.add_argument("--n_warm_up_episodes", type=int, default=50)
parser.add_argument("--last_few", type=int, default=50)
parser.add_argument("--n_updates_per_iter", type=int, default=100)
parser.add_argument("--n_episodes_per_iter", type=int, default=30)


# Environment parameters
parser.add_argument("--num_train_seeds", type=int, default=200)
parser.add_argument("--num_processes", type=int, default=64)
parser.add_argument("--env_name", default="starpilot")
parser.add_argument("--distribution_mode", default="easy")
parser.add_argument("--paint_vel_info", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--use_sequential_levels", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--start_level", type=int, default=0)
parser.add_argument("--render", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--checkpoint_interval", type=int, default=0)
parser.add_argument("--reward_clip", type=float, default=1)
parser.add_argument("--level_replay_alpha", type=float, default=1.0)
parser.add_argument("--level_replay_eps", type=float, default=0.05)
parser.add_argument("--level_replay_nu", type=float, default=0.5)
parser.add_argument("--level_replay_rho", type=float, default=1.0)
parser.add_argument("--level_replay_schedule", default="proportionate")
parser.add_argument("--level_replay_score_transform", default="rank")
parser.add_argument("--level_replay_strategy", default="random")
parser.add_argument("--level_replay_temperature", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=1)

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
parser.add_argument("--record_final_eval", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--xpid", default="latest")
parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--save_model", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--model_path", default="models/upside_down.tar")
