# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import collections
import timeit
import random

import numpy
import torch
import gym
import cv2
import numpy as np


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir, pattern='*'):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, pattern))
        for f in files:
            os.remove(f)


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Timings:
    """Not thread-safe."""

    def __init__(self):
        self._means = collections.defaultdict(int)
        self._vars = collections.defaultdict(int)
        self._counts = collections.defaultdict(int)
        self.reset()

    def reset(self):
        self.last_time = timeit.default_timer()

    def time(self, name):
        """Save an update for event `name`.

        Nerd alarm: We could just store a
            collections.defaultdict(list)
        and compute means and standard deviations at the end. But thanks to the
        clever math in Sutton-Barto
        (http://www.incompleteideas.net/book/first/ebook/node19.html) and
        https://math.stackexchange.com/a/103025/5051 we can update both the
        means and the stds online. O(1) FTW!
        """
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v ** 0.5 for k, v in self._vars.items()}

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fms +- %.6fms (%.2f%%) " % (
                k,
                1000 * means[k],
                1000 * stds[k],
                100 * means[k] / total,
            )
        result += "\nTotal: %.6fms" % (1000 * total)
        return result

# Atari Preprocessing
# Code is based on https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
class AtariPreprocessing(object):
	def __init__(
		self,
		env,
		frame_skip=4,
		frame_size=84,
		state_history=4,
		done_on_life_loss=False,
		reward_clipping=True, # Clips to a range of -1,1
		max_episode_timesteps=27000
	):
		self.env = env.env
		self.done_on_life_loss = done_on_life_loss
		self.frame_skip = frame_skip
		self.frame_size = frame_size
		self.reward_clipping = reward_clipping
		self._max_episode_steps = max_episode_timesteps
		self.observation_space = np.zeros((frame_size, frame_size))
		self.action_space = self.env.action_space

		self.lives = 0
		self.episode_length = 0

		# Tracks previous 2 frames
		self.frame_buffer = np.zeros(
			(2,
			self.env.observation_space.shape[0],
			self.env.observation_space.shape[1]),
			dtype=np.uint8
		)
		# Tracks previous 4 states
		self.state_buffer = np.zeros((state_history, frame_size, frame_size), dtype=np.uint8)


	def reset(self):
		self.env.reset()
		self.lives = self.env.ale.lives()
		self.episode_length = 0
		self.env.ale.getScreenGrayscale(self.frame_buffer[0])
		self.frame_buffer[1] = 0

		self.state_buffer[0] = self.adjust_frame()
		self.state_buffer[1:] = 0
		return self.state_buffer


	# Takes single action is repeated for frame_skip frames (usually 4)
	# Reward is accumulated over those frames
	def step(self, action):
		total_reward = 0.
		self.episode_length += 1

		for frame in range(self.frame_skip):
			_, reward, done, _ = self.env.step(action)
			total_reward += reward

			if self.done_on_life_loss:
				crt_lives = self.env.ale.lives()
				done = True if crt_lives < self.lives else done
				self.lives = crt_lives

			if done:
				break

			# Second last and last frame
			f = frame + 2 - self.frame_skip
			if f >= 0:
				self.env.ale.getScreenGrayscale(self.frame_buffer[f])

		self.state_buffer[1:] = self.state_buffer[:-1]
		self.state_buffer[0] = self.adjust_frame()

		done_float = float(done)
		if self.episode_length >= self._max_episode_steps:
			done = True

		return self.state_buffer, total_reward, done, [np.clip(total_reward, -1, 1), done_float]


	def adjust_frame(self):
		# Take maximum over last two frames
		np.maximum(
			self.frame_buffer[0],
			self.frame_buffer[1],
			out=self.frame_buffer[0]
		)

		# Resize
		image = cv2.resize(
			self.frame_buffer[0],
			(self.frame_size, self.frame_size),
			interpolation=cv2.INTER_AREA
		)
		return np.array(image, dtype=np.uint8)


	def seed(self, seed):
		self.env.seed(seed)

# Create environment, add wrapper if necessary and create env_properties
def make_env(env_name, atari_preprocessing):
	env = gym.make(env_name)

	env = AtariPreprocessing(env, **atari_preprocessing)

	state_dim = (
		atari_preprocessing["state_history"],
		atari_preprocessing["frame_size"],
		atari_preprocessing["frame_size"]
	)

	return (
		env,
		state_dim,
		env.action_space.n
	)
