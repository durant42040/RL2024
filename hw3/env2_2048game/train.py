import warnings
from pyexpat import features

import numpy as np
import torch as th
import gymnasium as gym
from click.core import batch
from gymnasium import spaces
from gymnasium.envs.registration import register

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from wandb.integration.torch.wandb_torch import torch

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DQN, PPO, SAC
from collections import Counter

class CustomAdjacentTile(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 6144):
        super().__init__(observation_space, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        device = observations.device

        tile_indices = th.argmax(observations, dim=1)

        left_tiles = tile_indices[:, :, :-1]
        right_tiles = tile_indices[:, :, 1:]
        left_tiles = left_tiles.reshape(batch_size, -1)
        right_tiles = right_tiles.reshape(batch_size, -1)

        top_tiles = tile_indices[:, :-1, :]
        bottom_tiles = tile_indices[:, 1:, :]
        top_tiles = top_tiles.reshape(batch_size, -1)
        bottom_tiles = bottom_tiles.reshape(batch_size, -1)

        tile1_indices = th.cat([left_tiles, top_tiles], dim=1)
        tile2_indices = th.cat([right_tiles, bottom_tiles], dim=1)

        pairs_tensor = th.zeros(batch_size, 24, 16, 16, device=device)

        batch_indices = th.arange(batch_size, device=device).unsqueeze(1).expand(-1, 24).reshape(-1)
        pair_indices = th.arange(24, device=device).unsqueeze(0).expand(batch_size, -1).reshape(-1)
        tile1_indices_flat = tile1_indices.reshape(-1)
        tile2_indices_flat = tile2_indices.reshape(-1)

        pairs_tensor[batch_indices, pair_indices, tile1_indices_flat, tile2_indices_flat] = 1
        features = pairs_tensor.reshape(batch_size, -1)

        return features

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, n_input_channels: int = 16):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_input = th.zeros(1, n_input_channels, 4, 4)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "DQN_MLP",

    "algorithm": DQN,
    "policy_network": "CnnPolicy",
    "save_path": "models/DQN_MLP",

    "epoch_num": 500,
    "timesteps_per_epoch": 1000,
    "eval_episode_num": 100,
    "learning_rate": 1e-3,
}


def make_env():
    env = gym.make('2048-v0')
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    score = []
    highest = []
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        avg_highest += info[0]['highest']
        avg_score   += info[0]['score']
        score.append(info[0]['score'])
        highest.append(info[0]['highest'])

    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num

    return avg_score, avg_highest, score, highest

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    for epoch in range(config["epoch_num"]):

        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=WandbCallback(
                gradient_save_freq=100,
                verbose=2,
            ),
        )

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_score, avg_highest, score, highest = eval(eval_env, model, config["eval_episode_num"])

        print("Avg_score:  ", avg_score)
        print("Avg_highest:", avg_highest)
        print()
        wandb.log(
            {"avg_highest": avg_highest,
             "avg_score": avg_score,
             "highest": max(highest),
             }
        )

        c = Counter(highest)
        for item in (sorted(c.items(), key=lambda i: i[0])):
            print(f"{item[0]}: {item[1]}")

        ### Save best model
        if current_best < avg_highest:
            print("Saving Model")
            current_best = avg_highest
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")

        print("---------------")


if __name__ == "__main__":
    # Create wandb session (Uncomment to enable wandb logging)
    run = wandb.init(
        project="assignment_3",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id="reduce penalty to 10"
    )

    # Create training environment
    num_train_envs = 2
    train_env = DummyVecEnv([make_env for _ in range(num_train_envs)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        verbose=0,
        tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128),
        ),
        exploration_fraction=0.5,
        exploration_final_eps=0.01,
        batch_size=32,
    )

    print(model.policy)

    train(eval_env, model, my_config)