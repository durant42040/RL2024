import warnings
import torch as th
import gymnasium as gym
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

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "DQN_CNN",

    "algorithm": DQN,
    "policy_network": "CnnPolicy",
    "save_path": "models/DQN_CNN",

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
        if current_best < avg_score:
            print("Saving Model")
            current_best = avg_score
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")

        print("---------------")


if __name__ == "__main__":
    # Create wandb session (Uncomment to enable wandb logging)
    run = wandb.init(
        project="assignment_3",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
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
        exploration_final_eps=0.05,
    )

    train(eval_env, model, my_config)