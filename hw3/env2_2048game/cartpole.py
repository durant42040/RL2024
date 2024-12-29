import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="assignment_3",
    name="cartpole_dqn",
    sync_tensorboard=True,
    id="cartpole_ppo",
    monitor_gym=True,
)

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    model_save_path="./models/",
    model_save_freq=5000,
    verbose=2,
)

model.learn(total_timesteps=100000, callback=wandb_callback)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})

model.save("dqn_cartpole_model")
env.close()

def eval():
    env = gym.make("CartPole-v1", render_mode="human")
    model = PPO.load("dqn_cartpole_model")

    obs, info = env.reset()
    for _ in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, truncate, info = env.step(action)
        env.render()
        if dones:
            break

    env.close()

# eval()
wandb.finish()
