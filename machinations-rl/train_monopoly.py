from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from monopoly import m  # Import the Machinations model
from machinations.gym_env import MachinationsEnv
import numpy as np
import pathlib
import pickle
import subprocess
import os
import signal

class AnimationCallback(BaseCallback):
    """Custom callback for saving animations of evaluation episodes."""
    
    def __init__(self, eval_env, freq=10000):
        super().__init__()
        self.eval_env = eval_env
        self.freq = freq
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            # Run one episode with animation recording
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            
            # Save and render the animation
            r = info["renderer"]
            _renderer_pkl = pathlib.Path("render") / "renderer.pkl"
            with _renderer_pkl.open("wb") as _fp:
                pickle.dump(r, _fp)
            
            # Run manim to create the animation
            quality = "m"  # medium quality
            episode_dir = pathlib.Path(f"animations/episode_{self.episode_count}")
            episode_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = ["manim", f"-q{quality}", "render/scene.py", "MachinationsScene"]
            proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
            try:
                proc.wait()
                # Move the generated video to our episode directory
                if quality == "m":
                    video_path = "media/videos/scene/720p30/MachinationsScene.mp4"
                    if os.path.exists(video_path):
                        os.rename(video_path, str(episode_dir / "animation.mp4"))
            except:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            
            self.episode_count += 1
        return True

# Create and wrap the environment with normalized observations
env = MachinationsEnv(
    m,
    max_steps=100,
    max_resource_value=10000.0  # Clip and normalize values to this maximum
)

# Validate the environment
check_env(env)

# Create evaluation environment with rendering enabled
eval_env = MachinationsEnv(
    m,
    max_steps=100,
    max_resource_value=10000.0,
    record_history=True  # Enable recording for animations
)

# Create the callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/results",
    eval_freq=1000,
    deterministic=True,
    render=False
)

animation_callback = AnimationCallback(eval_env, freq=10000)

# Create and train the agent with tuned hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./logs/tensorboard/"
)

# Train the agent
TOTAL_TIMESTEPS = 50000
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, animation_callback],
    progress_bar=True
)

# Save the final model
model.save("monopoly_ppo_final")

# Run some evaluation episodes
n_eval_episodes = 10
episode_rewards = []
episode_lengths = []

for _ in range(n_eval_episodes):
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

print("\nEvaluation Results:")
print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
print(f"Mean episode length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}") 