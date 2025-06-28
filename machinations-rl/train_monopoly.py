from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from monopoly import m, player, estate, money, buy  # Import node and resource references
from machinations.gym_env import MachinationsEnv
import numpy as np
import pathlib
import pickle
import subprocess
import os
import signal
import sys
from gymnasium import Wrapper

# Parse CLI args
TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
RENDER_FREQ = int(sys.argv[2]) if len(sys.argv) > 2 else None
QUALITY = sys.argv[3] if len(sys.argv) > 3 else None

class AnimationCallback(BaseCallback):
    """Custom callback for saving animations of evaluation episodes."""
    
    def __init__(self, eval_env):
        super().__init__()
        self.eval_env = eval_env
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        if RENDER_FREQ is None:  # Skip rendering if not requested
            return True
            
        if self.n_calls % RENDER_FREQ == 0:
            # Run one episode with animation recording
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)  # Allow exploration
                # Ensure action is a numpy array and track it
                action = np.array(action, dtype=np.int8)
                old_estate = self.eval_env.unwrapped._model.X[player.id, estate.id]
                old_money = self.eval_env.unwrapped._model.X[player.id, money.id]
                old_buy_money = self.eval_env.unwrapped._model.X[buy.id, money.id]
                
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                # Debug estate purchases
                new_estate = self.eval_env.unwrapped._model.X[player.id, estate.id]
                new_money = self.eval_env.unwrapped._model.X[player.id, money.id]
                new_buy_money = self.eval_env.unwrapped._model.X[buy.id, money.id]
                
                if action[0] == 1:  # If buy action was chosen
                    print(f"\nAttempted buy with ${old_money:.0f}")
                    print(f"  Buy node money: ${old_buy_money:.0f} -> ${new_buy_money:.0f}")
                if new_estate > old_estate:
                    print(f"\nEstate purchased! Estate: {old_estate:.0f} -> {new_estate:.0f}, Money: ${old_money:.0f} -> ${new_money:.0f}")
            
            # Save and render the animation
            r = info["renderer"]
            _renderer_pkl = pathlib.Path("render") / "renderer.pkl"
            with _renderer_pkl.open("wb") as _fp:
                pickle.dump(r, _fp)
            
            # Run manim to create the animation
            episode_dir = pathlib.Path(f"animations/episode_{self.episode_count}")
            episode_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = ["manim", 
                  f"-q{QUALITY}",  # quality from CLI arg
                  "--fps", "15",  # reduce framerate for faster rendering
                  "--disable_caching",  # faster for one-off renders
                  "--media_dir", str(episode_dir),  # Output directly to episode dir
                  "--format=mp4",  # force mp4 output
                  "--renderer=cairo",  # use cairo renderer which is faster for 2D
                  "render/scene.py", 
                  "MachinationsScene"]
            proc = subprocess.Popen(cmd, 
                                  preexec_fn=os.setsid,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            try:
                proc.wait()
            except:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            
            self.episode_count += 1
        return True

class TrainingMonitorCallback(BaseCallback):
    """Custom callback for monitoring training actions."""
    
    def __init__(self):
        super().__init__()
        self.episode_actions = {0: 0, 1: 0}
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Get the last action from the buffer
        last_action = self.training_env.get_attr('_last_action')[0]
        if last_action is not None:
            self.episode_actions[last_action[0]] += 1
        
        # If episode ended, print stats and reset
        if self.locals.get('dones')[0]:
            print(f"\nTraining episode {self.episode_count} actions: {self.episode_actions}")
            self.episode_actions = {0: 0, 1: 0}
            self.episode_count += 1
        return True

class ActionTrackingEnv(Wrapper):
    """Wrapper to track the last action taken."""
    
    def __init__(self, env):
        super().__init__(env)
        self._last_action = None
    
    def step(self, action):
        self._last_action = action
        return self.env.step(action)
    
    def reset(self, **kwargs):
        self._last_action = None
        return self.env.reset(**kwargs)

# Create and wrap the environment with normalized observations
env = MachinationsEnv(
    m,
    max_steps=200,  # Longer episodes to allow estate investments to pay off
    max_resource_value=10000.0  # Clip and normalize values to this maximum
)
env = ActionTrackingEnv(env)  # Add action tracking
env = Monitor(env)  # Add Monitor wrapper

# Validate the environment
check_env(env)

# Create evaluation environment with rendering enabled if requested
eval_env = MachinationsEnv(
    m,
    max_steps=200,  # Keep consistent with training
    max_resource_value=10000.0,
    record_history=RENDER_FREQ is not None  # Only record history if rendering
)
eval_env = ActionTrackingEnv(eval_env)  # Add action tracking
eval_env = Monitor(eval_env)  # Add Monitor wrapper

# Create the callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/results",
    eval_freq=1000,
    deterministic=False,  # Allow exploration during evaluation
    render=False
)

callbacks = [eval_callback]
if RENDER_FREQ is not None:
    animation_callback = AnimationCallback(eval_env)
    callbacks.append(animation_callback)

# Add training monitor
training_monitor = TrainingMonitorCallback()
callbacks.append(training_monitor)

# Create and train the agent with tuned hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,  # Longer steps to see more long-term effects
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,  # Increased entropy for more exploration
    target_kl=0.03,  # Early stopping on divergence
    verbose=1,
    tensorboard_log="./logs/tensorboard/"
)

# Print action space info
print("\nAction space info:")
print(f"Action space: {env.action_space}")
print(f"Sample actions:")
for _ in range(5):
    print(f"  {env.action_space.sample()}")

# Train the agent
print("\nStarting training...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    progress_bar=True
)

# Save the final model
model.save("monopoly_ppo_final")

# Run some evaluation episodes
print("\nRunning evaluation episodes with detailed resource tracking...")
n_eval_episodes = 10
episode_rewards = []
episode_lengths = []

for episode in range(n_eval_episodes):
    print(f"\nEpisode {episode + 1}:")
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    steps = 0
    max_estate = 0
    action_counts = {0: 0, 1: 0}  # Track distribution of actions
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Ensure action is a numpy array and track it
        action = np.array(action, dtype=np.int8)
        action_counts[action[0]] += 1
        
        # Track resources before step
        old_estate = eval_env.unwrapped._model.X[player.id, estate.id]
        old_money = eval_env.unwrapped._model.X[player.id, money.id]
        old_buy_money = eval_env.unwrapped._model.X[buy.id, money.id]
        
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        
        # Track resources after step
        new_estate = eval_env.unwrapped._model.X[player.id, estate.id]
        new_money = eval_env.unwrapped._model.X[player.id, money.id]
        new_buy_money = eval_env.unwrapped._model.X[buy.id, money.id]
        max_estate = max(max_estate, new_estate)
        
        if action[0] == 1:  # If buy action was chosen
            print(f"  Step {steps}: Attempted buy with ${old_money:.0f}")
            print(f"    Buy node money: ${old_buy_money:.0f} -> ${new_buy_money:.0f}")
        if new_estate > old_estate:
            print(f"  Step {steps}: Estate purchased! Estate: {old_estate:.0f} -> {new_estate:.0f}, Money: ${old_money:.0f} -> ${new_money:.0f}")
        elif action[0] == 1:  # Failed buy attempt
            print(f"    Buy failed! Not enough money?")
        
        total_reward += reward
        steps += 1
    
    print(f"  Episode ended after {steps} steps")
    print(f"  Final money: ${new_money:.0f}")
    print(f"  Max estate owned: {max_estate:.0f}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Actions taken: {action_counts}")  # Show action distribution
    
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

print("\nEvaluation Results:")
print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
print(f"Mean episode length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}") 