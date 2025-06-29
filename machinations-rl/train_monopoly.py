from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from monopoly import m, player, estate, money, buy, world, bank  # Import node and resource references
from machinations.gym_env import MachinationsEnv
import numpy as np
import pathlib
import pickle
import subprocess
import os
import signal
import sys
from gymnasium import Wrapper
import matplotlib.pyplot as plt

# Parse CLI args
TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200000  # Longer training for highly stochastic environment
RENDER_FREQ = int(sys.argv[2]) if len(sys.argv) > 2 else None
QUALITY = sys.argv[3] if len(sys.argv) > 3 else None
CONTROL_MODE = sys.argv[4].lower() == "control" if len(sys.argv) > 4 else False
NEVER_BUY_MODE = len(sys.argv) > 5 and sys.argv[5] == "0" and CONTROL_MODE

# Simple tracking arrays for evaluation progress
eval_timesteps = []
eval_episode_lengths = []

# ============================================================================
# PPO TRAINING STRATEGY for Highly Stochastic Environment
# ============================================================================
# This monopoly environment is extremely nondeterministic:
# - 70% chance to pay rent each step (major negative event)
# - Rent varies 25x ($2-50) creating high variance
# - Binary bankruptcy outcome (-1 reward) vs survival (+1 reward)
# 
# WHY PPO over DQN:
# 1. HANDLES HIGH VARIANCE: Policy gradients work better with stochastic rewards
# 2. SPARSE REWARDS: PPO's advantage estimation (GAE) helps with binary outcomes
# 3. EXPLORATION: Natural policy entropy maintains exploration in uncertain env
# 4. SAMPLE EFFICIENCY: Multiple epochs per batch compensate for short episodes
#
# PPO Hyperparameter Rationale for HIGHLY Stochastic Environment:
# - n_steps=4096: More diverse experiences to find signal in 70% rent chaos
# - n_epochs=15: Extract maximum learning from each noisy batch
# - ent_coef=0.05: HIGH exploration to discover estate-building strategies
# - gamma=0.98: Value future estate income more (need 4+ estates to survive)
# - Higher LR (0.0005): Faster adaptation to brutal stochastic punishments
# - clip_range=0.3: Allow bigger policy updates for strategy discovery
# ============================================================================

if CONTROL_MODE:
    if NEVER_BUY_MODE:
        print("\nRunning in CONTROL_0 mode - never buying (always action 0) instead of trained model")
    else:
        print("\nRunning in CONTROL mode - using random actions instead of trained model")

# Specify weights for resources at different nodes
player.weights = {
    estate.id: 0.0,  # Value estates highly
    money.id: 0.0001   # Value money moderately
}

# Make only relevant nodes visible to the agent
player.visible = True  # Agent can see its own resources
world.visible = True   # Agent can see world state
bank.visible = False   # Agent doesn't need to see bank's infinite resources

# Weight rent-generating connections and make them visible
for conn in m.resource_connections:
    if conn.resource_type == money and conn.dst.id == player.id:  # Money flowing to a node
        conn.weight = 0.0  # Value rent income rate
        conn.visible = True  # Agent can see income rates
    elif conn.resource_type == money and conn.src.id == player.id:  # Money flowing from player
        conn.visible = True  # Agent can see expense rates
    else:
        conn.visible = False  # Hide other connections

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
            # Enable history recording just for this episode
            self.eval_env.unwrapped._record_history = True
            
            # Run one episode with animation recording
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)  # Allow exploration
                # Ensure action is a numpy array and track it
                action = np.array(action, dtype=np.int8)
                
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
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
            
            # Disable history recording after the episode
            self.eval_env.unwrapped._record_history = False
            
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
            # For discrete actions, last_action is now an integer, not an array
            self.episode_actions[last_action] += 1
        
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

class EarlyStoppingCallback(BaseCallback):
    """Early stopping callback to prevent catastrophic forgetting."""
    
    def __init__(self, patience=10000, min_reward_threshold=0.6, check_freq=1000):
        super().__init__()
        self.patience = patience
        self.min_reward_threshold = min_reward_threshold
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.wait = 0
        
    def _on_step(self) -> bool:
        # Only check every check_freq steps
        if self.n_calls % self.check_freq != 0:
            return True
            
        # Get recent episode rewards from the monitor
        if hasattr(self.training_env, 'get_episode_rewards'):
            episode_rewards = self.training_env.get_episode_rewards()
            if len(episode_rewards) >= 10:  # Need at least 10 episodes
                current_mean_reward = np.mean(episode_rewards[-10:])  # Last 10 episodes
                
                # If performance is above threshold and improving, reset patience
                if current_mean_reward > self.best_mean_reward and current_mean_reward > self.min_reward_threshold:
                    self.best_mean_reward = current_mean_reward
                    self.wait = 0
                    print(f"New best mean reward: {self.best_mean_reward:.3f} at step {self.num_timesteps}")
                else:
                    self.wait += self.check_freq
                    
                # Stop if performance has been stagnant/declining for too long
                if self.wait >= self.patience and self.best_mean_reward > self.min_reward_threshold:
                    print(f"\nEarly stopping at timestep {self.num_timesteps}")
                    print(f"Best mean reward achieved: {self.best_mean_reward:.3f}")
                    print("Stopping to prevent catastrophic forgetting.")
                    return False
                    
        return True

class ConvergenceDetectionCallback(BaseCallback):
    """Detect if agent has converged to a simple strategy."""
    
    def __init__(self, check_freq=2000, patience=10000, action_threshold=0.85):
        super().__init__()
        self.check_freq = check_freq
        self.patience = patience  
        self.action_threshold = action_threshold
        self.action_history = []
        self.convergence_start = None
        
    def _on_step(self) -> bool:
        # Only check every check_freq steps
        if self.n_calls % self.check_freq != 0:
            return True
            
        # Get recent episode actions from training monitor
        if hasattr(self.parent, 'training_monitor'):
            monitor = self.parent.training_monitor
            if monitor.episode_count >= 10:  # Need some data
                # Check if agent is consistently choosing one action
                recent_actions = []
                for episode_data in list(monitor.action_history)[-10:]:  # Last 10 episodes
                    if episode_data['total_actions'] > 0:
                        buy_ratio = episode_data['buy_actions'] / episode_data['total_actions']
                        recent_actions.append(buy_ratio)
                
                if len(recent_actions) >= 5:
                    avg_buy_ratio = np.mean(recent_actions)
                    
                    # Check for convergence to always buy or always hold
                    converged = (avg_buy_ratio > self.action_threshold or 
                               avg_buy_ratio < (1 - self.action_threshold))
                    
                    if converged:
                        if self.convergence_start is None:
                            self.convergence_start = self.num_timesteps
                            strategy = "ALWAYS BUY" if avg_buy_ratio > 0.5 else "ALWAYS HOLD"
                            print(f"\n⚠️  Potential convergence detected at step {self.num_timesteps}")
                            print(f"   Strategy: {strategy} ({avg_buy_ratio:.1%} buy rate)")
                            print(f"   Monitoring for {self.patience} more steps...")
                        
                        # If converged for too long, suggest stopping
                        elif self.num_timesteps - self.convergence_start > self.patience:
                            strategy = "ALWAYS BUY" if avg_buy_ratio > 0.5 else "ALWAYS HOLD"
                            print(f"\n🛑 Agent has converged to {strategy} for {self.patience} steps")
                            print(f"   Consider stopping training or adjusting hyperparameters")
                            print(f"   Current performance: {avg_buy_ratio:.1%} buy rate")
                            # Don't force stop - let user decide
                    else:
                        self.convergence_start = None  # Reset if not converged anymore
                        
        return True

class EnhancedTrainingMonitorCallback(BaseCallback):
    """Enhanced monitoring with action history tracking."""
    
    def __init__(self):
        super().__init__()
        self.episode_actions = {0: 0, 1: 0}
        self.episode_count = 0
        self.action_history = []
        
    def _on_step(self) -> bool:
        # Get the last action from the buffer
        last_action = self.training_env.get_attr('_last_action')[0]
        if last_action is not None:
            self.episode_actions[last_action] += 1
        
        # If episode ended, record stats and reset
        if self.locals.get('dones')[0]:
            total_actions = sum(self.episode_actions.values())
            episode_data = {
                'episode': self.episode_count,
                'buy_actions': self.episode_actions[1],
                'hold_actions': self.episode_actions[0], 
                'total_actions': total_actions,
                'timestep': self.num_timesteps
            }
            self.action_history.append(episode_data)
            
            # Keep only last 50 episodes to avoid memory issues
            if len(self.action_history) > 50:
                self.action_history.pop(0)
            
            # Print periodic summaries
            if self.episode_count % 50 == 0 and self.episode_count > 0:
                recent_data = self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history
                if recent_data:
                    avg_buy_rate = np.mean([ep['buy_actions'] / max(ep['total_actions'], 1) 
                                          for ep in recent_data])
                    print(f"\nTraining Episode {self.episode_count}")
                    print(f"  Recent buy rate: {avg_buy_rate:.1%}")
                    print(f"  Episode actions: {self.episode_actions}")
            
            self.episode_actions = {0: 0, 1: 0}
            self.episode_count += 1
        return True

class SimpleEvalTracker(EvalCallback):
    """Simple callback to track evaluation episode lengths."""
    
    def __init__(self, eval_env, **kwargs):
        super().__init__(eval_env, **kwargs)
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # If we just finished an evaluation, record the results
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if hasattr(self, 'last_mean_reward'):
                eval_timesteps.append(self.num_timesteps)
                # Calculate mean episode length from the last evaluation
                if len(self.evaluations_length) > 0:
                    mean_ep_length = np.mean(self.evaluations_length[-1])
                    eval_episode_lengths.append(mean_ep_length)
                    print(f"📊 Eval at {self.num_timesteps}: Episode length = {mean_ep_length:.1f}")
        
        return result

# Create and wrap the environment with normalized observations
env = MachinationsEnv(
    m,
    max_steps=30,  # Longer episodes to allow estate investments to pay off
    max_resource_value=100.0,    # Tighter scaling: $50 starting + $50 buffer
    record_history=False  # Don't need history for training
)
env = ActionTrackingEnv(env)  # Add action tracking
env = Monitor(env)  # Add Monitor wrapper

# Validate the environment
check_env(env)

# Create evaluation environment with rendering enabled if requested
eval_env = MachinationsEnv(
    m,
    max_steps=30,  # Keep consistent with training
    max_resource_value=100.0,    # Match training environment scaling
    record_history=RENDER_FREQ is not None  # Only record history if we're going to render
)
eval_env = ActionTrackingEnv(eval_env)  # Add action tracking
eval_env = Monitor(eval_env)  # Add Monitor wrapper

# Create simple evaluation tracker (combines evaluation + tracking)
eval_tracker = SimpleEvalTracker(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/results",
    eval_freq=2500,
    n_eval_episodes=50,  # Good balance: reliable signal but not too slow
    deterministic=True,
    render=False
)

callbacks = [eval_tracker]

if RENDER_FREQ is not None:
    animation_callback = AnimationCallback(eval_env)
    callbacks.append(animation_callback)

# Create and train the agent with PPO hyperparameters tuned for HIGHLY stochastic environment
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256, 128]),  # Bigger network for complex patterns
    learning_rate=0.0005,            # Higher LR for faster adaptation to chaos
    n_steps=4096,                    # More diverse experiences per update
    batch_size=128,                  # Larger batches for more stable gradients
    n_epochs=15,                     # More epochs to extract signal from noise
    gamma=0.98,                      # Higher gamma to value future estate income
    gae_lambda=0.95,                 # GAE for better advantage estimation in stochastic env
    clip_range=0.3,                  # Slightly higher clipping for exploration
    ent_coef=0.05,                   # MUCH higher exploration for strategy discovery
    vf_coef=0.5,                     # Balance policy and value learning
    max_grad_norm=1.0,               # Higher grad norm for faster learning
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
if not CONTROL_MODE:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True
    )

    # Save the final model
    model.save("monopoly_ppo_final")

# Plot the evaluation progress with moving average
if len(eval_timesteps) > 1:
    plt.figure(figsize=(10, 6))
    
    # Calculate moving average (window size 3)
    window_size = min(3, len(eval_episode_lengths))
    if len(eval_episode_lengths) >= window_size:
        moving_avg = []
        for i in range(len(eval_episode_lengths)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            moving_avg.append(np.mean(eval_episode_lengths[start_idx:end_idx]))
    else:
        moving_avg = eval_episode_lengths
    
    # Plot raw points (lighter) and moving average (prominent)
    plt.plot(eval_timesteps, eval_episode_lengths, 'o-', color='lightblue', alpha=0.5, markersize=4, linewidth=1, label='Raw Evaluations')
    plt.plot(eval_timesteps, moving_avg, 'bo-', linewidth=2, markersize=4, label=f'Moving Average (window={window_size})')
    
    plt.xlabel('Training Timesteps')
    plt.ylabel('Evaluation Episode Length')
    plt.title('Agent Learning Progress: Episode Length vs Training Steps (Moving Average)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=30, color='r', linestyle='--', alpha=0.7, label='Max Episode Length (30)')
    plt.legend(loc='upper right')
    plt.savefig('eval_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Evaluation progress plot saved: eval_progress.png (moving average with window {window_size})")
else:
    print("⚠️  Not enough evaluation data for plotting")

# Run some evaluation episodes
print("\nRunning evaluation episodes with detailed resource tracking...")
n_eval_episodes = 1000
episode_rewards = []
episode_lengths = []
survival_data = []  # Track episode lengths for survival analysis

if CONTROL_MODE:
    if NEVER_BUY_MODE:
        print("\nRunning in CONTROL_0 mode - never buying (always action 0) instead of trained model")
    else:
        print("\nRunning in CONTROL mode - using random actions instead of trained model")
    for episode in range(n_eval_episodes):
        print(f"\nEpisode {episode + 1}:")
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_estate = 0
        action_counts = {0: 0, 1: 0}  # Track distribution of actions
        
        while not done:
            if NEVER_BUY_MODE:
                action = 0  # Never buy (always action 0)
            else:
                action = eval_env.action_space.sample()  # Use random actions
            action_counts[action] += 1
            
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            
            # Track resources after step
            new_estate = eval_env.unwrapped._model.X[player.id, estate.id]
            new_money = eval_env.unwrapped._model.X[player.id, money.id]
            max_estate = max(max_estate, new_estate)
            
            # If we went bankrupt, override total reward with the loss penalty
            if terminated and new_money <= 0:
                total_reward = reward  # Just take the loss penalty
            else:
                total_reward += reward  # Otherwise accumulate normally
            
            steps += 1
        
        print(f"  Episode ended after {steps} steps")
        print(f"  Final money: ${new_money:.0f}")
        print(f"  Max estate owned: {max_estate:.0f}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Actions taken: {action_counts}")  # Show action distribution
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        survival_data.append(steps)
    
    if NEVER_BUY_MODE:
        print("\nControl_0 Mode Evaluation Results:")
    else:
        print("\nControl Mode Evaluation Results:")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
else:
    for episode in range(n_eval_episodes):
        print(f"\nEpisode {episode + 1}:")
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_estate = 0
        action_counts = {0: 0, 1: 0}  # Track distribution of actions
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)  # Use stochastic policy for stochastic environment
            # For discrete actions, action is now an integer, not an array
            action = int(action)
            action_counts[action] += 1
            
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            # print(f"Observation: {obs}")
            done = terminated or truncated
            
            # Track resources after step
            new_estate = eval_env.unwrapped._model.X[player.id, estate.id]
            new_money = eval_env.unwrapped._model.X[player.id, money.id]
            max_estate = max(max_estate, new_estate)
            
            # If we went bankrupt, override total reward with the loss penalty
            if terminated and new_money <= 0:
                total_reward = reward  # Just take the loss penalty
            else:
                total_reward += reward  # Otherwise accumulate normally
            
            steps += 1
        
        print(f"  Episode ended after {steps} steps")
        print(f"  Final money: ${new_money:.0f}")
        print(f"  Max estate owned: {max_estate:.0f}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Actions taken: {action_counts}")  # Show action distribution
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        survival_data.append(steps)

    print("\nEvaluation Results:")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")

# Plot episode lengths over episodes (sample every 10th for clarity) - MOVED OUTSIDE IF/ELSE
plt.figure(figsize=(10, 6))
episode_numbers = range(1, len(survival_data) + 1)

# Sample every 10th episode for dots
sample_indices = range(0, len(survival_data), 10)
sample_episodes = [episode_numbers[i] for i in sample_indices]
sample_lengths = [survival_data[i] for i in sample_indices]

plt.plot(episode_numbers, survival_data, 'b-', linewidth=1, alpha=0.3)
plt.scatter(sample_episodes, sample_lengths, 
            c=['red' if length < 30 else 'green' for length in sample_lengths], 
            s=20, alpha=0.8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Episode Number')
plt.ylabel('Episode Length (Steps)')
plt.title('Episode Length per Episode (Red=Died, Green=Survived, Every 10th Episode Shown)')
plt.axhline(y=30, color='black', linestyle='--', alpha=0.7, label='Max Episode Length (30)')

# For PPO, show best achieved performance; for control modes, show mean
if not CONTROL_MODE and len(eval_episode_lengths) > 0:
    best_mean_length = max(eval_episode_lengths)
    plt.axhline(y=best_mean_length, color='orange', linestyle='-', alpha=0.8, label=f'Best Mean Episode Length ({best_mean_length:.1f})')
    print(f"📊 Best mean episode length achieved during training: {best_mean_length:.1f}")
else:
    mean_length = np.mean(survival_data)
    plt.axhline(y=mean_length, color='orange', linestyle='-', alpha=0.8, label=f'Mean Episode Length ({mean_length:.1f})')
    print(f"📊 Mean episode length: {mean_length:.1f}")

plt.legend(loc='upper right')

if NEVER_BUY_MODE:
    filename = 'episode_lengths_control_0.png'
elif CONTROL_MODE:
    filename = 'episode_lengths_control.png'
else:
    filename = 'episode_lengths_ppo.png'
plt.savefig(filename)
plt.close() 