from machinations.gym_env import MachinationsEnv
from machinations import Machinations
from machinations.definitions import *
from math import inf
import time
import math
import pathlib
import pickle
import os
import subprocess
import signal
import sys

r1 = Resource("Money")
r2 = Resource("Land")
luck = Resource("Luck")

# Player Money and Land
player = Pool(FiringMode.PASSIVE, [(r1, 100),(r2, 0)])

# Bank is an infinite money pool
bank = Pool(FiringMode.PASSIVE, [(r1, inf)])

# Pass Go means a certain chance of pulling some money from the bank each turn.
# https://chatgpt.com/c/685b525f-f544-8000-8eb9-6c1e2438efa0 <- specific odds
# basically 7/40
pass_go_r = ResourceConnection(bank, player, r1, 200.0)
d40 = Gate(FiringMode.AUTOMATIC, DistributionMode.NONDETERMINISTIC, 40, luck)
pass_go_t = Trigger(d40, pass_go_r, Predicate("<", 7))

# Broken if no interactive nodes..
dummy_interactive = Pool(FiringMode.INTERACTIVE, [])

m = Machinations.load((
    [player, bank, d40, dummy_interactive],
    [pass_go_r, pass_go_t],
    [r1, luck],
))

# ------------------------------------------------------------
# Decorate model BEFORE creating the Gym environment so that the
# deep-copied simulation inherits these properties.
# ------------------------------------------------------------
n_nodes = len(m.nodes)
radius  = 3.0
for i, node in enumerate(m.nodes):
    angle = 2 * math.pi * i / n_nodes
    node.pos  = (radius * math.cos(angle), radius * math.sin(angle))  # type: ignore[attr-defined]
    node.name = f"$V_{{{node.id}}}$"

for conn in m.connections:
    conn.name = f"$E_{{{conn.id}}}$"

for resource in m.resources:
    if resource.name == "Money":
        resource.color = "#3ef948"
    elif resource.name == "Property":
        resource.color = "#feae34"
    else:
        resource.color = "#0f0f0f"

env = MachinationsEnv(m, max_steps=200, record_history=True)

obs, info = env.reset()

for i in range(10):
    start_time = time.time()
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    end_time = time.time()
    iteration_time = end_time - start_time
    
    print("=" * 50)
    print(f"Iteration {i+1}/200")
    print(f"Time taken: {iteration_time} seconds")
    print(f"Action: {action}")
    # print(f"Reward: {reward}")
    # print(f"Terminated: {terminated}")
    # print(f"Truncated: {truncated}")
    
    print(f"Observation: {obs}")
    print("=" * 50)

# Retrieve renderer and persist it for Manim playback
r = info["renderer"]

_renderer_pkl = pathlib.Path("render") / "renderer.pkl"
with _renderer_pkl.open("wb") as _fp:
    pickle.dump(r, _fp)

quality = "l"
cmd = ["manim", f"-q{quality}", "render/scene.py", "MachinationsScene"]
proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

try:
    proc.wait()
except KeyboardInterrupt:
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    sys.exit(1)

if quality == "h":
    vlc = subprocess.Popen(["vlc", "media/videos/scene/1080p60/MachinationsScene.mp4"], preexec_fn=os.setsid)
else:
    vlc = subprocess.Popen(["vlc", "media/videos/scene/480p15/MachinationsScene.mp4"], preexec_fn=os.setsid)
try:
    vlc.wait()
except KeyboardInterrupt:
    os.killpg(os.getpgid(vlc.pid), signal.SIGINT)
    sys.exit(1)
