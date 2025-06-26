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

deck_values = [min(rank, 10) for rank in range(1, 14) for _ in range(4)]

card_value = Resource("Card")
luck = Resource("Luck")

# Hand
v_0 = Node(FiringMode.INTERACTIVE, distributions=[Distribution(card_value, deck_values)])
# Deck
v_1 = Node(FiringMode.PASSIVE, distributions=[Distribution(card_value, deck_values)])
# Player draws card
e_0 = ResourceConnection(v_1, v_0, card_value, rate=0)
e_1 = Modifier(v_1, e_0, card_value, card_value, rate=1.0)

m = Machinations.load((
    [v_0, v_1],
    [e_0, e_1],
    [card_value, luck],
))

# ------------------------------------------------------------
# Decorate model BEFORE creating the Gym environment so that the
# deep-copied simulation inherits these properties.
# ------------------------------------------------------------
n_nodes = len(m.nodes)
radius  = 3.2
for i, node in enumerate(m.nodes):
    angle = 2 * math.pi * i / n_nodes
    node.pos  = (radius * math.cos(angle), radius * math.sin(angle))  # type: ignore[attr-defined]
    node.name = f"$V_{{{node.id}}}$"

for conn in m.connections:
    conn.name = f"$E_{{{conn.id}}}$"

for resource in m.resources:
    if resource.name == "Card":
        resource.color = "#feae34"
    else:
        resource.color = "#0f0f0f"

env = MachinationsEnv(m, max_steps=200, record_history=True)

obs, info = env.reset()

for i in range(20):
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

quality = "h"
cmd = ["manim", f"-q{quality}", "render/scene.py", "MachinationsScene"]
proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

try:
    proc.wait()
except KeyboardInterrupt:
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    sys.exit(1)

if quality == "h":
    vlc = subprocess.Popen(["vlc", "media/videos/scene/1080p60/MachinationsScene.mp4"], preexec_fn=os.setsid)
elif quality == "m":
    vlc = subprocess.Popen(["vlc", "media/videos/scene/720p30/MachinationsScene.mp4"], preexec_fn=os.setsid)
else:
    vlc = subprocess.Popen(["vlc", "media/videos/scene/480p15/MachinationsScene.mp4"], preexec_fn=os.setsid)
try:
    vlc.wait()
except KeyboardInterrupt:
    os.killpg(os.getpgid(vlc.pid), signal.SIGINT)
    sys.exit(1)
