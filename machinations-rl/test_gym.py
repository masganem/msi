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
player = Pool(FiringMode.PASSIVE, [(r1, 1500),(r2, 0)])

# Bank is an infinite money pool
bank = Pool(FiringMode.PASSIVE, [(r1, inf)])

# Pass Go means a certain chance of pulling some money from the bank each turn.
# https://chatgpt.com/c/685b525f-f544-8000-8eb9-6c1e2438efa0 <- specific odds
# basically 7/40
pass_go_r = ResourceConnection(bank, player, r1, 200.0)
d40 = Gate(
    FiringMode.AUTOMATIC,
    DistributionMode.NONDETERMINISTIC,
    [*range(1, 20)],
    luck,
)
pass_go_t = Trigger(d40, pass_go_r, Predicate("<", 7))

# Broken if no interactive nodes..
# dummy_interactive = Pool(FiringMode.INTERACTIVE, [])

# Paying rent: here, we'll model a probability distribution using
# a nondeterministic gate. We'll take its X and add to the rate
# of the "player-pays-rent" resource connection.
player_pays_rent = ResourceConnection(player, bank, r1, .0)
other_d40 = Gate(FiringMode.AUTOMATIC, DistributionMode.NONDETERMINISTIC, [*range(2, 52, 2), *([25]*4), *([28]*2)], r1)
rent_label_modifier = LabelModifier(other_d40, player_pays_rent, r1, 1)
another_d40 = Gate(FiringMode.AUTOMATIC, DistributionMode.NONDETERMINISTIC, [*range(1, 40)], luck)
t_player_pays_rent = Trigger(another_d40, player_pays_rent, Predicate("<", 28))

# Buying estate: implementation of the converter function as described in 
# Machinations: Interactive node pulls, triggers a pull from source to node
# of desired resource type.
estate = Pool(FiringMode.PASSIVE, [(r1, inf),(r2, inf)])
player_gets_estate = ResourceConnection(estate, player, r2, 1)
buy_estate = Pool(FiringMode.INTERACTIVE, [])
player_pays_for_estate = ResourceConnection(player, buy_estate, r1, 100)
t_player_gets_estate = Trigger(buy_estate, player_gets_estate)

# Receiving rent: every turn, there's a 7/40 chance of getting paid rent.
# How much _is_ a variable, but one that gets influenced by how much property
# the player has.
yet_another_d40 = Gate(FiringMode.AUTOMATIC, DistributionMode.NONDETERMINISTIC, [*range(1, 40)], luck)
player_gets_rent = ResourceConnection(estate, player, r1, .0)
income_label_modifier = LabelModifier(player, player_gets_rent, r2, 20)
t_player_gets_rent = Trigger(yet_another_d40, player_gets_rent, Predicate("<", 7))


m = Machinations.load((
    [player, bank, d40, other_d40, another_d40, estate, buy_estate, yet_another_d40],
    [pass_go_r, pass_go_t, player_pays_rent, rent_label_modifier, t_player_pays_rent, player_gets_estate, player_pays_for_estate, t_player_gets_estate, player_gets_rent, income_label_modifier, t_player_gets_rent],
    [r1, r2, luck],
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
    if resource.name == "Money":
        resource.color = "#3ef948"
    elif resource.name == "Property":
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
