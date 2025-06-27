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

# deck_values = [min(rank, 10) for rank in range(1, 13) for _ in range(4)]
deck_values = [min(rank, 10) for rank in range(1, 6) for _ in range(4)]

card_value = Resource("Card")
ace = Resource("Ace")
turn = Resource("Turn")

# Hand
v_0 = Node(FiringMode.PASSIVE, distributions=[Distribution(card_value, deck_values, infinite=False), Distribution(ace, [0]), Distribution(turn, [1])])
# Deck (now passive; draws are pulled via triggers)
v_1 = Node(FiringMode.NONDETERMINISTIC, distributions=[Distribution(card_value, deck_values, infinite=False)])

# Player draws card
e_0 = ResourceConnection(v_1, v_0, card_value, rate=0)
e_1 = Modifier(v_1, e_0, card_value, card_value, rate=1.0)

# Ace
v_2 = Node(FiringMode.PASSIVE, initial_resources=[(ace, inf)])
e_2 = Trigger(v_1, v_2, card_value, Predicate("==", 1))
# Player gets ace
e_3 = ResourceConnection(v_2, v_0, ace, rate=1)
e_16 = Activator(v_0, e_3, turn, Predicate("==", 1)) 

# Lose condition: hand total > 21
lose_node = OutcomeNode("lose")
tie_node = OutcomeNode("tie")
win_node = OutcomeNode("win")
e_4 = Trigger(v_0, lose_node, card_value, Predicate(">", 21))

# Dealer
v_3 = Node(FiringMode.PASSIVE, distributions=[Distribution(card_value, deck_values, infinite=False), Distribution(ace, [0]), Distribution(turn, [0])])
e_17 = Trigger(v_3, win_node, card_value, Predicate(">", 21))
# Dealer draws card
e_5 = ResourceConnection(v_1, v_3, card_value, rate=0)
e_6 = Modifier(v_1, e_5, card_value, card_value, rate=1.0)
# Dealer gets ace
e_7 = ResourceConnection(v_2, v_3, ace, rate=1)
e_15 = Activator(v_3, e_7, turn, Predicate("==", 1)) 

# Turn logic – use triggers to pull from deck when `turn == 1`
e_8 = Trigger(v_0, e_0, turn, Predicate("==", 1))
e_9 = Trigger(v_3, e_5, turn, Predicate("==", 1))

# Stand
v_4 = Node(FiringMode.INTERACTIVE)
e_10 = ResourceConnection(v_0, v_3, turn, rate=1)
e_11 = Trigger(v_4, e_10)

# Win check
v_5 = Node(FiringMode.PASSIVE, initial_resources=[(card_value, 0)])
e_13 = Modifier(v_0, v_5, card_value, card_value, 1)
e_14 = Modifier(v_3, v_5, card_value, card_value, -1)
e_26 = Trigger(v_0, win_node, card_value, Predicate("==", 21))
e_27 = Trigger(v_3, lose_node, card_value, Predicate("==", 21))

e_28 = Trigger(v_5, win_node)
e_29 = Trigger(v_5, lose_node)
e_30 = Trigger(v_5, tie_node)
e_31 = Activator(v_5, win_node, card_value, Predicate(">", 0))
e_32 = Activator(v_5, tie_node, card_value, Predicate("==", 0))
e_33 = Activator(v_5, lose_node, card_value, Predicate("<", 0))

# Ace bonus
v_6 = Node(FiringMode.PASSIVE, distributions=[Distribution(card_value, [inf])])
e_18 = ResourceConnection(v_6, v_0, card_value, 10)
e_21 = Activator(v_0, e_18, ace, Predicate(">=", 1))
e_23 = Activator(v_0, e_18, card_value, Predicate("<=", 12))
e_19 = ResourceConnection(v_6, v_3, card_value, 10)
e_22 = Activator(v_3, e_19, ace, Predicate(">=", 1))
e_24 = Activator(v_3, e_19, card_value, Predicate("<=", 12))
e_20 = Trigger(v_3, v_6, card_value, Predicate(">=", 17))
e_12 = Trigger(v_6, v_5)

# player can only stand if cards >= 12
e_25 = Activator(v_0, v_4, card_value, Predicate(">=", 12))

m = Machinations.load((
    [v_0, v_1, v_2, v_3, v_4, v_5, v_6, lose_node, tie_node, win_node],
    [e_0, e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9, e_10, e_11, e_12, e_13, e_14, e_15, e_16, e_17, e_18, e_19, e_20, e_21, e_22, e_23, e_24, e_25, e_26, e_27, e_28, e_29, e_30, e_31, e_32, e_33],
    [card_value, ace, turn],
))

# ------------------------------------------------------------
# Positioning: v_0 at LEFT, v_3 at RIGHT; all others on a circle.
# ------------------------------------------------------------
radius = 3.4

# Explicit positions for player (v_0) and dealer (v_3)
v_0.pos = (0, 1.25)  # type: ignore[attr-defined]
v_1.pos = (3, 1.25)
v_2.pos = (3, -1.25)
v_3.pos = (0, -1.25)
v_4.pos = (1.45, 0)
v_5.pos = (-3, 1.25)
v_6.pos = (-3, -1.25)
lose_node.pos = (-2.5, 2.5)
tie_node.pos = (-3.75, 0)
win_node.pos = (-2.5, -2.5)

# Remaining nodes to place around the circle (exclude v_0 and v_3)
other_nodes = [n for n in m.nodes if not hasattr(n, "pos")]
n_rem = len(other_nodes)
for idx, node in enumerate(other_nodes):
    angle = 2 * math.pi * idx / n_rem
    node.pos = (radius * math.cos(angle), radius * math.sin(angle))  # type: ignore[attr-defined]

# ------------------------------------------------------------
# Assign display names and aliases for legend
# ------------------------------------------------------------
alias_map = {
    v_0.id: "Player",
    v_1.id: "Deck",
    v_2.id: "Ace",
    v_3.id: "Dealer",
    v_4.id: "Stand",
    v_5.id: "Check",
}

for node in m.nodes:
    if isinstance(node, OutcomeNode):
        node.name = f"$\\text{{{node.outcome.upper()}}}$"
    else:
        node.name = f"$V_{{{node.id}}}$"

    if node.id in alias_map:
        setattr(node, "alias", alias_map[node.id])

for conn in m.connections:
    conn.name = f"$E_{{{conn.id}}}$"

for resource in m.resources:
    if resource.name == "Card":
        resource.color = "#fe4822"
    elif resource.name == "Turn":
        resource.color = "#3333fe"
    elif resource.name == "Ace":
        resource.color = "#33fe33"
    else:
        resource.color = "#0f0f0f"

env = MachinationsEnv(m, max_steps=200, record_history=True)

obs, info = env.reset()

# ------------------------------------------------------------
# CLI arguments: 1st = number of iterations, 2nd = quality (l/m/h)
# ------------------------------------------------------------
steps = 10
if len(sys.argv) > 1:
    try:
        steps = int(sys.argv[1])
    except ValueError:
        print(f"[warn] Invalid episode count '{sys.argv[1]}', using 10.")

for i in range(steps):
    start_time = time.time()
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    end_time = time.time()
    iteration_time = end_time - start_time
    
    #print("=" * 50)
    #print(f"Iteration {i+1}/{steps}")
    #print(f"Time taken: {iteration_time} seconds")
    #print(f"Action: {action}")
    # print(f"Reward: {reward}")
    # print(f"Terminated: {terminated}")
    # print(f"Truncated: {truncated}")
    #print(f"Observation: {obs}")
    # -- Custom debug output: card totals --
    model = env._model  # current Machinations simulation
    print("Player(v0) | Dealer(v3) | Score(v5) | v0-v3")
    if model is not None:
        v0_cards = model.X[v_0.id, card_value.id]
        v3_cards = model.X[v_3.id, card_value.id]
        v5_cards = model.X[v_5.id, card_value.id]
        print(f"{v0_cards:11.1f} | {v3_cards:11.1f} | {v5_cards:10.1f} | {v0_cards - v3_cards:7.1f}")
    # print("=" * 50)

# Retrieve renderer and persist it for Manim playback
r = info["renderer"]

_renderer_pkl = pathlib.Path("render") / "renderer.pkl"
with _renderer_pkl.open("wb") as _fp:
    pickle.dump(r, _fp)

# --- Video quality ---------------------------------------------------
# Default quality = medium; can override with SECOND CLI arg (l/m/h)
quality = "m"
if len(sys.argv) > 2:
    q = sys.argv[2].lower()
    if q in ("l", "m", "h"):
        quality = q
    else:
        print(f"[warn] Unknown quality '{sys.argv[2]}', using 'm'.")

cmd = ["manim", f"-q{quality}", "render/scene.py", "MachinationsScene"]
# Suppress Manim's stdout/stderr to keep the console quiet
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
