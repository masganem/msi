from monopoly import *
import sys
import subprocess
import pathlib
import pickle
import os
import signal

# =============== #
# RENDERING SETUP #
# =============== #
if len(sys.argv) > 2:
    bank.pos = (-3, 0)
    world.pos = (3, 0)
    buy.pos = (0, -3)
    pass_go_odds.pos = (-1.5, 2)
    pay_rent_odds.pos = (1.5, -2)
    get_rent_odds.pos = (1.5, 2)
    random_rent_value1.pos = (3, -2)

    alias_map = {
        player.id: "Player",
        world.id: "World",
        bank.id: "Bank",
    }

    for node in m.nodes:
        if isinstance(node, OutcomeNode):
            node.name = f"$\\text{{{node.outcome.upper()}}}$"
        else:
            node.name = f"$V_{{{node.id}}}$"

        if node.id in alias_map:
            setattr(node, "alias", alias_map[node.id])
    buy.name = "$\\text{BUY}$"

    for conn in m.connections:
        conn.name = f"$E_{{{conn.id}}}$"

    for resource in m.resources:
        if resource.name == "Money":
            resource.color = "#22fe22"
        elif resource.name == "Estate":
            resource.color = "#882222"
        else:
            resource.color = "#0f0f0f"

env = MachinationsEnv(m, max_steps=200, record_history=True)

obs, info = env.reset()

steps = 10
if len(sys.argv) > 1:
    try:
        steps = int(sys.argv[1])
    except ValueError:
        print(f"[warn] Invalid episode count '{sys.argv[1]}', using 10.")

for i in range(steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

# ================ #
# BEGIN RENDERING  #
# ================ #

if len(sys.argv) > 2:


    # Retrieve renderer and persist it for Manim playback
    r = info["renderer"]
    _renderer_pkl = pathlib.Path("render") / "renderer.pkl"
    with _renderer_pkl.open("wb") as _fp:
        pickle.dump(r, _fp)
    # --- Video quality ---------------------------------------------------
    # Default quality = medium; can override with SECOND CLI arg (l/m/h)
    quality = "m"
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
