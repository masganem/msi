import pickle, pathlib, subprocess, os, signal, sys
import math, random

from machinations import Machinations
from render import Renderer

_model_pkl = pathlib.Path("machinations.pkl")
if not _model_pkl.exists():
    raise FileNotFoundError("machinations.pkl not found. Run test.py first.")

with _model_pkl.open("rb") as _fp:
    m: Machinations = pickle.load(_fp)

r = Renderer(m)
r.render(steps=3)  # adjust steps if desired

n_nodes = len(m.nodes)
radius = 2.0

for i, node in enumerate(m.nodes):
    angle = 2 * math.pi * i / n_nodes
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    node.pos = (x, y)  # type: ignore[attr-defined]
    node.name = "$V_{" + str(i) + "}$"

for i, c in enumerate(m.connections):
    c.name = "$E_{" + str(i) + "}$"

for i, resource in enumerate(m.resources):
    if resource.name == "HP":
        resource.color = "#ff0000"
    elif resource.name == "Mana":
        resource.color = "#0000ff"

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
else:
    vlc = subprocess.Popen(["vlc", "media/videos/scene/480p15/MachinationsScene.mp4"], preexec_fn=os.setsid)
try:
    vlc.wait()
except KeyboardInterrupt:
    os.killpg(os.getpgid(vlc.pid), signal.SIGINT)
    sys.exit(1)
