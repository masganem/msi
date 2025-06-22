import pickle, pathlib, subprocess, os
import math, random

from machinations import Machinations
from render import Renderer

_model_pkl = pathlib.Path("machinations.pkl")
if not _model_pkl.exists():
    raise FileNotFoundError("machinations.pkl not found. Run test.py first.")

with _model_pkl.open("rb") as _fp:
    m: Machinations = pickle.load(_fp)

r = Renderer(m)
r.render(steps=10)  # adjust steps if desired

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

subprocess.run(["manim", "-pqh", "render/scene.py", "MachinationsScene"], check=True)

try:
    os.remove(_renderer_pkl)
except OSError:
    print("Warning: could not remove renderer.pkl")
    pass 
