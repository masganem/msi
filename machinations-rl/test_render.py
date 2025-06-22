import pickle, pathlib, subprocess, os
import math, random

from machinations import Machinations
from render import Renderer

_model_pkl = pathlib.Path("machinations.pkl")
if not _model_pkl.exists():
    raise FileNotFoundError("machinations.pkl not found. Run test.py first.")

with _model_pkl.open("rb") as _fp:
    mach_model: Machinations = pickle.load(_fp)

renderer = Renderer(mach_model)
renderer.render(steps=10)  # adjust steps if desired

n_nodes = len(mach_model.nodes)
radius = 4.0

for idx, node in enumerate(mach_model.nodes):
    angle = 2 * math.pi * idx / n_nodes
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    node.pos = (x, y)  # type: ignore[attr-defined]
    node.color = random.choice(["#d62728", "#9467bd", "#ff7f0e"])

_renderer_pkl = pathlib.Path("render") / "renderer.pkl"
with _renderer_pkl.open("wb") as _fp:
    pickle.dump(renderer, _fp)

subprocess.run(["manim", "-pql", "render/scene.py", "MachinationsScene"], check=True)

try:
    os.remove(_renderer_pkl)
except OSError:
    print("Warning: could not remove renderer.pkl")
    pass 
