from manim import *
from random import random
import numpy as np

# Making sure the project root is on sys.path so that 'render.renderer' resolves.
import sys, pathlib, pickle

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from render.renderer import Renderer  # noqa: E402
from machinations.definitions import *

# Obtain a Renderer instance:
# If the driving script runs Manim _in the same process_, it can set
# `scene.renderer` before calling `MachinationsScene`.
renderer: Renderer | None = globals().get("renderer")  # type: ignore
# If Manim is spawned as a subprocess, we look for 'render/renderer.pkl'.
_pkl = pathlib.Path(__file__).with_name("renderer.pkl")
if _pkl.exists():
    with _pkl.open("rb") as _fp:
        renderer = pickle.load(_fp)
else:
    raise RuntimeError(
        "Machinations renderer not provided. Run test_render.py to create\n"
        "render/renderer.pkl or inject `scene.renderer` programmatically."
    )

def chord_endpoints(src_center, dst_center):
    src = np.asarray(src_center, dtype=float)
    dst = np.asarray(dst_center, dtype=float)
    vec = dst - src
    dist = np.linalg.norm(vec)
    if dist == 0:
        return src.copy(), dst.copy()
    u = vec / dist
    start_pt = u * 0.33
    end_pt   = u * -0.33
    R = random_rotation_matrix()
    return src + start_pt @ R , dst + end_pt

def random_rotation_matrix():
    theta = np.random.uniform(0, 0.4*np.pi)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

class MachinationsScene(Scene):
    def construct(self):
        assert renderer is not None  # for static type checkers
        m = renderer.model
        # Draw static graph based on renderer.model
        for node, resources in zip(m.nodes, m.X):
            x, y = getattr(node, "pos", (0, 0))
            color = getattr(node, "color", "WHITE")

            if node.type == ElementType.GATE:
                dot = Square(side_length=0.7, color=color).move_to([x, y, 0]).rotate(3.14/4)
            else:
                dot = Circle(radius=0.35, color=color).move_to([x, y, 0])

            label = Tex(node.name, font_size=24).next_to(dot, direction=UP)
            self.add(dot, label)

            for i, amount in enumerate(resources):
                sq = (
                    Square(side_length=0.35, fill_opacity=0.7)
                        .set_stroke(m.resources[i].color, width=2)
                        .set_fill(m.resources[i].color)
                        .next_to(dot, DOWN, buff=0.25)
                        .shift(RIGHT * (i-0.5) * 0.4)
                    )
                sq_label = (
                        Text(m.resources[i].name, font_size=24)
                        .next_to(sq, direction=UP, buff=0.01)
                        .scale(0.4)
                    )
                sq_value = (
                        Text(str(amount), font_size=24)
                        .next_to(sq, direction=DOWN, buff=-0.3)
                        .scale(0.4)
                    )
                self.add(sq, sq_label, sq_value)

        for c in m.resource_connections:
            c1 = np.array(getattr(c.src,  "pos", (0,0)), float)
            c2 = np.array(getattr(c.dst,  "pos", (0,0)), float)
            p1, p2 = chord_endpoints(c1, c2)
            arrow = CurvedArrow(
                start_point=[*p1, 0],
                end_point=[*c2, 0],
                stroke_width=2,
                color=c.resource_type.color,
            )
            arrow.points = arrow.points[:-1]
            arrow.tip.move_to(arrow.points[-2])
            arrow.set_z_index(-1)
            arrow_label = Tex(c.name, font_size=24).move_to(arrow.points[len(arrow.points)//2] + UP * 0.25)
            self.add(arrow, arrow_label)

        for _ in range(1, len(renderer.history)):
            self.wait(0.5)

