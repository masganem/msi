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
        self.camera.background_color = WHITE
        assert renderer is not None  # for static type checkers
        m = renderer.model
        time_display = Integer(0, font_size=20, color=BLACK).move_to(3*DOWN+3.5*RIGHT)
        self.add(Tex("$t =$", font_size=20, color=BLACK).next_to(time_display, direction=LEFT, buff=.1))
        self.add(time_display)
        node_displays = [None for _ in m.nodes]
        value_displays = [
            [None for _ in m.resources]
            for _ in m.nodes
        ]
        rate_displays = [None for _ in m.connections]
        connection_displays = [None for _ in m.connections]

        # Draw static graph based on renderer.model
        for i, (node, resources) in enumerate(zip(m.nodes, m.X)):
            x, y = getattr(node, "pos", (0, 0))
            color = getattr(node, "color", "BLACK")

            if node.type == ElementType.GATE:
                dot = Square(side_length=0.7, color=color, stroke_width=2, fill_opacity=1).move_to([x, y, 0]).rotate(3.14/4)
            else:
                dot = Circle(radius=0.35, color=color, stroke_width=2, fill_opacity=1).move_to([x, y, 0])
            dot.set_fill("#f8f8f8")
            dot.set_z_index(1)
            node_displays[i] = dot

            label = Tex(node.name, font_size=20, color=BLACK).next_to(dot, direction=ORIGIN)
            label.set_z_index(2)
            self.add(dot, label)

            if node.firing_mode == FiringMode.AUTOMATIC:
                mode_label = Tex("*", font_size=20, color=BLACK).move_to([x+0.33,y+0.33,0])
                self.add(mode_label)

            k = 0
            for j, amount in enumerate(resources):
                if node.type == ElementType.GATE and not m.resources[j].id == node.resource_type.id:
                    continue
                sq = (
                    Square(side_length=0.25, fill_opacity=0.9)
                        .set_stroke(m.resources[j].color, width=2)
                        .set_fill(m.resources[j].color)
                        .next_to(dot, DOWN, buff=0.05)
                        .shift(RIGHT * k * 0.3)
                    )
                sq_label = (
                        Text(m.resources[j].name, font_size=20, color=m.resources[j].color)
                        .next_to(sq, direction=DOWN, buff=-0.04)
                        .scale(0.3)
                    )
                sq_value = (
                        Integer(amount, font_size=20)
                        .next_to(sq, direction=ORIGIN)
                        .scale(0.6)
                    )
                value_displays[i][j]=sq_value
                self.add(sq, sq_label, sq_value)
                k += 1

        for c in m.connections:
            c1 = np.array(getattr(c.src,  "pos", (0,0)), float)
            c2 = np.array(getattr(c.dst,  "pos", (0,0)), float)
            p1, p2 = chord_endpoints(c1, c2)
            arrow_color = c.resource_type.color if hasattr(c, "resource_type") and c.resource_type else BLACK
            if c.type == ElementType.TRIGGER:
                arrow = Arrow(
                    start=[*p1, 0],
                    end=[*c2, 0],
                    stroke_width=0,
                    color=arrow_color,
                )
                dl = DashedLine(
                    start=[*p1, 0],
                    end=[*c2, 0],
                    stroke_width=1,
                    color=arrow_color,
                )
                self.add(dl, Tex("*", font_size=20, color=BLACK).move_to(arrow.points[-2]))
                arrow.tip.scale(0.5)
            else:
                arrow = CurvedArrow(
                    start_point=[*p1, 0],
                    end_point=[*c2, 0],
                    stroke_width=2,
                    color=arrow_color,
                )
                arrow.tip.move_to(arrow.points[-2]).scale(0.5)
                arrow_dot = Circle(radius=0.02, stroke_width=0, color=BLACK, fill_opacity=1).move_to(arrow.points[len(arrow.points)//2])
                arrow_dot.set_fill(arrow_color)
            connection_displays[c.id] = arrow
            arrow.set_z_index(-1)
            arrow_label = Tex(c.name, font_size=20, color=BLACK).move_to(arrow.points[len(arrow.points)//2] + UP * 0.175)

            # Display rate as label + numeric value for easy in-place updating
            if hasattr(c, "rate"):
                label_tex = Tex("$T_{E_" + str(c.id) + "} =$", font_size=16, color=BLACK)
                value_num = Integer(c.rate, font_size=16, color=BLACK)
                value_num.next_to(label_tex, RIGHT, buff=0.05)

                group_pos = arrow.points[len(arrow.points)//2] + DOWN * 0.175
                VGroup(label_tex, value_num).move_to(group_pos)

                rate_displays[c.id] = value_num  # store the numeric part for updates
                self.add(label_tex, value_num)

            # Predicate label (static)
            if hasattr(c, "predicate") and c.predicate is not None:
                # Remove existing $ delimiters from predicate repr to avoid nested math environments
                pred_body = str(c.predicate).strip("$")
                pred_text = "$P_{E_" + str(c.id) + "}\;=\;(" + pred_body + ")$"
                pred_tex = Tex(pred_text, font_size=16, color=BLACK)
                offset_factor = 2 if hasattr(c, "rate") else 1
                pred_tex.move_to(arrow.points[len(arrow.points)//2] + DOWN * 0.175 * offset_factor)
                self.add(pred_tex)
            self.add(arrow, arrow_dot, arrow_label)

        for step in renderer.history:
            t, X, T_e, V_active, E_R_active, E_G_active = step.values()
            print(f"{m.V_active=}")

            animations = [time_display.animate.set_value(t)]
            self.play(*animations, run_time=0.5)

            animations = []
            for i, row in enumerate(node_displays):
                if m.V_satisfied[i]:
                    animations.append(row.animate.set_fill(GREEN))
            self.play(*animations, run_time=0.5, rate_func=there_and_back)

            # Show random gates generating
            for i,row in enumerate(value_displays):
                if m.nodes[i].type == ElementType.GATE and m.nodes[i].distribution_mode == DistributionMode.NONDETERMINISTIC:
                    for j,col in enumerate(row):
                        if col:
                            animations.append(col.animate.set_value(X[i,j]))
            if animations:
                self.play(*animations, run_time=0.5)

            animations = []
            for i, row in enumerate(E_G_active):
                if row:
                    arrow = connection_displays[m.triggers[i].id]
                    animations.append(arrow.animate.set_stroke_width(2))
            if animations:
                self.play(*animations, run_time=0.5, rate_func=there_and_back)

            animations = []
            for i, row in enumerate(V_active):
                if row:
                    animations.append(node_displays[i].animate.set_stroke_width(5))
            for i, row in enumerate(E_R_active):
                if row:
                    arrow = connection_displays[m.resource_connections[i].id]
                    animations.append(arrow.animate.set_stroke_width(5))
                    animations.append(arrow.tip.animate.move_to(arrow.points[-1]))
            if animations:
                self.play(*animations, run_time=1.0, rate_func=there_and_back)

            animations = []
            for i, row in enumerate(value_displays):
                for j, col in enumerate(row):
                    if col:
                        animations.append(col.animate.set_value(X[i,j]))
            for i, row in enumerate(T_e):
                conn_id = m.resource_connections[i].id
                if row:
                    animations.append(rate_displays[conn_id].animate.set_value(row))
            self.play(*animations, run_time=0.5)

