from fractions import Fraction
from random import random
from manim import *
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
    with _pkl.open("rb") as _fp: renderer = pickle.load(_fp)
else:
    raise RuntimeError(
        "Machinations renderer not provided. Run test_render.py to create\n"
        "render/renderer.pkl or inject `scene.renderer` programmatically."
    )

def chord_endpoints(src_center, dst_center, dst_is_connection: bool = False):
    src = np.asarray(src_center, dtype=float)
    dst = np.asarray(dst_center, dtype=float)
    vec = dst - src
    dist = np.linalg.norm(vec)
    if dist == 0:
        return src.copy(), dst.copy()
    u = vec / dist
    # Base offsets (how far we move away from the centres along the src→dst vector)
    start_mag = 0.3               # keep constant for the source node
    end_mag   = 0.2 if dst_is_connection else 0.3  # shorter if arrow points to a connection centre

    start_pt = u * -start_mag
    end_pt   = u * end_mag
    R = random_rotation_matrix()
    return src + start_pt @ R , dst + end_pt

def random_rotation_matrix():
    theta = np.random.uniform(0, 0.4*np.pi)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

class MachinationsScene(Scene):
    def construct(self):
        assert renderer is not None  # for static type checkers
        self.camera.background_color = WHITE
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
                if node.firing_mode == FiringMode.INTERACTIVE:
                    dot_2 = Circle(radius=0.38, color=color, stroke_width=2, fill_opacity=0).move_to([x, y, 0])
                    self.add(dot_2)
            dot.set_fill("#f8f8f8")
            dot.set_z_index(1)
            node_displays[i] = dot

            label_text = getattr(node, "name", f"$V_{{{node.id}}}$")
            label = Tex(label_text, font_size=20, color=BLACK).next_to(dot, direction=ORIGIN)
            label.set_z_index(2)
            self.add(dot, label)

            if node.firing_mode == FiringMode.AUTOMATIC:
                mode_label = Tex("*", font_size=20, color=BLACK).move_to([x+0.33,y+0.33,0])
                self.add(mode_label)

            k = 0
            # Determine which resource ids were explicitly set in the node definition.
            init_res_ids = {res.id for res, _ in getattr(node, "initial_resources", [])}
            for j, amount in enumerate(resources):
                # Skip resources that were not mentioned in the constructor call for this node.
                if node.type == ElementType.GATE:
                    # For gates we still want to show only their configured resource_type
                    if m.resources[j].id != node.resource_type.id:
                        continue
                else:
                    if m.resources[j].id not in init_res_ids:
                        continue
                res_color = getattr(m.resources[j], "color", BLACK)
                sq = (
                    Square(side_length=0.25, fill_opacity=0.9)
                        .set_stroke(res_color, width=2)
                        .set_fill(res_color)
                        .next_to(dot, DOWN, buff=0.05)
                        .shift(RIGHT * k * 0.3)
                    )
                sq_label = (
                        Text(m.resources[j].name, font_size=20, color=res_color)
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
            c1 = np.array(getattr(c.src,  "pos", (0,0)), float)[:2]
            c2 = np.array(getattr(c.dst,  "pos", (0,0)), float)[:2]
            dst_is_conn = isinstance(c.dst, Connection)
            p1, p2 = chord_endpoints(c1, c2, dst_is_connection=dst_is_conn)
            arrow_color = (
                getattr(c.resource_type, "color", BLACK)
                if hasattr(c, "resource_type") and c.resource_type is not None
                else BLACK
            )
            if c.type in [ElementType.TRIGGER, ElementType.LABEL_MODIFIER, ElementType.NODE_MODIFIER]:
                arrow = Arrow(
                    start=[*p1, 0],
                    end=[*p2, 0],
                    stroke_width=0,
                    color=arrow_color,
                )
                arrow.tip.scale(0.3)

                # Add dot + label in the middle for triggers as well
                arrow_dot = (
                    Circle(radius=0.14, stroke_width=2, color=arrow_color, fill_opacity=1)
                    .set_fill(WHITE)
                    .move_to(arrow.points[len(arrow.points)//2])
                )
                arrow_dot.set_z_index(20)
                if c.type == ElementType.TRIGGER:
                    self.add(Tex("*", font_size=20, color=BLACK).move_to(arrow_dot.get_center() + [.125, .125, .0]))

                conn_label_text = getattr(c, "name", f"$E_{{{c.id}}}$")
                arrow_label = Tex(
                    conn_label_text,
                    font_size=12,
                    color=BLACK,
                ).move_to(arrow_dot.get_center())
                arrow_label.set_z_index(21)
            else:
                arrow = CurvedArrow(
                    start_point=[*p1, 0],
                    end_point=[*p2, 0],
                    stroke_width=2,
                    color=arrow_color,
                )
                arrow.tip.move_to(arrow.points[-2]).scale(0.3)

                # Middle dot that will carry the edge label
                arrow_dot = (
                    Circle(radius=0.14, stroke_width=2, color=arrow_color, fill_opacity=1)
                    .set_fill(WHITE)
                    .move_to(arrow.points[len(arrow.points)//2])
                )
                arrow_dot.set_z_index(20)

                # Label (e.g. $E_i$) shown inside the dot
                conn_label_text = getattr(c, "name", f"$E_{{{c.id}}}$")
                arrow_label = Tex(
                    conn_label_text,
                    font_size=12,
                    color=BLACK,
                ).move_to(arrow_dot.get_center())
                # Above the dot
                arrow_label.set_z_index(21)

            connection_displays[c.id] = arrow
            arrow.set_z_index(-1)
            c.pos = arrow_dot.get_center()

            # Display rate as label + numeric value for easy in-place updating
            if hasattr(c, "rate"):
                if c.type == ElementType.LABEL_MODIFIER or c.type == ElementType.NODE_MODIFIER:
                    label_str = "$\\dot{T}_{E_" + str(c.id) + "} =$"
                    frac = Fraction(c.rate).limit_denominator(100)
                    value_num = Tex("$\\frac{" + str(frac.numerator) + "}{" + str(frac.denominator) + "}$", font_size=12, color=BLACK)
                else:
                    label_str = "$T_{E_" + str(c.id) + "} =$"
                    value_num = Integer(c.rate, font_size=12, color=BLACK)
                    if c.type == ElementType.RESOURCE_CONNECTION:
                        rate_displays[c.id] = value_num  # store the numeric part for updates

                label_tex = Tex(label_str, font_size=12, color=BLACK)
                value_num.next_to(label_tex, RIGHT, buff=0.05)

                # Ensure these labels sit above nodes as well
                label_tex.set_z_index(40)
                value_num.set_z_index(40)

                group_pos = arrow_dot.get_center() + DOWN * 0.195
                VGroup(label_tex, value_num).move_to(group_pos)

                self.add(label_tex, value_num)

            # Predicate label (static)
            if hasattr(c, "predicate") and c.predicate is not None:
                # Remove existing $ delimiters from predicate repr to avoid nested math environments
                pred_body = str(c.predicate).strip("$")
                pred_text = "$P_{E_" + str(c.id) + "}\;=\;(" + pred_body + ")$"
                pred_tex = Tex(pred_text, font_size=12, color=BLACK)
                pred_tex.set_z_index(40)
                offset_factor = 2 if hasattr(c, "rate") else 1
                pred_tex.move_to(arrow_dot.get_center() + DOWN * 0.195 * offset_factor)
                self.add(pred_tex)
            arrow.points[-1] = arrow.tip.get_center()
            if c.type in [ElementType.TRIGGER, ElementType.LABEL_MODIFIER, ElementType.NODE_MODIFIER]:
                dl = DashedLine(
                    start=[*p1, 0],
                    end=arrow.tip.get_center(),
                    stroke_width=1,
                    color=arrow_color,
                )
                self.add(dl)
            self.add(arrow, arrow_dot, arrow_label)

        for step in renderer.history:
            # Access by key to remain robust even if extra fields are present
            t            = step['t']
            X            = step['X']
            T_e          = step['T_e']
            V_active     = step['V_active']
            E_R_active   = step['E_R_active']
            E_G_active   = step['E_G_active']
            print(f"{m.V_active=}")

            animations = [time_display.animate.set_value(t)]
            self.play(*animations, run_time=0.5)

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
            # Update all rate displays
            for idx, rate_val in enumerate(T_e):
                conn_id = m.resource_connections[idx].id
                animations.append(rate_displays[conn_id].animate.set_value(rate_val))
            self.play(*animations, run_time=0.5)

