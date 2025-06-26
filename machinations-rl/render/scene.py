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

# Per-pair edge index bookkeeping (used to stagger overlapping arrows)
_edge_counter: dict[tuple[int, int], int] = {}

# CurvedArrow radius tweaking (avoids overlap)
_RADIUS_DELTA = 0.25  # incremental change per overlapping edge


def _next_edge_radius(src_id: int, dst_id: int, base_radius: float) -> float | None:
    """Return a signed *radius* to feed into CurvedArrow for the next edge
    connecting *src_id → dst_id* so that multiple edges are drawn with
    alternating curvature.

    The first edge keeps Manim's default (return *None*).  Subsequent edges
    alternate sign and gradually increase absolute value.
    """

    key = (src_id, dst_id)
    idx = _edge_counter.get(key, 0)
    _edge_counter[key] = idx + 1

    if idx == 0:
        return None  # default curvature

    # idx = 1 => -1 * (base+0Δ)
    # idx = 2 => +1 * (base+1Δ)
    # idx = 3 => -1 * (base+1Δ)
    # idx = 4 => +1 * (base+2Δ), etc.
    sign = -1 if idx % 2 == 1 else 1
    k = (idx + 1) // 2 - 1  # 0,0,1,1,2,2,…
    return sign * (base_radius + k * _RADIUS_DELTA)


def chord_endpoints(src_center, dst_center, dst_is_connection: bool = False):
    """Compute visually pleasing arrow endpoints that avoid entering the nodes.

    The function trims a *radius* from each node centre along the connecting
    line, then adds a small perpendicular offset so parallel edges are more
    distinguishable.  This keeps both the shaft and the tip clear of the
    circular/square glyphs used for nodes while preserving consistent spacing
    irrespective of zoom level.
    """

    src = np.asarray(src_center, dtype=float)
    dst = np.asarray(dst_center, dtype=float)

    # Vector from src → dst
    vec = dst - src
    length = np.linalg.norm(vec)
    if length == 0:
        return src.copy(), dst.copy()

    u = vec / length                           # unit direction

    # Keep arrowheads safely outside node shapes.
    R_SRC = 0.25                               # modest clearance at source
    # Shorter clearance for node destinations so arrow ends closer.
    R_DST = 0.25 if not dst_is_connection else 0.08

    perp = vec / length

    start_pt = src + u * R_SRC
    end_pt   = dst - u * R_DST

    return start_pt, end_pt

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
            distrib_res_ids = {d.resource_type.id for d in getattr(node, "distributions", [])}
            for j, amount in enumerate(resources):
                # Skip resources that were not mentioned in the constructor call for this node.
                if m.resources[j].id not in [*init_res_ids, *distrib_res_ids]:
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

            # Compute non-overlapping curvature radius for CurvedArrow
            base_rad = np.linalg.norm(c2 - c1) * 0.6  # heuristic: 60 % of chord len
            custom_radius = _next_edge_radius(c.src.id, c.dst.id, base_rad)

            p1, p2 = chord_endpoints(c1, c2, dst_is_connection=dst_is_conn)
            arrow_color = (
                getattr(c.resource_type, "color", BLACK)
                if hasattr(c, "resource_type") and c.resource_type is not None
                else BLACK
            )
            if c.type in [ElementType.TRIGGER, ElementType.MODIFIER]:
                arrow = Arrow(
                    start=[*p1, 0],
                    end=[*p2, 0],
                    stroke_width=0,
                    color=arrow_color,
                    buff=0.0,  # eliminate automatic gap between tip and end point
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
                ca_kwargs = {
                    "stroke_width": 2,
                    "color": arrow_color,
                }
                if custom_radius is not None:
                    ca_kwargs["radius"] = custom_radius

                arrow = CurvedArrow(
                    start_point=[*p1, 0],
                    end_point=[*p2, 0],
                    **ca_kwargs,
                )
                # Leave the tip at the pre-computed end position and orient it so
                # that it points directly towards the destination node centre
                # (instead of along the tangential direction of the curve).
                arrow.tip.scale(0.3)
                cur_vec = arrow.points[-1][:2] - arrow.points[-2][:2]
                desired_vec = c2 - p2  # centre of dst minus arrow end
                # Guard against zero-length vectors (can happen in degenerate layouts).
                if np.linalg.norm(cur_vec) > 1e-6 and np.linalg.norm(desired_vec) > 1e-6:
                    cur_ang = np.arctan2(cur_vec[1], cur_vec[0])
                    des_ang = np.arctan2(desired_vec[1], desired_vec[0])
                    arrow.tip.rotate(des_ang - cur_ang)

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
                if c.type == ElementType.MODIFIER:
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

            if hasattr(c, "predicate") and c.predicate is not None:
                # Remove existing $ delimiters from predicate repr to avoid nested math environments
                pred_body = str(c.predicate).strip("$")
                pred_text = "$P_{E_{" + str(c.id) + "}}\;=\;(" + pred_body + ")$"
                pred_tex = Tex(pred_text, font_size=12, color=BLACK)
                pred_tex.set_z_index(40)
                offset_factor = 2 if hasattr(c, "rate") else 1
                pred_tex.move_to(arrow_dot.get_center() + DOWN * 0.195 * offset_factor)
                self.add(pred_tex)
            arrow.points[-1] = arrow.tip.get_center()
            if c.type in [ElementType.TRIGGER, ElementType.MODIFIER]:
                dl = DashedLine(
                    start=[*p1, 0],
                    end=arrow.tip.get_center(),
                    stroke_width=1,
                    color=arrow_color,
                )
                self.add(dl)
            self.add(arrow, arrow_dot, arrow_label)

        for step in renderer.history:
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
                if m.nodes[i].distributions:
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
                    if col is None:
                        continue
                    # Already updated
                    if (m.nodes[i].distributions):
                        continue
                    animations.append(col.animate.set_value(X[i, j]))
            # Update all rate displays
            for idx, rate_val in enumerate(T_e):
                conn_id = m.resource_connections[idx].id
                animations.append(rate_displays[conn_id].animate.set_value(rate_val))
            if animations:
                self.play(*animations, run_time=0.5)

