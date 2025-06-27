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
    R_SRC = 0.34                               # modest clearance at source
    # Bring the tip much closer when the arrow targets another *connection* –
    # i.e. its destination visualised as a tiny dot.  Using zero clearance
    # makes the tip apex coincide with the dot centre.
    R_DST = 0.25 if not dst_is_connection else -0.1

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
        predicates_list = []
        aliases_list = []
        resources_list = []

        # Draw static graph based on renderer.model
        for i, (node, resources) in enumerate(zip(m.nodes, m.X)):
            x, y = getattr(node, "pos", (0, 0))
            color = getattr(node, "color", "BLACK")

            dot = Circle(radius=0.35, color=color, stroke_width=2, fill_opacity=0).move_to([x, y, 0])
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

            # Collect alias label (for legend) if provided
            if hasattr(node, "alias"):
                alias_str = getattr(node, "alias")
                alias_tex = Tex(f"$V_{{{node.id}}} = \\text{{{alias_str}}}$", font_size=16, color=BLACK)
                alias_tex.set_z_index(40)
                aliases_list.append(alias_tex)

            # Determine which resource ids were explicitly set in the node definition.
            init_res_ids = {res.id for res, _ in node.initial_resources} if node.initial_resources else set()
            distrib_res_ids = {d.resource_type.id for d in node.distributions} if node.distributions else set()

            # Collect the indices of resources that should be displayed so we can
            # compute a centred layout for their numeric labels.
            display_res_indices: list[int] = []
            for j, _ in enumerate(resources):
                if m.resources[j].id in (*init_res_ids, *distrib_res_ids):
                    display_res_indices.append(j)

            n_vals = len(display_res_indices)
            spacing = 0.175  # horizontal distance between successive values

            for k, j in enumerate(display_res_indices):
                amount = resources[j]
                res_color = getattr(m.resources[j], "color", BLACK)

                # Compute horizontal offset so the group of values is centred
                # beneath the node.
                offset_x = (k - (n_vals - 1) / 2) * spacing
                shift_vec = DOWN * 0.175 + RIGHT * offset_x

                sq_value = (
                    Integer(amount, font_size=20, stroke_width=0.7, color=res_color)
                    .next_to(dot, ORIGIN)
                    .shift(shift_vec)
                )
                sq_value.scale(0.6)
                sq_value.set_z_index(100)

                value_displays[i][j] = sq_value
                self.add(sq_value)

        for c in m.connections:
            c1 = np.array(getattr(c.src,  "pos", (0,0)), float)[:2]
            c2 = np.array(getattr(c.dst,  "pos", (0,0)), float)[:2]
            # Destination is considered a "connection" when it is an actual
            # Connection instance (i.e. Modifier/Activator points to a
            # ResourceConnection).  We use this to shrink the final clearance
            # so the arrow tip sits flush with the tiny connection dot.
            dst_is_conn = isinstance(c.dst, Connection)

            # Compute non-overlapping curvature radius for CurvedArrow
            base_rad = np.linalg.norm(c2 - c1) * 0.6  # heuristic: 60 % of chord len
            custom_radius = _next_edge_radius(c.src.id, c.dst.id, base_rad)

            p1, p2 = chord_endpoints(c1, c2, dst_is_connection=dst_is_conn)
            # --------------------------------------------------
            # Arrow color logic
            #   • Triggers & Activators → BLACK
            #   • Modifiers            → color of *destination* resource
            #   • Resource connections → color of connection's resource_type
            # --------------------------------------------------
            if c.type in (ElementType.TRIGGER, ElementType.ACTIVATOR):
                arrow_color = BLACK
            elif c.type == ElementType.MODIFIER:
                # Prefer destination resource color if available
                if hasattr(c, "dst_resource_type") and c.dst_resource_type is not None:
                    arrow_color = getattr(c.dst_resource_type, "color", BLACK)
                elif hasattr(c, "resource_type") and c.resource_type is not None:
                    arrow_color = getattr(c.resource_type, "color", BLACK)
                else:
                    arrow_color = BLACK
            else:
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
                    Circle(radius=0.05, stroke_width=0, color=arrow_color, fill_opacity=1)
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
                    Circle(radius=0.05, stroke_width=0, color=arrow_color, fill_opacity=1)
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
                is_resource_conn = c.type == ElementType.RESOURCE_CONNECTION
                is_modifier      = c.type == ElementType.MODIFIER

                # --------------------------------------------------
                # (1) Resource-connection rate == 1 → omit label
                # --------------------------------------------------
                if is_resource_conn and abs(c.rate - 1) < 1e-9:
                    rate_displays[c.id] = None  # placeholder so index exists
                # --------------------------------------------------
                # (2) Modifier with ±1 rate → show + / – sign
                # --------------------------------------------------
                elif is_modifier and abs(abs(c.rate) - 1) < 1e-9:
                    sign = "+" if c.rate > 0 else "-"
                    sign_tex = Tex(sign, font_size=20, color=BLACK).move_to(arrow_dot.get_center() + np.array([.125, .125, 0]))
                    sign_tex.set_z_index(21)
                    self.add(sign_tex)
                # --------------------------------------------------
                # (3) All other rates → full label
                # --------------------------------------------------
                else:
                    if is_modifier:
                        label_str = f"$\\dot{{T}}_{{E_{{{c.id}}}}} =$"
                        frac = Fraction(c.rate).limit_denominator(100)
                        value_num = Tex(f"$\\frac{{{frac.numerator}}}{{{frac.denominator}}}$", font_size=12, color=BLACK)
                    else:
                        label_str = f"$T_{{E_{{{c.id}}}}} =$"
                        value_num = Integer(c.rate, font_size=12, color=BLACK)
                        if is_resource_conn:
                            rate_displays[c.id] = value_num

                    label_tex = Tex(label_str, font_size=12, color=BLACK)
                    value_num.next_to(label_tex, RIGHT, buff=0.05)

                    # Ensure these labels sit above nodes as well
                    label_tex.set_z_index(40)
                    value_num.set_z_index(40)

                    group_pos = arrow_dot.get_center() + DOWN * 0.195
                    VGroup(label_tex, value_num).move_to(group_pos)

                    self.add(label_tex, value_num)

            # Collect predicate labels to list later
            if hasattr(c, "predicate") and c.predicate is not None:
                pred_body = str(c.predicate).strip("$")
                pred_text = f"$P_{{E_{{{c.id}}}}} = ({pred_body})$"
                print(f"[renderer] Collect predicate label for edge E_{c.id}: {pred_body}")
                _pred_label = Tex(pred_text, font_size=16, color=BLACK)
                _pred_label.set_z_index(40)
                predicates_list.append(_pred_label)

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

        # ------------------------------------------------------------
        # Top-left legend of node aliases
        # ------------------------------------------------------------
        if aliases_list:
            alias_group = VGroup(*aliases_list).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            alias_group.scale(0.8)
            alias_group.to_corner(UP + LEFT)
            self.add(alias_group)

        # ------------------------------------------------------------
        # Top-right legend of resources (name in its colour)
        # ------------------------------------------------------------
        for res in m.resources:
            res_name = getattr(res, "name", f"R_{res.id}")
            res_color = getattr(res, "color", BLACK)
            res_tex = Tex(f"$\\text{{{res_name}}}$", font_size=22, color=res_color)
            res_tex.set_z_index(40)
            resources_list.append(res_tex)

        if resources_list:
            res_group = VGroup(*resources_list).arrange(DOWN, aligned_edge=RIGHT, buff=0.1)
            res_group.scale(0.8)
            res_group.to_corner(UP + RIGHT)
            self.add(res_group)

        # ------------------------------------------------------------
        # Bottom-left list of predicate expressions
        # ------------------------------------------------------------
        if predicates_list:
            total = len(predicates_list)
            print(f"[renderer] Total predicate labels: {total}")
            preds_group = VGroup(*predicates_list).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            preds_group.scale(0.6)
            preds_group.to_corner(DOWN + LEFT)
            self.add(preds_group)

        for step in renderer.history:
            t            = step['t']
            X            = step['X']
            T_e          = step['T_e']
            V_active     = step['V_active']
            E_R_active   = step['E_R_active']
            E_G_active   = step['E_G_active']
            print(f"{m.V_active=}")

            phase = step.get("phase", "post")

            # Time always advances with every snapshot
            animations = [time_display.animate.set_value(t)]

            if phase == "pre":
                # Pre-transfer snapshot: show new resource values *and* any
                # updated edge rates so the viewer sees the numbers that the
                # upcoming transfer will use.  Edge highlights are still
                # skipped – they belong to the post-phase.

                # Update node resource values
                for i, row in enumerate(value_displays):
                    for j, col in enumerate(row):
                        if col is None:
                            continue
                        animations.append(col.animate.set_value(X[i, j]))

                # Do NOT update connection-rate labels here – they will be
                # refreshed in the subsequent *post* snapshot so that the
                # displayed value matches the actual transfer that is about
                # to be visualised.

                # Play and continue to next snapshot.
                if animations:
                    self.play(*animations, run_time=0.5)
                continue  # skip the rest of the loop for pre-phase

            if phase == "mid":
                # Show updated edge rates that will be used in the upcoming
                # transfer.  We do NOT touch node resource values here – those
                # will change only after the transfer.

                for idx, rate_val in enumerate(T_e):
                    conn_id = m.resource_connections[idx].id
                    label = rate_displays[conn_id]
                    if label is not None:
                        animations.append(label.animate.set_value(rate_val))

                if animations:
                    self.play(*animations, run_time=0.3)
                continue  # proceed to next snapshot

            # --- POST phase (after kernel) ---
            # Only resource values change here because edge-rate labels were
            # already updated in the preceding *mid* snapshot.

            # Store resource value updates to apply AFTER highlights
            value_updates = []
            for i, row in enumerate(value_displays):
                for j, col in enumerate(row):
                    if col is None:
                        continue
                    value_updates.append(col.animate.set_value(X[i, j]))

            # ------------------------------------------------------------
            # Highlight triggers first (E_G_active)
            # ------------------------------------------------------------
            highlight_anims = []
            for i, row in enumerate(E_G_active):
                if row:
                    arrow = connection_displays[m.triggers[i].id]
                    highlight_anims.append(arrow.animate.set_stroke_width(2))
            if highlight_anims:
                self.play(*highlight_anims, run_time=0.5, rate_func=there_and_back)

            # ------------------------------------------------------------
            # Highlight active nodes and resource connections
            # ------------------------------------------------------------
            highlight_anims = []
            for i, row in enumerate(V_active):
                if row:
                    highlight_anims.append(node_displays[i].animate.set_stroke_width(5))
            for i, row in enumerate(E_R_active):
                if row:
                    arrow = connection_displays[m.resource_connections[i].id]
                    highlight_anims.append(arrow.animate.set_stroke_width(5))
                    highlight_anims.append(arrow.tip.animate.move_to(arrow.points[-1]))
            if highlight_anims:
                self.play(*highlight_anims, run_time=1.0, rate_func=there_and_back)

            # Now apply the deferred resource value updates
            if value_updates:
                self.play(*value_updates, run_time=0.5)

            # rate labels remain unchanged in POST

