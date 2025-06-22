from manim import Scene, Circle, Arrow, Dot, Text, DOWN

# Making sure the project root is on sys.path so that 'render.renderer' resolves.
import sys, pathlib, pickle

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from render.renderer import Renderer  # noqa: E402

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


def node_name(node):
    """Return a readable label for a Machinations node."""
    return getattr(node, "name", str(node))


class MachinationsScene(Scene):
    def construct(self):
        assert renderer is not None  # for static type checkers
        # Draw static graph based on renderer.model
        for node in renderer.model.nodes:
            x, y = getattr(node, "pos", (0, 0))
            color = getattr(node, "color", "WHITE")
            dot = Circle(radius=0.35, color=color, fill_opacity=0.7).set_fill(color).move_to([x, y, 0])
            label = Text(node_name(node), font_size=24).next_to(dot, direction=DOWN)
            self.add(dot, label)

        for conn in renderer.model.resource_connections:
            src_pos = getattr(conn.src, "pos", (0, 0))
            dst_pos = getattr(conn.dst, "pos", (0, 0))
            arrow = Arrow(start=[*src_pos, 0], end=[*dst_pos, 0], stroke_width=4)
            self.add(arrow)

        for _ in range(1, len(renderer.history)):
            self.wait(0.5)

