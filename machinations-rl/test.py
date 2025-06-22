from machinations import Machinations
from render import Renderer
from machinations.definitions import *
import pickle, pathlib

r1 = Resource("HP")
r2 = Resource("Mana")
n1 = Pool(FiringMode.AUTOMATIC, [("HP", 10)])
n2 = Pool(FiringMode.PASSIVE, [("Mana", 5)])
n3 = Gate(FiringMode.AUTOMATIC, DistributionMode.NONDETERMINISTIC, OutputMode.CONDITIONAL, 6, "HP")
e1 = ResourceConnection(
        n1, n2, "HP", 2.0
    )
e2 = ResourceConnection(
        n2, n1, "Mana", -1.0
    )
p1 = Predicate("==", 0)
e3 = Activator(n2, n1, p1, "HP")

m = Machinations.load((
    [n1, n2],
    [e1, e2, e3],
    [r1, r2],
))

r = Renderer(m)

# -----------------------------------------------------------------------
# Export the Machinations model so that it can be rendered later.
# -----------------------------------------------------------------------

_pkl_path = pathlib.Path("machinations.pkl")
with _pkl_path.open("wb") as _fp:
    pickle.dump(m, _fp)

print(f"Machinations model written to {_pkl_path}")
