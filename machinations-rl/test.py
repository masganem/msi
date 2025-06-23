from machinations import Machinations
from render import Renderer
from machinations.definitions import *
import pickle, pathlib

r1 = Resource("HP")
r2 = Resource("Mana")
luck = Resource("Luck")

n1 = Pool(FiringMode.AUTOMATIC, [(r1, 10)])
n2 = Pool(FiringMode.PASSIVE, [(r2, 5)])
n3 = Gate(FiringMode.AUTOMATIC, DistributionMode.NONDETERMINISTIC, 6, luck)
n4 = Pool(FiringMode.PASSIVE, [(r1, 1)])

e1 = ResourceConnection(
        n1, n2, r1, 1.0
    )
e2 = ResourceConnection(
        n2, n1, r2, 1.0
    )
p1 = Predicate(">", 3)
e3 = Activator(n4, n1, p1, r1)
e4 = Trigger(n3, n2, Predicate("<", 5))
e5 = Trigger(n2, n4)

m = Machinations.load((
    [n1, n2, n3, n4],
    [e1, e2, e3, e4, e5],
    [r1, r2, luck],
))

r = Renderer(m)

# -----------------------------------------------------------------------
# Export the Machinations model so that it can be rendered later.
# -----------------------------------------------------------------------

_pkl_path = pathlib.Path("machinations.pkl")
with _pkl_path.open("wb") as _fp:
    pickle.dump(m, _fp)

print(f"Machinations model written to {_pkl_path}")
