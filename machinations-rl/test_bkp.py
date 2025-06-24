from math import inf
from machinations import Machinations
from render import Renderer
from machinations.definitions import *
import pickle, pathlib

r1 = Resource("HP")
r2 = Resource("Mana")
luck = Resource("Luck")

n1 = Pool(FiringMode.PASSIVE, [(r1, 10)])
n2 = Pool(FiringMode.PASSIVE, [(r2, 5)])
n3 = Gate(FiringMode.AUTOMATIC, DistributionMode.NONDETERMINISTIC, 6, luck)
n4 = Pool(FiringMode.PASSIVE, [(r1, 1)])
n5 = Gate(FiringMode.PASSIVE, DistributionMode.DETERMINISTIC, 6, r2)
n6 = Pool(FiringMode.PASSIVE, [(r2, inf)])

e1 = ResourceConnection(
        n1, n2, r1, 1.0
    )
e2 = ResourceConnection(
        n2, n1, r2, 1.0
    )
e3 = ResourceConnection(
        n5, n4, r2, 15.0, Predicate(">=", 15)
    )
e4 = Trigger(n3, n2, Predicate("<=", 3))
e5 = Trigger(n2, n5)
e6 = ResourceConnection(
        n6, n5, r2, 5.0
    )

m = Machinations.load((
    [n1, n2, n3, n4, n5, n6],
    [e1, e2, e3, e4, e5, e6],
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
