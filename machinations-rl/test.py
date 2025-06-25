from math import inf
from machinations import Machinations
from render import Renderer
from machinations.definitions import *
import pickle, pathlib

r1 = Resource("Money")
r2 = Resource("Land")
luck = Resource("Luck")

# Player Money and Land
n1 = Pool(FiringMode.PASSIVE, [(r1, 100),(r2, 0)])

n2 = Gate(FiringMode.AUTOMATIC, DistributionMode.NONDETERMINISTIC, 12, luck)

# 1d6 generator
n3 = Gate(FiringMode.AUTOMATIC, DistributionMode.NONDETERMINISTIC, 6, luck)

# Bank
n4 = Pool(FiringMode.PASSIVE, [(r1, inf)])

e1 = ResourceConnection(
        n1, n4, r1, 10.0
    )
e2 = Trigger(n3, n4, Predicate("<=", 2))
e3 = ResourceConnection(
        n4, n1, r1, 100.0
    )
e4 = Trigger(n2, e3, Predicate("<=", 3))
e5 = LabelModifier(n1, e1, r1, .1)

n5 = Pool(FiringMode.PASSIVE, [(r2, 0)])
e6 = NodeModifier(n1, n5, r1, r2, .1)

n6 = Pool(FiringMode.INTERACTIVE, [])
e7 = ResourceConnection(n1, n6, r1, 120.0)
n7 = Pool(FiringMode.PASSIVE, [(r2, inf)])
e8 = Trigger(n6, n7)
e9 = ResourceConnection(n7, n1, r2, 1.0)

m = Machinations.load((
    [n1, n2, n3, n4, n5, n6, n7],
    [e1, e2, e3, e4, e5, e6, e7, e8, e9],
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
