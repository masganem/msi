from machinations import Machinations
from definitions import *

r1 = Resource("HP")
r2 = Resource("Mana")
n1 = Pool(FiringMode.AUTOMATIC, [("HP", 10)])
n2 = Pool(FiringMode.PASSIVE, [("Mana", 5)])
e1 = ResourceConnection(
        n1, n2, "HP", 2.0
    )
e2 = ResourceConnection(
        n2, n1, "Mana", -1.0
    )
p1 = Predicate(lambda x: x == 0)
e3 = Activator(n1, n2, p1)

m = Machinations.load((
    [n1, n2],
    [e1, e2, e3],
    [r1, r2],
))

print(m)
