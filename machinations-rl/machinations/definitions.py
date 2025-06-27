from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from numba import njit
import numpy as np
import operator

class Predicate:
    _map = {'==':0,'<':1,'<=':2,'>':3,'>=':4,'!=':5}
    def __init__(self, op: str, c: float):
        try:
            self.op_code = self._map[op]
        except KeyError:
            raise ValueError(f"bad op {op!r}")
        self.c = c
    def __repr__(self):
        return f"${list(self._map)[self.op_code]}{self.c}$"

class Resource:
    def __init__(self, name):
        self.name = name
        self.id = None

class FiringMode(Enum):
    PASSIVE = 0
    AUTOMATIC = 1
    INTERACTIVE = 2
    NONDETERMINISTIC = 3

class ElementType(Enum):
    NODE = 1
    RESOURCE_CONNECTION = 2
    MODIFIER = 3
    TRIGGER = 4
    ACTIVATOR = 5

class Distribution:
    def __init__(self, resource_type: Resource, values, *, infinite: bool = True):
        """A resource distribution.

        Parameters
        ----------
        resource_type : Resource
            The type of resource generated.
        values : Sequence
            Population to sample from.
        infinite : bool, optional
            If *True* (default) the population is sampled **with** replacement
            (i.e. unlimited supply).  If *False*, every sample is removed
            from *values*; once the list becomes empty, further sampling has
            no effect.
        """
        self.resource_type = resource_type
        # Store as a list so we can ``pop`` when finite.
        self.values = list(values)
        self.infinite = infinite

    def draw(self, rng) -> float | None:
        """Sample one value according to *infinite* flag.

        Returns the drawn value **or** *None* if the distribution is empty
        (finite and depleted).
        """
        if not self.values:
            return None  # empty finite deck

        choice = float(rng.choice(self.values))
        if not self.infinite:
            # Remove the selected card so it cannot be drawn again.
            # Using list.remove rather than pop(index) keeps it simple.
            self.values.remove(choice)
        return choice

class Node:
    def __init__(self, firing_mode: FiringMode, initial_resources = [], distributions: List[Distribution] = None):
        # Please mirror this in Machinations.load
        self.id = None
        self.firing_mode = firing_mode
        self.type = ElementType.NODE
        # If a node has a distribution, it will sample from it 
        # whenever it fires
        self.distributions = distributions 
        self.initial_resources = initial_resources

# ---------------------------------------------------------------------------
# Special terminal nodes (WIN / TIE / LOSE)
# ---------------------------------------------------------------------------

class OutcomeNode(Node):
    """A node that immediately ends the episode when it fires.

    Parameters
    ----------
    outcome : str
        One of ``"win"``, ``"tie"``, ``"lose"``.
    firing_mode : FiringMode, optional
        Defaults to :pyattr:`FiringMode.PASSIVE` – typically an OutcomeNode
        is triggered by another connection rather than sampling on its own.
    """

    VALID = {"win", "tie", "lose"}

    def __init__(self, outcome: str, *, firing_mode: FiringMode = FiringMode.PASSIVE):
        if outcome not in self.VALID:
            raise ValueError(f"Outcome must be one of {self.VALID}, got {outcome!r}")
        super().__init__(firing_mode=firing_mode, initial_resources=[], distributions=None)
        self.outcome = outcome

class Connection:
    def __init__(self, src: Node, dst: Node | Connection):
        self.id = None
        self.src = src
        self.dst = dst
        self.predicate = None

class ResourceConnection(Connection):
    def __init__(self, src: Node, dst: Node, resource_type: Resource, rate=1.0):
        super().__init__(src, dst)
        self.type = ElementType.RESOURCE_CONNECTION
        self.rate = rate
        self.resource_type = resource_type

    def pack(self):
        return np.array([
                self.id,
                self.src.id,
                self.dst.id,
                self.resource_type.id,
                self.rate,  
            ], dtype=np.float64)


class Modifier(Connection):
    def __init__(self, src: Node, dst: Node | Connection, src_resource_type: Resource, dst_resource_type: Resource, rate=1.0):
        super().__init__(src, dst)
        self.type = ElementType.MODIFIER
        self.rate = rate
        self.src_resource_type = src_resource_type
        self.dst_resource_type = dst_resource_type
        self.dst_type = dst.type
        assert self.dst_type in [ElementType.RESOURCE_CONNECTION, ElementType.NODE]

    def pack(self):
        return np.array([
                self.id,
                self.src.id,
                self.dst.id,
                self.src_resource_type.id,
                self.dst_resource_type.id,
                self.rate,
                self.dst_type.value,
            ], dtype=np.float64)

class Trigger(Connection):
    def __init__(self, src: Node, dst: Node | Connection, resource_type = None, predicate: Predicate = None):
        super().__init__(src, dst)
        self.type = ElementType.TRIGGER
        self.predicate = predicate
        self.resource_type = resource_type

        self.dst_type = dst.type
        assert self.dst_type in [ElementType.RESOURCE_CONNECTION, ElementType.NODE]

    def pack(self):
        return np.array([
                self.id,
                self.src.id,
                self.dst.id,
                self.predicate.id if self.predicate else -1,
                self.resource_type.id if self.resource_type else -1,
                self.dst_type.value,
            ], dtype=np.float64)

class Activator(Connection):
    def __init__(self, src: Node, dst: Node | Connection, resource_type: Resource, predicate: Predicate):
        super().__init__(src, dst)
        self.type = ElementType.ACTIVATOR
        self.resource_type = resource_type
        self.predicate = predicate

        self.dst_type = dst.type
        assert self.dst_type in [ElementType.RESOURCE_CONNECTION, ElementType.NODE]

    def pack(self):
        return np.array([
                self.id,
                self.src.id,
                self.dst.id,
                self.predicate.id,
                self.resource_type.id,
                self.dst_type.value,
            ], dtype=np.float64)

Diagram = Tuple[List[Node], List[Connection], List[Resource]]
