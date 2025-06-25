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
    def __init__(self, name, unique = False):
        self.name = name
        self.unique = unique
        self.id = None

class FiringMode(Enum):
    PASSIVE = 0
    AUTOMATIC = 1
    INTERACTIVE = 2

class DistributionMode(Enum):
    ANY = -1
    DETERMINISTIC = 0
    NONDETERMINISTIC = 1

class ElementType(Enum):
    ANY = -1
    POOL = 0
    GATE = 1
    RESOURCE_CONNECTION = 2
    LABEL_MODIFIER = 3
    NODE_MODIFIER = 4
    TRIGGER = 5
    ACTIVATOR = 6

class Node:
    def __init__(self, firing_mode: FiringMode, initial_resources = []):
        # Please mirror this in Machinations.load
        self.id = None
        self.firing_mode = firing_mode
        self.type = ElementType.ANY
        self.distribution_mode = DistributionMode.ANY 
        self.quotient = -1

        self.initial_resources = initial_resources
        self.resource_type = None

class Pool(Node):
    def __init__(self, firing_mode: FiringMode, initial_resources = []):
        super().__init__(firing_mode, initial_resources)
        self.type = ElementType.POOL
        
class Gate(Node):
    def __init__(self, firing_mode: FiringMode, distribution_mode: DistributionMode, values_or_quotient, resource_type: str):
        """Create a Gate node.

        Parameters
        ----------
        firing_mode : FiringMode
            PASSIVE / AUTOMATIC / INTERACTIVE.
        distribution_mode : DistributionMode
            DETERMINISTIC or NONDETERMINISTIC.  For *NONDETERMINISTIC* gates, the
            *values_or_quotient* argument can be either an *int* (old behaviour –
            upper-bound of a uniform integer distribution starting at 0) **or** a
            *list/tuple/np.ndarray* of numerical values from which one will be
            sampled uniformly.  For deterministic gates the argument is stored in
            *quotient* unchanged.
        values_or_quotient : int | Sequence[float]
            Behaviour depends on *distribution_mode* (see above).
        resource_type : Resource
            Single resource the gate operates on.
        """

        super().__init__(firing_mode)
        self.type = ElementType.GATE
        self.distribution_mode = distribution_mode

        if distribution_mode == DistributionMode.NONDETERMINISTIC:
            # Accept both legacy *int* and new *sequence* inputs.
            if isinstance(values_or_quotient, (list, tuple, np.ndarray)):
                self.values: list[float] = list(values_or_quotient)
                # For compatibility fill *quotient* with the number of choices – 1.
                self.quotient = len(self.values) - 1
            else:
                # Legacy: treat integer as inclusive upper-bound 0..N
                self.values = list(range(int(values_or_quotient) + 1))
                self.quotient = int(values_or_quotient)
        else:
            # Deterministic gate – keep old behaviour.
            self.quotient = values_or_quotient
            self.values = []  # unused

        self.resource_type = resource_type

class Connection:
    def __init__(self, src: Node, dst: Node | Connection):
        self.id = None
        self.src = src
        self.dst = dst
        self.predicate = None
        self.weight = 1.0

class ResourceConnection(Connection):
    def __init__(self, src: Node, dst: Node, resource_type: Resource, rate=1.0, predicate = None):
        super().__init__(src, dst)
        self.type = ElementType.RESOURCE_CONNECTION
        # TODO: Model this as an actual random variable...
        self.rate = rate
        self.resource_type = resource_type

        # Only for gates
        self.predicate = predicate

class LabelModifier(Connection):
    def __init__(self, src: Node, dst: Connection, resource_type: Resource, rate=1.0):
        super().__init__(src, dst)
        self.type = ElementType.LABEL_MODIFIER
        self.rate = rate
        self.resource_type = resource_type

class NodeModifier(Connection):
    def __init__(self, src: Node, dst: Node, src_resource_type: Resource, dst_resource_type: Resource, rate=1.0):
        super().__init__(src, dst)
        self.type = ElementType.NODE_MODIFIER
        self.rate = rate
        self.src_resource_type = src_resource_type
        self.dst_resource_type = dst_resource_type

class Trigger(Connection):
    def __init__(self, src: Node, dst: Node | Connection, predicate = None, weight = 1.0):
        super().__init__(src, dst)
        self.type = ElementType.TRIGGER

        # Only for gates
        self.predicate = predicate
        self.weight = weight

        self.dst_type = dst.type
        assert self.dst_type in [ElementType.RESOURCE_CONNECTION, ElementType.POOL, ElementType.GATE]

class Activator(Connection):
    def __init__(self, src: Node, dst: Node, predicate: Predicate, resource_type: Resource):
        super().__init__(src, dst)
        self.type = ElementType.ACTIVATOR
        self.predicate = predicate
        self.resource_type = resource_type

Diagram = Tuple[List[Node], List[Connection], List[Resource]]
