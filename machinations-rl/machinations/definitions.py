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

class ElementType(Enum):
    NODE = 1
    RESOURCE_CONNECTION = 2
    MODIFIER = 3
    TRIGGER = 4
    ACTIVATOR = 5

class Distribution:
    def __init__(self, resource_type: Resource, values):
        self.resource_type = resource_type
        self.values = values

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
    def __init__(self, src: Node, dst: Node | Connection, resource_type = None, predicate = None):
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
                self.predicate.id if c.predicate else -1,
                self.resource_type.id if c.resource_type else -1,
                self.dst_type.value,
            ], dtype=np.float64)

class Activator(Connection):
    def __init__(self, src: Node, dst: Node | Connection, predicate: Predicate, resource_type: Resource):
        super().__init__(src, dst)
        self.type = ElementType.ACTIVATOR
        self.predicate = predicate
        self.resource_type = resource_type

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
