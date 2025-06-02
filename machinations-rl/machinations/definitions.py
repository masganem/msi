from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class Predicate:
    def __init__(self, f: Callable):
        self.f = f
        self.id = None

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

class OutputMode(Enum):
    ANY = -1
    CONDITIONAL = 0
    RANDOM = 1

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
        self.output_mode = OutputMode.ANY
        self.quotient = -1

        self.initial_resources = initial_resources

class Pool(Node):
    def __init__(self, firing_mode: FiringMode, initial_resources = []):
        super().__init__(firing_mode, initial_resources)
        self.type = ElementType.POOL
        
class Gate(Node):
    def __init__(self, firing_mode: FiringMode, distribution_mode: DistributionMode, output_mode: OutputMode, quotient = -1):
        super().__init__(firing_mode)
        self.type = ElementType.GATE
        self.distribution_mode = distribution_mode
        self.output_mode = output_mode
        self.quotient = quotient

class Connection:
    def __init__(self, src: Node, dst: Node | Connection):
        self.id = None
        self.src = src
        self.dst = dst
        self.predicate = None
        self.weight = 1.0

class ResourceConnection(Connection):
    def __init__(self, src: Node, dst: Node, resource_type: str, rate=1.0, predicate = None, weight = 1.0):
        super().__init__(src, dst)
        self.type = ElementType.RESOURCE_CONNECTION
        # TODO: Model this as an actual random variable...
        self.rate = rate
        self.resource_type = resource_type

        # Only for gates
        self.predicate = predicate
        self.weight = weight

class LabelModifier(Connection):
    def __init__(self, src: Node, dst: Connection, resource_type: str, rate=1.0):
        super().__init__(src, dst)
        self.type = ElementType.LABEL_MODIFIER
        self.rate = rate
        self.resource_type = resource_type

class NodeModifier(Connection):
    def __init__(self, src: Node, dst: Node, resource_type: str, rate=1.0):
        super().__init__(src, dst)
        self.type = ElementType.NODE_MODIFIER
        self.rate = rate
        self.resource_type = resource_type

class Trigger(Connection):
    def __init__(self, src: Node, dst: Node, predicate = None, weight = 1.0):
        super().__init__(src, dst)
        self.type = ElementType.TRIGGER

        # Only for gates
        self.predicate = predicate
        self.weight = weight

class Activator(Connection):
    def __init__(self, src: Node, dst: Node, predicate: Predicate):
        super().__init__(src, dst)
        self.type = ElementType.ACTIVATOR
        self.predicate = predicate

Diagram = Tuple[List[Node], List[Connection], List[Resource]]
