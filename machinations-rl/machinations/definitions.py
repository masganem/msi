from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class Resource:
    def __init__(self, name, unique = False):
        self.name = name
        self.unique = unique
        self.id = None

class FiringMode(Enum):
    PASSIVE = 0
    AUTOMATIC = 1
    INTERACTIVE = 2

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
        self.initial_resources = initial_resources
        self.firing_mode = firing_mode
        self.id = None
        self.type = ElementType.ANY

class Pool(Node):
    def __init__(self, firing_mode: FiringMode, initial_resources = []):
        super().__init__(firing_mode, initial_resources)
        self.type = ElementType.POOL
        
class Gate(Node):
    pass

class Connection:
    def __init__(self, src: Node, dst: Node | Connection):
        self.src = src
        self.dst = dst
        self.id = None

class ResourceConnection(Connection):
    def __init__(self, src: Node, dst: Node, resource_type: str, rate=1.0):
        super().__init__(src, dst)
        self.type = ElementType.RESOURCE_CONNECTION
        self.rate = rate
        self.resource_type = resource_type

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
    def __init__(self, src: Node, dst: Node):
        super().__init__(src, dst)
        self.type = ElementType.TRIGGER

class Predicate:
    def __init__(self, f: Callable):
        self.f = f
        self.id = None

class Activator(Connection):
    def __init__(self, src: Node, dst: Node, predicate: Predicate):
        super().__init__(src, dst)
        self.type = ElementType.ACTIVATOR
        self.predicate = predicate

Diagram = Tuple[List[Node], List[Connection], List[Resource]]
