import numpy as np
from step_jit import step_jit
from definitions import *

class Machinations:
    def __init__(
            self,
            nodes,
            resource_connections,
            label_modifiers,
            node_modifiers,
            triggers,
            activators,
            V,
            E,
            X,
            T_e
        ):
        self.t = 0
        self.nodes = nodes
        self.resource_connections = resource_connections,
        self.label_modifiers = label_modifiers,
        self.triggers = triggers,
        self.activators = activators,
        self.V = V
        (self.E_R, self.E_T, self.E_N, self.E_G, self.E_A) = E
        self.X = X
        self.T_e = T_e
    

    @classmethod
    def load(cls, d: Diagram):
        # Load diagram definitions
        nodes, connections, resources = d

        for i, node in enumerate(nodes):
            node.id = i

        V = np.array([np.array([
                n.id,
                n.firing_mode.value,
                n.type.value,
                n.distribution_mode.value,
                n.output_mode.value,
                n.quotient,
                resource_dict[n.resource_type].id if n.resource_type else -1
            ]) for n in nodes], dtype=float)

        for i, connection in enumerate(connections):
            connection.id = i

        # Split connections
        resource_connections = list(filter(lambda c: c.type == ElementType.RESOURCE_CONNECTION, connections))
        label_modifiers = list(filter(lambda c: c.type == ElementType.LABEL_MODIFIER, connections))
        node_modifiers = list(filter(lambda c: c.type == ElementType.NODE_MODIFIER, connections))
        triggers = list(filter(lambda c: c.type == ElementType.TRIGGER, connections))
        activators = list(filter(lambda c: c.type == ElementType.ACTIVATOR, connections))
        predicates = list()
        for i, e in enumerate(list(filter(lambda c: c.predicate, resource_connections + triggers + activators))):
            e.predicate.id = i
            predicates.append(e.predicate.f)

        resource_dict = dict()
        for i, resource in enumerate(resources):
            resource_dict[resource.name] = resource
            resource.id = i

        E_R = np.array([
                np.array([
                    c.id,
                    c.src.id, # u
                    c.dst.id, # v
                    resource_dict[c.resource_type].id, # r
                    c.predicate.id if c.predicate else -1, # gates only
                    c.weight, # gates only
                ]) for c in resource_connections], dtype=float)

        E_T = np.array([
                np.array([
                    c.id,
                    c.src.id, # u
                    c.dst.id, # e'
                    resource_dict[c.resource_type].id, # r
                    c.rate,   # rate (immutable)
                ]) for c in label_modifiers], dtype=float)

        E_N = np.array([
                np.array([
                    c.id,
                    c.src.id, # u
                    c.dst.id, # v
                    resource_dict[c.resource_type].id, # r
                    c.rate,   # rate (immutable)
                ]) for c in node_modifiers], dtype=float)

        E_G = np.array([
                np.array([
                    c.id,
                    c.src.id, # u
                    c.dst.id, # v
                    c.predicate.id if c.predicate else -1, # gates only
                    c.weight, # gates only
                ]) for c in triggers], dtype=float)

        E_A = np.array([
                np.array([
                    c.id,
                    c.src.id, # u
                    c.dst.id, # v
                    c.predicate.id, # P
                    resource_dict[c.resource_type].id, # r
                ]) for c in activators], dtype=float)

        # Build the initial state for each node
        X = np.zeros((len(nodes), len(resources)), dtype=float)
        for node in nodes:
            for resource_name, amount in node.initial_resources:
                resource = resource_dict[resource_name]
                X[node.id][resource.id] = float(amount)

        # Resource connection rates are separated as they are part of the state
        T_e = np.zeros((len(resource_connections),), dtype=float)
        for i, connection in enumerate(resource_connections):
            T_e[i] = float(connection.rate)

        return cls(
                nodes,
                resource_connections,
                label_modifiers,
                node_modifiers,
                triggers,
                activators,
                V,
                (E_R, E_T, E_N, E_G, E_A),
                X,
                T_e
            )


    def step(self):
        X_new, T_e_new = step_jit(self.V, self.E_R, self.E_T, self.E_N, self.E_G, self.E_A, self.X, self.T_e)
        self.t += 1
        pass
