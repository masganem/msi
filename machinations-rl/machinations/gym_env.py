import copy
from typing import Any, Dict, Tuple, List, Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .machinations import Machinations
from .definitions import FiringMode, OutcomeNode
from render.renderer import Renderer

class MachinationsEnv(gym.Env):
    """Gymnasium wrapper for a *Machinations* model.

    Parameters
    ----------
    base_model : Machinations
        A fully-initialized Machinations instance representing the *initial* state
        of the simulation.  Each call to :py:meth:`reset` deep-copies this model
        so the environment can be restarted without side-effects.
    max_steps : int, optional
        Episode length limit.  If *None*, the episode never terminates on a
        timeout.
    record_history : bool, optional
        Whether to record a visualisable history of the simulation.
    max_resource_value : float, optional
        Maximum value for any resource. Values above this will be clipped.
        This helps handle infinite values and normalize the observation space.
        Defaults to 10000.0.
    X_weights : np.ndarray, optional
        Weights for each node-resource pair in X (shape: num_nodes x num_resources).
        Used to compute weighted sum of resources in reward calculation.
        If None, resources are not included in reward calculation.
    T_e_weights : np.ndarray, optional
        Weights for each resource connection rate in T_e (shape: num_resource_connections,).
        Used to compute weighted sum of connection rates in reward calculation.
        If None, connection rates are not included in reward calculation.
    """

    metadata = {"render_modes": []}  # rendering handled separately via render/scene.py

    def __init__(self,
                 base_model: Machinations,
                 max_steps: int | None = None,
                 record_history: bool = False,
                 max_resource_value: float = 10000.0):
        super().__init__()
        self._base_model: Machinations = copy.deepcopy(base_model)
        self._model: Machinations | None = None
        self._max_resource_value = max_resource_value
        self._record_history = record_history  # Can be toggled during runtime
        
        # Track which nodes and edges are visible in observations
        self._visible_node_mask = np.zeros(len(self._base_model.nodes), dtype=bool)
        self._visible_edge_mask = np.zeros(len(self._base_model.resource_connections), dtype=bool)
        
        # Build visibility masks from model elements
        for i, node in enumerate(self._base_model.nodes):
            if hasattr(node, 'visible') and node.visible:
                self._visible_node_mask[i] = True
                
        for i, conn in enumerate(self._base_model.resource_connections):
            if hasattr(conn, 'visible') and conn.visible:
                self._visible_edge_mask[i] = True
        
        # If nothing is marked visible, make everything visible by default
        if not np.any(self._visible_node_mask):
            self._visible_node_mask[:] = True
        if not np.any(self._visible_edge_mask):
            self._visible_edge_mask[:] = True
        
        # Build weight matrices from any weights specified on model elements
        self._X_weights = np.zeros_like(self._base_model.X)
        self._T_e_weights = np.zeros_like(self._base_model.T_e)
        
        # Check for node.weights[resource.id] attributes
        for node in self._base_model.nodes:
            if hasattr(node, 'weights'):
                for resource_id, weight in node.weights.items():
                    self._X_weights[node.id, resource_id] = weight
        
        # Check for connection.weight attributes
        for i, conn in enumerate(self._base_model.resource_connections):
            if hasattr(conn, 'weight'):
                self._T_e_weights[i] = conn.weight

        # Interactive nodes (actionable by the agent)
        self._interactive_ids: List[int] = [
            n.id for n in self._base_model.nodes if n.firing_mode == FiringMode.INTERACTIVE
        ]
        self.n_interactive: int = len(self._interactive_ids)

        # Outcome nodes (WIN/TIE/LOSE)
        self._outcome_map: dict[int, str] = {
            n.id: n.outcome for n in self._base_model.nodes if isinstance(n, OutcomeNode)
        }

        # ----------- Gymnasium spaces -----------
        # Action = discrete choice of which interactive node(s) to fire next tick
        # For single interactive node, this becomes 0=don't fire, 1=fire
        self.action_space = spaces.Discrete(2 ** self.n_interactive)

        # Observation = concatenation of visible X (resources) and visible T_e (edge rates)
        #   X shape: (num_visible_nodes, num_resources)
        #   T_e    : (num_visible_edges,)
        X_shape = (np.sum(self._visible_node_mask), self._base_model.X.shape[1])  # (visible_N, R)
        Te_shape = (np.sum(self._visible_edge_mask),)  # (visible_E_R,)
        
        low = np.zeros(X_shape, dtype=np.float64)
        high = np.full(X_shape, self._max_resource_value, dtype=np.float64)
        self._obs_low = np.concatenate([
            low.flatten(),
            np.zeros(Te_shape, dtype=np.float64)
        ])
        self._obs_high = np.concatenate([
            high.flatten(),
            np.full(Te_shape, self._max_resource_value, dtype=np.float64)
        ])
        self.observation_space = spaces.Box(
            low=self._obs_low,
            high=self._obs_high,
            dtype=np.float64
        )

        # Episode bookkeeping
        self._max_steps = max_steps
        self._elapsed_steps: int = 0

        # Pre-allocate observation buffer for performance
        self._obs_buffer = np.empty(self.observation_space.shape, dtype=np.float64)

    # ------------------------------------------------------------
    # Gymnasium mandatory API
    # ------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        # Reset simulation to pristine state
        self._model = copy.deepcopy(self._base_model)
        self._elapsed_steps = 0

        # Build / reset renderer
        if self._record_history:
            self._renderer = Renderer(self._model)
            # Ensure the model knows about the renderer
            self._model._renderer = self._renderer
        else:
            self._renderer = None

        observation = self._build_observation()
        info: Dict[str, Any] = {"renderer": self._renderer} if self._renderer else {}
        return observation, info

    def step(self, action):
        assert self._model is not None, "Call reset() before step()."
        # If the episode has already terminated, do nothing.
        if getattr(self._model, "terminated", False):
            observation = self._build_observation()
            reward = 0.0  # no additional reward
            terminated = True
            truncated = False
            info: Dict[str, Any] = {"renderer": self._renderer} if self._renderer else {}
            return observation, reward, terminated, truncated, info
        
        # Convert discrete action to binary vector
        action = int(action)
        binary_action = np.zeros(self.n_interactive, dtype=np.int8)
        for i in range(self.n_interactive):
            if action & (1 << i):
                binary_action[i] = 1
        
        if binary_action.shape[0] != self.n_interactive:
            raise ValueError(f"Action must have length {self.n_interactive}.")

        # Push action into the model via V_pending (fire on next tick)
        # Map binary action vector onto the full V_pending array
        self._model.V_pending[:] = False
        for bit, node_id in zip(binary_action, self._interactive_ids):
            if bit:
                self._model.V_pending[node_id] = True

        # Advance simulation by one step
        self._model.step()
        self._elapsed_steps += 1

        # ----------------------------------------------------------------
        # Check for terminal outcomes and assign rewards
        # ----------------------------------------------------------------
        outcome: str | None = None
        for node_id, kind in self._outcome_map.items():
            if self._model.V_active[node_id]:
                outcome = kind
                break

        truncated = (self._max_steps is not None) and (self._elapsed_steps >= self._max_steps)
        terminated = outcome is not None

        # Calculate reward based on outcome
        if outcome == "lose":
            # On loss, ONLY apply the loss penalty, no resource rewards at all
            reward = -1.0
            
        elif outcome == "win":
            # Calculate normal resource reward
            resource_reward = self._calculate_resource_reward()
            reward = 1.0 + resource_reward
            
        elif truncated:
            # Survived to max steps
            resource_reward = self._calculate_resource_reward()
            reward = 1.0 + resource_reward
            
        else:
            # Ongoing episode
            resource_reward = self._calculate_resource_reward()
            reward = resource_reward

        observation = self._build_observation()
        info: Dict[str, Any] = {"renderer": self._renderer} if self._renderer else {}
        return observation, reward, terminated, truncated, info

    def _calculate_resource_reward(self) -> float:
        """Helper to calculate the normal (non-loss) resource reward."""
        resource_reward = 0.0
        if self._X_weights is not None:
            X_clipped = np.clip(self._model.X, 0, self._max_resource_value)
            resource_reward += np.sum(X_clipped * self._X_weights)
        
        if self._T_e_weights is not None:
            # Use T_e from snapshot if available (includes modifier effects)
            if hasattr(self._model, "_renderer") and self._model._renderer is not None:
                last_snapshot = self._model._renderer.history[-1]
                if last_snapshot.get("phase") == "mid" and "T_e" in last_snapshot:
                    T_e_effective = last_snapshot["T_e"]
                else:
                    T_e_effective = self._model.T_e
            else:
                T_e_effective = self._model.T_e
                
            T_e_clipped = np.clip(T_e_effective, 0, self._max_resource_value)
            resource_reward += np.sum(T_e_clipped * self._T_e_weights)
        
        return resource_reward

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        """Flatten visible X and T_e into a 1-D observation and normalize values."""
        assert self._model is not None
        
        # Clip infinite values and normalize X
        X_clipped = np.clip(self._model.X, 0, self._max_resource_value)
        
        # Filter to visible nodes only
        X_visible = X_clipped[self._visible_node_mask]
        X_norm = X_visible / self._max_resource_value
        
        # Get T_e from snapshot if available (includes modifier effects)
        if hasattr(self._model, "_renderer") and self._model._renderer is not None:
            last_snapshot = self._model._renderer.history[-1]
            if last_snapshot.get("phase") == "mid" and "T_e" in last_snapshot:
                T_e_effective = last_snapshot["T_e"]
            else:
                T_e_effective = self._model.T_e
        else:
            T_e_effective = self._model.T_e
            
        T_e_clipped = np.clip(T_e_effective, 0, self._max_resource_value)
        
        # Filter to visible edges only
        T_e_visible = T_e_clipped[self._visible_edge_mask]
        T_e_norm = T_e_visible / self._max_resource_value
        
        # Combine into observation
        obs = self._obs_buffer  # reuse buffer
        np.concatenate((X_norm.flatten(), T_e_norm), out=obs)
        return obs.copy()  # Gymnasium expects a new array each call

    # ------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------
    @property
    def renderer(self) -> Renderer | None:
        """Renderer that records the current episode's history (if enabled)."""
        return self._renderer 