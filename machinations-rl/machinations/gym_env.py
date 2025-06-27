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
    reward_fn : callable, optional
        `reward = reward_fn(model)` is called *after* every simulation step to
        compute the scalar reward.  If *None*, the default reward is the total
        sum of all resources across all nodes.
    record_history : bool, optional
        Whether to record a visualisable history of the simulation.
    """

    metadata = {"render_modes": []}  # rendering handled separately via render/scene.py

    def __init__(self,
                 base_model: Machinations,
                 max_steps: int | None = None,
                 reward_fn: Callable[[Machinations], float] | None = None,
                 record_history: bool = False):
        super().__init__()
        self._base_model: Machinations = copy.deepcopy(base_model)
        self._model: Machinations | None = None

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
        # Action = binary vector deciding which interactive node(s) to fire next tick
        self.action_space = spaces.MultiBinary(self.n_interactive)

        # Observation = concatenation of X (resources) and T_e (edge rates)
        #   X shape: (num_nodes, num_resources)
        #   T_e    : (num_resource_edges,)
        X_shape = self._base_model.X.shape  # (N, R)
        Te_shape = self._base_model.T_e.shape  # (E_R,)
        low = np.zeros(X_shape, dtype=np.float64)
        high = np.full(X_shape, np.inf, dtype=np.float64)
        self._obs_low = np.concatenate([
            low.flatten(),
            np.zeros(Te_shape, dtype=np.float64)
        ])
        self._obs_high = np.concatenate([
            high.flatten(),
            np.full(Te_shape, np.inf, dtype=np.float64)
        ])
        self.observation_space = spaces.Box(
            low=self._obs_low,
            high=self._obs_high,
            dtype=np.float64
        )

        # Flag indicating whether we should record history via a Renderer.
        self._record_history_flag: bool = record_history
        self._renderer: Renderer | None = None  # created in reset()

        # Episode bookkeeping
        self._max_steps = max_steps
        self._reward_fn: Callable[[Machinations], float] = (
            reward_fn if reward_fn is not None else self._default_reward
        )
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
        if self._record_history_flag:
            self._renderer = Renderer(self._model)
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
        action = np.asarray(action, dtype=np.int8).flatten()
        if action.shape[0] != self.n_interactive:
            raise ValueError(f"Action must have length {self.n_interactive}.")

        # Push action into the model via V_pending (fire on next tick)
        # Map binary action vector onto the full V_pending array
        self._model.V_pending[:] = False
        for bit, node_id in zip(action, self._interactive_ids):
            if bit:
                self._model.V_pending[node_id] = True

        # Advance simulation by one step
        self._model.step()
        self._elapsed_steps += 1

        # ----------------------------------------------------------------
        # Check for terminal outcomes
        # ----------------------------------------------------------------
        outcome: str | None = None
        for node_id, kind in self._outcome_map.items():
            if self._model.V_active[node_id]:
                outcome = kind
                break

        terminated = False
        reward: float

        if outcome is not None:
            terminated = True
            reward = {"win": 1.0, "tie": 0.0, "lose": -1.0}[outcome]
        else:
            terminated = False  # domain-specific termination could be added here
            reward = float(self._reward_fn(self._model))

        # Log snapshot if renderer is on (capture terminal frame as well)
        if self._renderer is not None:
            self._renderer._record_snapshot(extra={
                'action': action.copy(),
                'phase': 'post',
            })

        observation = self._build_observation()
        truncated = (self._max_steps is not None) and (self._elapsed_steps >= self._max_steps)
        info: Dict[str, Any] = {"renderer": self._renderer} if self._renderer else {}
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        """Flatten X and T_e into a 1-D observation."""
        assert self._model is not None
        flat_X = self._model.X.flatten()
        obs = self._obs_buffer  # reuse buffer
        np.concatenate((flat_X, self._model.T_e), out=obs)
        return obs.copy()  # Gymnasium expects a new array each call

    @staticmethod
    def _default_reward(model: Machinations) -> float:
        """Default reward: total amount of all resources across all nodes."""
        return float(model.X.sum())

    # ------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------
    @property
    def renderer(self) -> Renderer | None:
        """Renderer that records the current episode's history (if enabled)."""
        return self._renderer 