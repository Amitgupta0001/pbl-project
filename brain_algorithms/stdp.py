"""
Spike-Timing Dependent Plasticity (STDP)
=========================================

STDP is a biological learning rule discovered experimentally in the late
1990s. It refines Hebb's rule by incorporating the precise timing of
spikes:

- If a pre-synaptic neuron fires BEFORE a post-synaptic neuron
  (pre → post), the synapse is STRENGTHENED (Long-Term Potentiation, LTP).
  This is causal: the pre-synaptic neuron helped cause the post-synaptic
  spike.

- If a pre-synaptic neuron fires AFTER a post-synaptic neuron
  (post → pre), the synapse is WEAKENED (Long-Term Depression, LTD).
  The pre-synaptic neuron was not contributing to the post-synaptic
  activity.

The magnitude of change depends on the time difference (Δt):
    dW = A+ * exp(-Δt / tau+)   if Δt > 0  (LTP)
    dW = -A- * exp(Δt / tau-)   if Δt < 0  (LTD)

Where Δt = t_post - t_pre.
"""

from typing import Optional

import numpy as np


class STDPNetwork:
    """A network of spiking neurons with STDP-based learning.

    Parameters
    ----------
    n_pre : int
        Number of pre-synaptic (input) neurons.
    n_post : int
        Number of post-synaptic (output) neurons.
    a_plus : float
        Maximum potentiation amplitude.
    a_minus : float
        Maximum depression amplitude.
    tau_plus : float
        Time constant for potentiation (ms).
    tau_minus : float
        Time constant for depression (ms).
    w_max : float
        Maximum synaptic weight.
    w_min : float
        Minimum synaptic weight.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_max: float = 1.0,
        w_min: float = 0.0,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max
        self.w_min = w_min

        # Initialize weights uniformly
        self.weights = np.random.uniform(0.3, 0.7, (n_pre, n_post))

        # Eligibility traces for efficient online STDP
        self.pre_trace = np.zeros(n_pre)
        self.post_trace = np.zeros(n_post)

        self.weight_history: list[np.ndarray] = []

    def stdp_window(self, delta_t: float) -> float:
        """Compute the STDP weight change for a given time difference.

        Parameters
        ----------
        delta_t : float
            Time difference t_post - t_pre (ms).

        Returns
        -------
        float
            Weight change amount.
        """
        if delta_t > 0:
            return self.a_plus * np.exp(-delta_t / self.tau_plus)
        elif delta_t < 0:
            return -self.a_minus * np.exp(delta_t / self.tau_minus)
        return 0.0

    def update_traces(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, dt: float) -> None:
        """Update eligibility traces based on current spikes.

        Parameters
        ----------
        pre_spikes : np.ndarray
            Binary spike indicators for pre-synaptic neurons.
        post_spikes : np.ndarray
            Binary spike indicators for post-synaptic neurons.
        dt : float
            Time step (ms).
        """
        # Decay traces
        self.pre_trace *= np.exp(-dt / self.tau_plus)
        self.post_trace *= np.exp(-dt / self.tau_minus)

        # Increment traces on spikes
        self.pre_trace += pre_spikes.astype(float)
        self.post_trace += post_spikes.astype(float)

    def apply_stdp(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> np.ndarray:
        """Apply STDP weight updates based on current spikes and traces.

        Parameters
        ----------
        pre_spikes : np.ndarray
            Binary spike indicators for pre-synaptic neurons.
        post_spikes : np.ndarray
            Binary spike indicators for post-synaptic neurons.

        Returns
        -------
        np.ndarray
            Weight change matrix.
        """
        dw = np.zeros_like(self.weights)

        # LTP: post-synaptic spike → strengthen synapses from recently active pre neurons
        for j in range(self.n_post):
            if post_spikes[j]:
                dw[:, j] += self.a_plus * self.pre_trace

        # LTD: pre-synaptic spike → weaken synapses to recently active post neurons
        for i in range(self.n_pre):
            if pre_spikes[i]:
                dw[i, :] -= self.a_minus * self.post_trace

        # Apply weight changes with bounds
        self.weights = np.clip(self.weights + dw, self.w_min, self.w_max)
        return dw

    def simulate(
        self,
        pre_spike_trains: np.ndarray,
        post_spike_trains: np.ndarray,
        dt: float = 1.0,
        record_weights: bool = True,
    ) -> np.ndarray:
        """Run STDP learning over given spike trains.

        Parameters
        ----------
        pre_spike_trains : np.ndarray
            Pre-synaptic spikes of shape (n_steps, n_pre).
        post_spike_trains : np.ndarray
            Post-synaptic spikes of shape (n_steps, n_post).
        dt : float
            Time step (ms).
        record_weights : bool
            Whether to record weight snapshots.

        Returns
        -------
        np.ndarray
            Final weight matrix.
        """
        n_steps = pre_spike_trains.shape[0]
        self.pre_trace = np.zeros(self.n_pre)
        self.post_trace = np.zeros(self.n_post)
        self.weight_history = []

        for t in range(n_steps):
            pre_spikes = pre_spike_trains[t]
            post_spikes = post_spike_trains[t]

            self.apply_stdp(pre_spikes, post_spikes)
            self.update_traces(pre_spikes, post_spikes, dt)

            if record_weights and t % max(1, n_steps // 100) == 0:
                self.weight_history.append(self.weights.copy())

        return self.weights.copy()

    def get_stdp_curve(
        self, dt_range: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate the STDP learning window curve.

        Parameters
        ----------
        dt_range : np.ndarray, optional
            Array of delta_t values. Defaults to [-50, 50] ms.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (delta_t values, weight changes).
        """
        if dt_range is None:
            dt_range = np.linspace(-50, 50, 1000)

        dw = np.array([self.stdp_window(dt) for dt in dt_range])
        return dt_range, dw
