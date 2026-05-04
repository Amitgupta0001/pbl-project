"""
Spiking Neural Network — Leaky Integrate-and-Fire Model
========================================================

Biological neurons communicate through discrete electrical impulses called
"spikes" or "action potentials." Unlike artificial neurons that output
continuous values, spiking neurons accumulate input over time and fire
only when a voltage threshold is reached — then they reset.

This module implements the Leaky Integrate-and-Fire (LIF) neuron model,
one of the most widely used models in computational neuroscience:

    tau_m * dV/dt = -(V - V_rest) + R * I(t)

Where:
- V is the membrane potential
- V_rest is the resting potential
- tau_m is the membrane time constant
- R is the membrane resistance
- I(t) is the input current

When V >= V_threshold, the neuron fires a spike and V resets to V_reset.
"""

from typing import Optional

import numpy as np


class LIFNeuron:
    """A single Leaky Integrate-and-Fire neuron.

    Parameters
    ----------
    tau_m : float
        Membrane time constant (ms). Larger values = slower leak.
    v_rest : float
        Resting membrane potential (mV).
    v_threshold : float
        Spike threshold potential (mV).
    v_reset : float
        Reset potential after spike (mV).
    r_membrane : float
        Membrane resistance (MOhm).
    refractory_period : float
        Absolute refractory period after spike (ms).
    """

    def __init__(
        self,
        tau_m: float = 20.0,
        v_rest: float = -70.0,
        v_threshold: float = -55.0,
        v_reset: float = -75.0,
        r_membrane: float = 10.0,
        refractory_period: float = 2.0,
    ):
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.r_membrane = r_membrane
        self.refractory_period = refractory_period

        self.v = v_rest
        self.refractory_timer = 0.0
        self.spike_times: list[float] = []

    def step(self, current: float, dt: float = 0.1) -> bool:
        """Simulate one time step of the neuron.

        Parameters
        ----------
        current : float
            Input current (nA).
        dt : float
            Time step size (ms).

        Returns
        -------
        bool
            True if the neuron fired a spike this step.
        """
        spiked = False

        if self.refractory_timer > 0:
            self.refractory_timer -= dt
            self.v = self.v_reset
            return False

        # Leaky integrate
        dv = (-(self.v - self.v_rest) + self.r_membrane * current) / self.tau_m
        self.v += dv * dt

        # Check threshold
        if self.v >= self.v_threshold:
            spiked = True
            self.v = self.v_reset
            self.refractory_timer = self.refractory_period

        return spiked

    def simulate(
        self, currents: np.ndarray, dt: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the neuron over a time series of input currents.

        Parameters
        ----------
        currents : np.ndarray
            Array of input currents at each time step.
        dt : float
            Time step size (ms).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (voltages, spike_train) — membrane potential and binary spike indicators.
        """
        n_steps = len(currents)
        voltages = np.zeros(n_steps)
        spike_train = np.zeros(n_steps, dtype=bool)

        self.v = self.v_rest
        self.refractory_timer = 0.0
        self.spike_times = []

        for t in range(n_steps):
            spike_train[t] = self.step(currents[t], dt)
            voltages[t] = self.v
            if spike_train[t]:
                self.spike_times.append(t * dt)
                voltages[t] = self.v_threshold + 20  # Visual spike peak

        return voltages, spike_train

    def reset(self) -> None:
        """Reset neuron to resting state."""
        self.v = self.v_rest
        self.refractory_timer = 0.0
        self.spike_times = []


class SpikingNeuralNetwork:
    """A network of connected LIF neurons.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the network.
    connectivity : float
        Probability of connection between any two neurons (0 to 1).
    weight_scale : float
        Scale of random synaptic weights.
    neuron_params : dict, optional
        Parameters to pass to each LIFNeuron.
    """

    def __init__(
        self,
        n_neurons: int,
        connectivity: float = 0.3,
        weight_scale: float = 0.5,
        neuron_params: Optional[dict] = None,
    ):
        self.n_neurons = n_neurons
        params = neuron_params or {}
        self.neurons = [LIFNeuron(**params) for _ in range(n_neurons)]

        # Random sparse connectivity matrix
        self.weights = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and np.random.random() < connectivity:
                    # 80% excitatory, 20% inhibitory (like biological ratio)
                    sign = 1.0 if np.random.random() < 0.8 else -1.0
                    self.weights[i, j] = sign * np.random.exponential(weight_scale)

    def simulate(
        self,
        external_currents: np.ndarray,
        dt: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the entire network.

        Parameters
        ----------
        external_currents : np.ndarray
            External input currents of shape (n_steps, n_neurons).
        dt : float
            Time step size (ms).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (voltages, spike_trains) — shape (n_steps, n_neurons).
        """
        n_steps = external_currents.shape[0]
        voltages = np.zeros((n_steps, self.n_neurons))
        spike_trains = np.zeros((n_steps, self.n_neurons), dtype=bool)

        for neuron in self.neurons:
            neuron.reset()

        for t in range(n_steps):
            # Compute synaptic currents from spikes at previous step
            synaptic_input = np.zeros(self.n_neurons)
            if t > 0:
                for j in range(self.n_neurons):
                    if spike_trains[t - 1, j]:
                        synaptic_input += self.weights[j, :]

            # Total current = external + synaptic
            total_current = external_currents[t] + synaptic_input

            for i in range(self.n_neurons):
                spike_trains[t, i] = self.neurons[i].step(total_current[i], dt)
                voltages[t, i] = self.neurons[i].v
                if spike_trains[t, i]:
                    self.neurons[i].spike_times.append(t * dt)
                    voltages[t, i] = self.neurons[i].v_threshold + 20

        return voltages, spike_trains

    def get_spike_times(self) -> list[list[float]]:
        """Return spike times for each neuron."""
        return [neuron.spike_times.copy() for neuron in self.neurons]
