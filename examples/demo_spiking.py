#!/usr/bin/env python3
"""
Demo: Spiking Neural Network
=============================

Demonstrates the Leaky Integrate-and-Fire neuron model and a small
spiking neural network, showing how biological neurons communicate
through discrete spike events.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_algorithms import LIFNeuron, SpikingNeuralNetwork


def main() -> None:
    np.random.seed(42)

    print("=" * 60)
    print("  Spiking Neural Network — LIF Neuron Demo")
    print("=" * 60)

    # --- Single Neuron Simulation ---
    print("\n1. Single LIF Neuron Response")
    neuron = LIFNeuron(tau_m=20.0, v_rest=-70.0, v_threshold=-55.0)

    # Constant current injection
    dt = 0.1
    duration = 200  # ms
    n_steps = int(duration / dt)
    currents = np.ones(n_steps) * 2.5  # Constant 2.5 nA

    voltages, spikes = neuron.simulate(currents, dt=dt)
    n_spikes = np.sum(spikes)
    firing_rate = n_spikes / (duration / 1000)

    print(f"   Stimulus: constant current 2.5 nA for {duration} ms")
    print(f"   Spikes generated: {n_spikes}")
    print(f"   Firing rate: {firing_rate:.1f} Hz")

    # Variable current
    print("\n2. Response to Variable Current")
    currents_var = np.zeros(n_steps)
    currents_var[500:800] = 1.5   # Sub-threshold
    currents_var[1000:1300] = 3.0  # Supra-threshold
    currents_var[1500:1800] = 5.0  # Strong stimulus

    neuron.reset()
    voltages_var, spikes_var = neuron.simulate(currents_var, dt=dt)

    for label, start, end in [("Low (1.5 nA)", 500, 800),
                               ("Med (3.0 nA)", 1000, 1300),
                               ("High (5.0 nA)", 1500, 1800)]:
        n = np.sum(spikes_var[start:end])
        print(f"   {label}: {n} spikes")

    # --- Network Simulation ---
    print("\n3. Spiking Neural Network (10 neurons)")
    snn = SpikingNeuralNetwork(
        n_neurons=10,
        connectivity=0.4,
        weight_scale=0.3,
        neuron_params={"tau_m": 20.0, "v_threshold": -55.0},
    )

    duration_net = 500  # ms
    n_steps_net = int(duration_net / dt)
    external = np.random.randn(n_steps_net, 10) * 1.5

    # Inject stronger stimulus to first 3 neurons
    external[:, :3] += 2.0

    voltages_net, spike_trains = snn.simulate(external, dt=dt)

    print("   Network activity summary:")
    for i in range(10):
        n_spikes_i = np.sum(spike_trains[:, i])
        rate = n_spikes_i / (duration_net / 1000)
        status = "★ stimulated" if i < 3 else ""
        print(f"   Neuron {i:2d}: {n_spikes_i:4d} spikes ({rate:6.1f} Hz) {status}")

    total_spikes = np.sum(spike_trains)
    print(f"\n   Total network spikes: {total_spikes}")
    print(f"   Mean firing rate: {total_spikes / 10 / (duration_net / 1000):.1f} Hz")

    print("\n" + "=" * 60)
    print("  The LIF model captures key properties of biological")
    print("  neurons: threshold-based firing, refractory periods,")
    print("  and membrane potential leak (forgetting).")
    print("=" * 60)

    # Visualization
    try:
        from brain_algorithms.visualization import plot_membrane_potential, plot_spike_raster

        print("\nGenerating visualizations...")
        plot_spike_raster(spike_trains, dt=dt, title="Network Spike Raster")
        plot_membrane_potential(voltages_net, dt=dt, neuron_indices=[0, 1, 5, 9])
    except ImportError:
        print("\n(Install matplotlib for visualizations: pip install matplotlib)")


if __name__ == "__main__":
    main()
