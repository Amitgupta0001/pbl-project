#!/usr/bin/env python3
"""
Demo: Spike-Timing Dependent Plasticity (STDP)
================================================

Demonstrates how the precise timing of neural spikes drives
synaptic plasticity — the biological mechanism behind learning
and memory formation.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_algorithms import STDPNetwork


def main() -> None:
    np.random.seed(42)

    print("=" * 60)
    print("  STDP — Spike-Timing Dependent Plasticity Demo")
    print("=" * 60)

    # --- STDP Learning Window ---
    print("\n1. STDP Learning Window")
    network = STDPNetwork(n_pre=5, n_post=2, a_plus=0.01, a_minus=0.012)

    dt_values = [-40, -20, -10, -5, -1, 1, 5, 10, 20, 40]
    print("   Δt (ms)    ΔW (weight change)")
    print("   " + "-" * 35)
    for dt_val in dt_values:
        dw = network.stdp_window(float(dt_val))
        direction = "LTP ↑" if dw > 0 else "LTD ↓"
        bar = "█" * int(abs(dw) * 2000)
        print(f"   {dt_val:+6.1f}     {dw:+.6f}  {direction}  {bar}")

    # --- Causal Spike Pattern Learning ---
    print("\n2. Learning Causal Relationships")
    net = STDPNetwork(n_pre=3, n_post=1, a_plus=0.005, a_minus=0.006)
    initial_weights = net.weights.copy()

    # Create spike trains where pre[0] consistently fires BEFORE post[0]
    # and pre[1] fires AFTER post[0] (anti-causal)
    n_steps = 1000
    pre_trains = np.zeros((n_steps, 3), dtype=bool)
    post_trains = np.zeros((n_steps, 1), dtype=bool)

    # Neuron 0: fires 5ms before post (causal → should strengthen)
    # Neuron 1: fires 5ms after post (anti-causal → should weaken)
    # Neuron 2: fires randomly (no consistent timing → mixed)
    for t in range(50, n_steps, 50):
        post_trains[t, 0] = True
        if t - 5 >= 0:
            pre_trains[t - 5, 0] = True   # Causal
        if t + 5 < n_steps:
            pre_trains[t + 5, 1] = True   # Anti-causal
        # Random timing for neuron 2
        rand_offset = np.random.randint(-20, 20)
        rt = t + rand_offset
        if 0 <= rt < n_steps:
            pre_trains[rt, 2] = True

    final_weights = net.simulate(pre_trains, post_trains, dt=1.0)

    print("   Synapse  |  Initial  |  Final    |  Change  | Expected")
    print("   " + "-" * 58)
    expected = ["Strengthen", "Weaken", "Mixed"]
    for i in range(3):
        w0 = initial_weights[i, 0]
        wf = final_weights[i, 0]
        change = wf - w0
        direction = "↑" if change > 0 else "↓"
        print(
            f"   Pre[{i}]    |  {w0:.4f}   |  {wf:.4f}   "
                f"| {change:+.4f} {direction} | {expected[i]}"
        )

    # --- Pattern Selectivity ---
    print("\n3. Developing Pattern Selectivity")
    net2 = STDPNetwork(n_pre=8, n_post=2, a_plus=0.003, a_minus=0.0035)

    n_steps2 = 2000
    pre_trains2 = np.zeros((n_steps2, 8), dtype=bool)
    post_trains2 = np.zeros((n_steps2, 2), dtype=bool)

    # Pattern A: neurons 0-3 fire together → drive post 0
    # Pattern B: neurons 4-7 fire together → drive post 1
    for t in range(20, n_steps2, 40):
        # Pattern A
        for i in range(4):
            pre_trains2[t + i, i] = True
        post_trains2[t + 5, 0] = True

        # Pattern B (offset)
        for i in range(4):
            if t + 20 + i < n_steps2:
                pre_trains2[t + 20 + i, 4 + i] = True
        if t + 25 < n_steps2:
            post_trains2[t + 25, 1] = True

    net2.simulate(pre_trains2, post_trains2, dt=1.0)

    print("   Weight matrix (pre → post):")
    print("         Post 0    Post 1")
    for i in range(8):
        group = "A" if i < 4 else "B"
        print(
            f"   Pre[{i}] ({group}):  {net2.weights[i, 0]:.4f}    {net2.weights[i, 1]:.4f}"
        )

    print("\n" + "=" * 60)
    print("  STDP enables neurons to learn temporal correlations")
    print("  and develop selectivity for specific input patterns —")
    print("  a fundamental mechanism for memory and learning.")
    print("=" * 60)

    # Visualization
    try:
        from brain_algorithms.visualization import plot_stdp_curve

        print("\nGenerating STDP learning window plot...")
        plot_stdp_curve(network)
    except ImportError:
        print("\n(Install matplotlib for visualizations: pip install matplotlib)")


if __name__ == "__main__":
    main()
