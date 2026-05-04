#!/usr/bin/env python3
"""
Demo: Hebbian Learning
======================

Demonstrates how Hebbian learning extracts principal components from data,
mimicking how biological neurons become tuned to frequently occurring
patterns in their inputs.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_algorithms import HebbianNetwork


def main() -> None:
    np.random.seed(42)

    print("=" * 60)
    print("  Hebbian Learning — Feature Extraction Demo")
    print("=" * 60)

    # Create 2D data with a clear principal axis
    n_samples = 500
    cov = np.array([[2.0, 1.5], [1.5, 2.0]])
    data = np.random.multivariate_normal([0, 0], cov, n_samples)

    # Normalize
    data = (data - data.mean(axis=0)) / data.std(axis=0)

    # --- Classical Hebbian Learning ---
    print("\n1. Classical Hebbian Learning")
    net_hebb = HebbianNetwork(n_inputs=2, n_outputs=1, learning_rate=0.001, rule="hebbian")
    net_hebb.train(data, epochs=5)
    w = net_hebb.weights[:, 0]
    print(f"   Learned weight direction: [{w[0]:.4f}, {w[1]:.4f}]")
    print("   (Should align with principal component of data)")

    # --- Oja's Rule ---
    print("\n2. Oja's Rule (Normalized Hebbian)")
    net_oja = HebbianNetwork(n_inputs=2, n_outputs=1, learning_rate=0.01, rule="oja")
    net_oja.train(data, epochs=50)
    w_oja = net_oja.weights[:, 0]
    w_oja_norm = w_oja / np.linalg.norm(w_oja)
    print(f"   Learned weight direction: [{w_oja_norm[0]:.4f}, {w_oja_norm[1]:.4f}]")

    # Compare with actual PCA
    cov_matrix = np.cov(data.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    pc1 = eigenvectors[:, -1]  # First principal component
    print(f"   True PC1 direction:       [{pc1[0]:.4f}, {pc1[1]:.4f}]")
    alignment = abs(np.dot(w_oja_norm, pc1))
    print(f"   Alignment (cosine sim):   {alignment:.4f}")

    # --- Competitive Learning ---
    print("\n3. Competitive Hebbian Learning")
    # Create clustered data
    cluster1 = np.random.randn(100, 3) + np.array([3, 0, 0])
    cluster2 = np.random.randn(100, 3) + np.array([0, 3, 0])
    cluster3 = np.random.randn(100, 3) + np.array([0, 0, 3])
    clustered_data = np.vstack([cluster1, cluster2, cluster3])
    clustered_data = (clustered_data - clustered_data.mean(axis=0)) / clustered_data.std(axis=0)

    net_comp = HebbianNetwork(
        n_inputs=3, n_outputs=3, learning_rate=0.01, rule="competitive"
    )
    net_comp.train(clustered_data, epochs=50)
    print("   Learned prototype vectors (each row = one output neuron):")
    for i in range(3):
        w = net_comp.weights[:, i]
        print(f"   Neuron {i}: [{w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f}]")

    print("\n" + "=" * 60)
    print("  Summary: Hebbian learning successfully extracted")
    print("  meaningful features from the input data, similar")
    print("  to how biological neurons develop receptive fields.")
    print("=" * 60)

    # Visualization (if matplotlib available)
    try:
        from brain_algorithms.visualization import plot_hebbian_weights

        print("\nGenerating weight visualizations...")
        plot_hebbian_weights(net_comp, title="Competitive Learning — Learned Prototypes")
    except ImportError:
        print("\n(Install matplotlib for visualizations: pip install matplotlib)")


if __name__ == "__main__":
    main()
