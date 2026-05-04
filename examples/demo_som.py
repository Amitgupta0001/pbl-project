#!/usr/bin/env python3
"""
Demo: Self-Organizing Map (SOM)
================================

Demonstrates how SOMs create topology-preserving maps of
high-dimensional data, similar to how the brain's cortex organizes
sensory information spatially.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_algorithms import SelfOrganizingMap


def main() -> None:
    np.random.seed(42)

    print("=" * 60)
    print("  Self-Organizing Map — Topology-Preserving Mapping")
    print("=" * 60)

    # Create color data (RGB)
    print("\n1. Color Mapping (RGB → 2D grid)")
    n_colors = 500
    colors = np.random.rand(n_colors, 3)

    # Add some pure colors to bias the mapping
    pure_colors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
    ])
    colors = np.vstack([colors, np.tile(pure_colors, (10, 1))])

    som = SelfOrganizingMap(
        map_height=10, map_width=10, n_features=3, learning_rate=0.5
    )

    print("   Training SOM (10x10 grid, 3D color space)...")
    som.train(colors, n_iterations=5000, track_error=True)

    print(f"   Final quantization error: {som.quantization_errors[-1]:.4f}")
    err_start = som.quantization_errors[0]
    err_end = som.quantization_errors[-1]
    print(f"   Error reduction: {err_start:.4f} → {err_end:.4f}")

    # Analyze U-Matrix
    u_matrix = som.get_u_matrix()
    print("\n   U-Matrix statistics:")
    print(f"   Min distance: {u_matrix.min():.4f}")
    print(f"   Max distance: {u_matrix.max():.4f}")
    print(f"   Mean distance: {u_matrix.mean():.4f}")

    # --- Clustering demo with distinct clusters ---
    print("\n2. Cluster Discovery")
    cluster_a = np.random.randn(100, 4) * 0.5 + np.array([3, 0, 0, 0])
    cluster_b = np.random.randn(100, 4) * 0.5 + np.array([0, 3, 0, 0])
    cluster_c = np.random.randn(100, 4) * 0.5 + np.array([0, 0, 3, 0])
    data = np.vstack([cluster_a, cluster_b, cluster_c])
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    som2 = SelfOrganizingMap(
        map_height=8, map_width=8, n_features=4, learning_rate=0.5
    )
    som2.train(data, n_iterations=3000)

    # Check if clusters map to distinct regions
    bmus = som2.map_data(data)
    print("   Cluster centroids on SOM grid:")
    for c in range(3):
        mask = labels == c
        centroid = bmus[mask].mean(axis=0)
        print(f"   Cluster {c}: ({centroid[0]:.1f}, {centroid[1]:.1f})")

    # Check separation
    centroids = [bmus[labels == c].mean(axis=0) for c in range(3)]
    min_dist = float("inf")
    for i in range(3):
        for j in range(i + 1, 3):
            d = np.linalg.norm(centroids[i] - centroids[j])
            min_dist = min(min_dist, d)
    print(f"   Min inter-cluster distance on map: {min_dist:.2f}")

    print("\n" + "=" * 60)
    print("  SOMs successfully organize data topologically,")
    print("  placing similar inputs near each other on the map.")
    print("  This mirrors cortical organization in the brain.")
    print("=" * 60)

    # Visualization
    try:
        from brain_algorithms.visualization import plot_som_map

        print("\nGenerating SOM visualizations...")
        plot_som_map(som, colors, title="Color SOM")
        plot_som_map(som2, data, labels, title="Cluster SOM")
    except ImportError:
        print("\n(Install matplotlib for visualizations: pip install matplotlib)")


if __name__ == "__main__":
    main()
