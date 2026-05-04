"""Tests for Self-Organizing Map."""

import numpy as np

from brain_algorithms.som import SelfOrganizingMap


class TestSelfOrganizingMap:
    def test_initialization(self) -> None:
        som = SelfOrganizingMap(map_height=5, map_width=5, n_features=3)
        assert som.weights.shape == (5, 5, 3)
        assert som.initial_sigma == 2.5

    def test_find_bmu(self) -> None:
        som = SelfOrganizingMap(map_height=3, map_width=3, n_features=2)
        # Set one neuron to exactly match the input
        som.weights[1, 2] = np.array([0.5, 0.5])
        bmu = som.find_bmu(np.array([0.5, 0.5]))
        assert bmu == (1, 2)

    def test_training_reduces_error(self) -> None:
        np.random.seed(42)
        data = np.random.rand(100, 3)
        som = SelfOrganizingMap(map_height=5, map_width=5, n_features=3)
        som.train(data, n_iterations=500, track_error=True)
        assert len(som.quantization_errors) > 1
        assert som.quantization_errors[-1] < som.quantization_errors[0]

    def test_map_data(self) -> None:
        som = SelfOrganizingMap(map_height=5, map_width=5, n_features=3)
        data = np.random.rand(10, 3)
        bmus = som.map_data(data)
        assert bmus.shape == (10, 2)
        assert np.all(bmus[:, 0] >= 0) and np.all(bmus[:, 0] < 5)
        assert np.all(bmus[:, 1] >= 0) and np.all(bmus[:, 1] < 5)

    def test_u_matrix(self) -> None:
        som = SelfOrganizingMap(map_height=4, map_width=4, n_features=2)
        u_matrix = som.get_u_matrix()
        assert u_matrix.shape == (4, 4)
        assert np.all(u_matrix >= 0)

    def test_topology_preservation(self) -> None:
        np.random.seed(42)
        # Two well-separated clusters
        c1 = np.random.randn(50, 3) * 0.3 + np.array([5, 0, 0])
        c2 = np.random.randn(50, 3) * 0.3 + np.array([0, 5, 0])
        data = np.vstack([c1, c2])
        labels = np.array([0] * 50 + [1] * 50)

        som = SelfOrganizingMap(map_height=10, map_width=10, n_features=3)
        som.train(data, n_iterations=2000)

        bmus = som.map_data(data)
        centroid_0 = bmus[labels == 0].mean(axis=0)
        centroid_1 = bmus[labels == 1].mean(axis=0)
        dist = np.linalg.norm(centroid_0 - centroid_1)
        assert dist > 2.0, f"Clusters not well separated on map: {dist}"
