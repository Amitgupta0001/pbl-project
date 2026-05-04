"""
Self-Organizing Map (SOM) — Kohonen Network
============================================

Inspired by the topographic organization of the cerebral cortex, where
nearby neurons respond to similar stimuli. The Self-Organizing Map,
developed by Teuvo Kohonen (1982), creates a low-dimensional
(typically 2D) representation of high-dimensional input data while
preserving the topological properties of the input space.

The SOM learning process:
1. Present an input vector
2. Find the Best Matching Unit (BMU) — the neuron whose weight vector
   is closest to the input
3. Update the BMU and its neighbors to become more similar to the input
4. The neighborhood size and learning rate decrease over time

This mimics cortical map formation where neurons become specialized for
certain input features while maintaining spatial organization.
"""

from typing import Optional

import numpy as np


class SelfOrganizingMap:
    """A 2D Self-Organizing Map (Kohonen network).

    Parameters
    ----------
    map_height : int
        Height of the 2D neuron grid.
    map_width : int
        Width of the 2D neuron grid.
    n_features : int
        Dimensionality of input vectors.
    learning_rate : float
        Initial learning rate.
    sigma : float, optional
        Initial neighborhood radius. Defaults to max(map_height, map_width) / 2.
    """

    def __init__(
        self,
        map_height: int,
        map_width: int,
        n_features: int,
        learning_rate: float = 0.5,
        sigma: Optional[float] = None,
    ):
        self.map_height = map_height
        self.map_width = map_width
        self.n_features = n_features
        self.initial_learning_rate = learning_rate
        self.initial_sigma = sigma or max(map_height, map_width) / 2.0

        # Initialize weight vectors randomly
        self.weights = np.random.rand(map_height, map_width, n_features)

        # Precompute grid coordinates for distance calculations
        self.grid_coords = np.array(
            [
                [np.array([i, j]) for j in range(map_width)]
                for i in range(map_height)
            ]
        )

        self.quantization_errors: list[float] = []

    def find_bmu(self, x: np.ndarray) -> tuple[int, int]:
        """Find the Best Matching Unit for input vector x.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (n_features,).

        Returns
        -------
        tuple[int, int]
            (row, col) coordinates of the BMU on the map.
        """
        # Compute Euclidean distance from x to each neuron's weight vector
        diff = self.weights - x
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return int(bmu_idx[0]), int(bmu_idx[1])

    def _neighborhood(
        self, bmu: tuple[int, int], sigma: float
    ) -> np.ndarray:
        """Compute Gaussian neighborhood function centered on BMU.

        Parameters
        ----------
        bmu : tuple[int, int]
            Coordinates of the Best Matching Unit.
        sigma : float
            Current neighborhood radius.

        Returns
        -------
        np.ndarray
            Neighborhood influence of shape (map_height, map_width).
        """
        bmu_coord = np.array(bmu)
        distances_sq = np.sum(
            (self.grid_coords - bmu_coord) ** 2, axis=2
        )
        return np.exp(-distances_sq / (2 * sigma ** 2))

    def _decay(self, initial: float, iteration: int, total: int) -> float:
        """Exponential decay for learning rate and neighborhood radius."""
        return initial * np.exp(-iteration / total)

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        track_error: bool = True,
    ) -> None:
        """Train the SOM on the given data.

        Parameters
        ----------
        data : np.ndarray
            Training data of shape (n_samples, n_features).
        n_iterations : int
            Number of training iterations.
        track_error : bool
            Whether to track quantization error over training.
        """
        self.quantization_errors = []

        for iteration in range(n_iterations):
            # Decay learning rate and neighborhood radius
            lr = self._decay(self.initial_learning_rate, iteration, n_iterations)
            sigma = self._decay(self.initial_sigma, iteration, n_iterations)
            sigma = max(sigma, 0.5)  # Minimum neighborhood radius

            # Select a random input
            idx = np.random.randint(len(data))
            x = data[idx]

            # Find BMU
            bmu = self.find_bmu(x)

            # Compute neighborhood influence
            neighborhood = self._neighborhood(bmu, sigma)

            # Update weights
            for i in range(self.map_height):
                for j in range(self.map_width):
                    self.weights[i, j] += (
                        lr * neighborhood[i, j] * (x - self.weights[i, j])
                    )

            # Track quantization error
            if track_error and iteration % max(1, n_iterations // 100) == 0:
                error = self._quantization_error(data)
                self.quantization_errors.append(error)

    def _quantization_error(self, data: np.ndarray) -> float:
        """Compute mean quantization error over the dataset."""
        errors = []
        for x in data:
            bmu = self.find_bmu(x)
            errors.append(np.linalg.norm(x - self.weights[bmu[0], bmu[1]]))
        return float(np.mean(errors))

    def map_data(self, data: np.ndarray) -> np.ndarray:
        """Map each data point to its BMU coordinates.

        Parameters
        ----------
        data : np.ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            BMU coordinates of shape (n_samples, 2).
        """
        bmus = np.array([self.find_bmu(x) for x in data])
        return bmus

    def get_u_matrix(self) -> np.ndarray:
        """Compute the Unified Distance Matrix (U-Matrix).

        The U-Matrix shows the average distance between each neuron's
        weight vector and those of its neighbors. High values indicate
        cluster boundaries, low values indicate cluster interiors.

        Returns
        -------
        np.ndarray
            U-Matrix of shape (map_height, map_width).
        """
        u_matrix = np.zeros((self.map_height, self.map_width))

        for i in range(self.map_height):
            for j in range(self.map_width):
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.map_height and 0 <= nj < self.map_width:
                        neighbors.append(
                            np.linalg.norm(
                                self.weights[i, j] - self.weights[ni, nj]
                            )
                        )
                u_matrix[i, j] = np.mean(neighbors) if neighbors else 0.0

        return u_matrix
