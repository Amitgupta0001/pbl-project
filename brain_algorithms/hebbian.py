"""
Hebbian Learning Algorithm
==========================

Implements Donald Hebb's postulate (1949): "When an axon of cell A is near
enough to excite a cell B and repeatedly or persistently takes part in firing
it, some growth process or metabolic change takes place in one or both cells
such that A's efficiency, as one of the cells firing B, is increased."

In short: "Neurons that fire together, wire together."

This module provides:
- Classical Hebbian learning
- Oja's rule (normalized Hebbian learning to prevent unbounded weight growth)
- Competitive Hebbian learning
"""


import numpy as np


class HebbianNetwork:
    """A neural network that learns using Hebbian learning rules.

    The network adjusts synaptic weights based on the correlation between
    pre-synaptic and post-synaptic activity, mimicking the biological
    process of synaptic strengthening.

    Parameters
    ----------
    n_inputs : int
        Number of input neurons.
    n_outputs : int
        Number of output neurons.
    learning_rate : float
        Learning rate (eta). Controls how fast weights change.
    rule : str
        Learning rule to use: 'hebbian', 'oja', or 'competitive'.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        learning_rate: float = 0.01,
        rule: str = "hebbian",
    ):
        if rule not in ("hebbian", "oja", "competitive"):
            raise ValueError(f"Unknown rule: {rule}. Use 'hebbian', 'oja', or 'competitive'.")

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.rule = rule

        # Initialize weights with small random values
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.1
        self.weight_history: list[np.ndarray] = []

    def activate(self, x: np.ndarray) -> np.ndarray:
        """Compute output activations given input pattern.

        Parameters
        ----------
        x : np.ndarray
            Input pattern of shape (n_inputs,) or (batch_size, n_inputs).

        Returns
        -------
        np.ndarray
            Output activations.
        """
        return x @ self.weights

    def _hebbian_update(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Classical Hebbian: dW = eta * x^T * y"""
        return self.learning_rate * np.outer(x, y)

    def _oja_update(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Oja's rule: dW = eta * (x * y^T - y^2 * W)

        This is a normalized version of Hebbian learning that converges
        to the principal component of the input data, preventing
        unbounded weight growth.
        """
        dw = np.zeros_like(self.weights)
        for j in range(self.n_outputs):
            dw[:, j] = self.learning_rate * (
                y[j] * x - y[j] ** 2 * self.weights[:, j]
            )
        return dw

    def _competitive_update(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Competitive Hebbian: only the winning (most active) neuron updates.

        This implements a winner-take-all mechanism where only the output
        neuron with the highest activation strengthens its connections.
        """
        winner = np.argmax(y)
        dw = np.zeros_like(self.weights)
        dw[:, winner] = self.learning_rate * (x - self.weights[:, winner])
        return dw

    def learn(self, x: np.ndarray) -> np.ndarray:
        """Present one input pattern and update weights.

        Parameters
        ----------
        x : np.ndarray
            Input pattern of shape (n_inputs,).

        Returns
        -------
        np.ndarray
            Output activations after processing the input.
        """
        x = np.asarray(x, dtype=np.float64)
        y = self.activate(x)

        if self.rule == "hebbian":
            dw = self._hebbian_update(x, y)
        elif self.rule == "oja":
            dw = self._oja_update(x, y)
        else:
            dw = self._competitive_update(x, y)

        self.weights += dw
        self.weight_history.append(self.weights.copy())
        return y

    def train(
        self, data: np.ndarray, epochs: int = 100, shuffle: bool = True
    ) -> list[np.ndarray]:
        """Train the network on a dataset for multiple epochs.

        Parameters
        ----------
        data : np.ndarray
            Training data of shape (n_samples, n_inputs).
        epochs : int
            Number of training epochs.
        shuffle : bool
            Whether to shuffle data each epoch.

        Returns
        -------
        list[np.ndarray]
            Weight snapshots after each epoch.
        """
        epoch_weights = []
        for _ in range(epochs):
            indices = np.arange(len(data))
            if shuffle:
                np.random.shuffle(indices)
            for i in indices:
                self.learn(data[i])
            epoch_weights.append(self.weights.copy())
        return epoch_weights

    def get_receptive_fields(self) -> np.ndarray:
        """Return weight vectors as receptive fields for visualization."""
        return self.weights.T.copy()
