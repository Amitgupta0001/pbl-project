"""
Backpropagation Neural Network
===============================

While backpropagation itself is not biologically plausible (the brain
doesn't propagate error signals backward through synapses), it remains
the most successful algorithm inspired by neural architecture. It
demonstrates how layered networks of simple units (neurons) can learn
complex patterns — a key insight from neuroscience.

This implementation includes:
- Configurable multi-layer architecture
- Multiple activation functions (sigmoid, tanh, ReLU)
- Mini-batch gradient descent
- L2 regularization
- Momentum-based optimization

The architecture mirrors biological neural circuits:
- Input layer: sensory neurons
- Hidden layers: interneurons performing feature extraction
- Output layer: motor/decision neurons
"""

from typing import Optional

import numpy as np


class NeuralNetwork:
    """Multi-layer neural network with backpropagation learning.

    Parameters
    ----------
    layer_sizes : list[int]
        Number of neurons in each layer, e.g., [784, 128, 64, 10].
    activation : str
        Activation function: 'sigmoid', 'tanh', or 'relu'.
    learning_rate : float
        Learning rate for gradient descent.
    momentum : float
        Momentum coefficient for SGD.
    l2_lambda : float
        L2 regularization strength.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "sigmoid",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        l2_lambda: float = 0.0001,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output layers.")
        if activation not in ("sigmoid", "tanh", "relu"):
            raise ValueError(f"Unknown activation: {activation}")

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_lambda = l2_lambda

        # Xavier/He initialization depending on activation
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.velocity_w: list[np.ndarray] = []
        self.velocity_b: list[np.ndarray] = []

        for i in range(self.n_layers - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            if activation == "relu":
                # He initialization
                scale = np.sqrt(2.0 / fan_in)
            else:
                # Xavier initialization
                scale = np.sqrt(2.0 / (fan_in + fan_out))

            self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros(fan_out))
            self.velocity_w.append(np.zeros((fan_in, fan_out)))
            self.velocity_b.append(np.zeros(fan_out))

        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

    def _activate(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_name == "tanh":
            return np.tanh(z)
        else:  # relu
            return np.maximum(0, z)

    def _activate_derivative(self, a: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function given activations."""
        if self.activation_name == "sigmoid":
            return a * (1 - a)
        elif self.activation_name == "tanh":
            return 1 - a ** 2
        else:  # relu
            return (a > 0).astype(float)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax for output layer."""
        shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Forward pass through the network.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            (pre_activations, activations) for each layer.
        """
        activations = [x]
        pre_activations = [x]

        current = x
        for i in range(self.n_layers - 1):
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)

            if i == self.n_layers - 2:
                # Softmax for output layer
                a = self._softmax(z)
            else:
                a = self._activate(z)

            activations.append(a)
            current = a

        return pre_activations, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Compute network output.

        Parameters
        ----------
        x : np.ndarray
            Input of shape (batch_size, n_inputs) or (n_inputs,).

        Returns
        -------
        np.ndarray
            Output predictions.
        """
        _, activations = self.forward(x)
        return activations[-1]

    def _cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss with L2 regularization."""
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        ce_loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

        # L2 regularization
        l2_reg = sum(np.sum(w ** 2) for w in self.weights)
        return ce_loss + 0.5 * self.l2_lambda * l2_reg

    def backward(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Backpropagation to compute gradients.

        Parameters
        ----------
        x : np.ndarray
            Input batch of shape (batch_size, n_inputs).
        y_true : np.ndarray
            True labels (one-hot) of shape (batch_size, n_outputs).

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            (weight_gradients, bias_gradients).
        """
        batch_size = x.shape[0]
        _, activations = self.forward(x)

        weight_grads = [np.zeros_like(w) for w in self.weights]
        bias_grads = [np.zeros_like(b) for b in self.biases]

        # Output layer error (softmax + cross-entropy simplification)
        delta = activations[-1] - y_true

        for i in range(self.n_layers - 2, -1, -1):
            weight_grads[i] = (
                activations[i].T @ delta / batch_size
                + self.l2_lambda * self.weights[i]
            )
            bias_grads[i] = np.mean(delta, axis=0)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self._activate_derivative(
                    activations[i]
                )

        return weight_grads, bias_grads

    def _update_weights(
        self,
        weight_grads: list[np.ndarray],
        bias_grads: list[np.ndarray],
    ) -> None:
        """Update weights using momentum SGD."""
        for i in range(len(self.weights)):
            self.velocity_w[i] = (
                self.momentum * self.velocity_w[i]
                - self.learning_rate * weight_grads[i]
            )
            self.velocity_b[i] = (
                self.momentum * self.velocity_b[i]
                - self.learning_rate * bias_grads[i]
            )
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> dict:
        """Train the network using mini-batch gradient descent.

        Parameters
        ----------
        x_train : np.ndarray
            Training inputs of shape (n_samples, n_inputs).
        y_train : np.ndarray
            Training labels (one-hot) of shape (n_samples, n_outputs).
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        x_val : np.ndarray, optional
            Validation inputs.
        y_val : np.ndarray, optional
            Validation labels.
        verbose : bool
            Whether to print training progress.

        Returns
        -------
        dict
            Training history with 'loss', 'accuracy', 'val_loss', 'val_accuracy'.
        """
        history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        n_samples = x_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                weight_grads, bias_grads = self.backward(x_batch, y_batch)
                self._update_weights(weight_grads, bias_grads)

                predictions = self.predict(x_batch)
                epoch_loss += self._cross_entropy_loss(predictions, y_batch)
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            train_acc = self._accuracy(x_train, y_train)
            history["loss"].append(avg_loss)
            history["accuracy"].append(train_acc)

            # Validation
            if x_val is not None and y_val is not None:
                val_pred = self.predict(x_val)
                val_loss = self._cross_entropy_loss(val_pred, y_val)
                val_acc = self._accuracy(x_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
            else:
                val_loss = 0.0
                val_acc = 0.0

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} — "
                    f"Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}"
                    + (
                        f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                        if x_val is not None
                        else ""
                    )
                )

        self.loss_history = history["loss"]
        self.accuracy_history = history["accuracy"]
        return history

    def _accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self.predict(x)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        return float(np.mean(pred_classes == true_classes))

    def summary(self) -> str:
        """Print a summary of the network architecture."""
        lines = ["Neural Network Summary", "=" * 40]
        total_params = 0
        for i in range(len(self.weights)):
            n_params = self.weights[i].size + self.biases[i].size
            total_params += n_params
            lines.append(
                f"Layer {i}: {self.layer_sizes[i]} → {self.layer_sizes[i + 1]} "
                f"({n_params:,} params)"
            )
        lines.append("=" * 40)
        lines.append(f"Total parameters: {total_params:,}")
        lines.append(f"Activation: {self.activation_name}")
        lines.append(f"Learning rate: {self.learning_rate}")
        return "\n".join(lines)
