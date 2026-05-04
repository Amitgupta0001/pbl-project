"""Tests for Backpropagation Neural Network."""

import numpy as np
import pytest

from brain_algorithms.backprop import NeuralNetwork


class TestNeuralNetwork:
    def test_initialization(self) -> None:
        nn = NeuralNetwork(layer_sizes=[4, 8, 3])
        assert len(nn.weights) == 2
        assert nn.weights[0].shape == (4, 8)
        assert nn.weights[1].shape == (8, 3)

    def test_invalid_layers(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            NeuralNetwork(layer_sizes=[5])

    def test_invalid_activation(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            NeuralNetwork(layer_sizes=[2, 3], activation="unknown")

    def test_forward_shape(self) -> None:
        nn = NeuralNetwork(layer_sizes=[4, 8, 3])
        x = np.random.randn(10, 4)
        pre, acts = nn.forward(x)
        assert len(acts) == 3
        assert acts[-1].shape == (10, 3)

    def test_predict(self) -> None:
        nn = NeuralNetwork(layer_sizes=[3, 5, 2])
        x = np.random.randn(5, 3)
        out = nn.predict(x)
        assert out.shape == (5, 2)
        # Softmax output should sum to 1
        np.testing.assert_array_almost_equal(out.sum(axis=1), np.ones(5))

    def test_xor_learning(self) -> None:
        np.random.seed(42)
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.zeros((4, 2))
        y[0, 0] = y[3, 0] = 1  # Class 0
        y[1, 1] = y[2, 1] = 1  # Class 1

        nn = NeuralNetwork(
            layer_sizes=[2, 16, 2],
            activation="sigmoid",
            learning_rate=0.5,
            momentum=0.9,
        )
        nn.train(x, y, epochs=1000, batch_size=4, verbose=False)
        acc = nn._accuracy(x, y)
        assert acc == 1.0, f"XOR accuracy {acc} < 1.0"

    def test_training_reduces_loss(self) -> None:
        np.random.seed(42)
        x = np.random.randn(50, 3)
        y = np.zeros((50, 2))
        y[x[:, 0] > 0, 0] = 1
        y[x[:, 0] <= 0, 1] = 1

        nn = NeuralNetwork(layer_sizes=[3, 8, 2], learning_rate=0.1)
        history = nn.train(x, y, epochs=100, batch_size=16, verbose=False)
        assert history["loss"][-1] < history["loss"][0]

    def test_summary(self) -> None:
        nn = NeuralNetwork(layer_sizes=[10, 32, 16, 5])
        s = nn.summary()
        assert "Total parameters" in s
        assert "10 → 32" in s

    def test_all_activations(self) -> None:
        for act in ["sigmoid", "tanh", "relu"]:
            nn = NeuralNetwork(layer_sizes=[3, 5, 2], activation=act)
            x = np.random.randn(5, 3)
            out = nn.predict(x)
            assert out.shape == (5, 2)
            np.testing.assert_array_almost_equal(out.sum(axis=1), np.ones(5))
