"""Tests for Hebbian Learning algorithm."""

import numpy as np
import pytest

from brain_algorithms.hebbian import HebbianNetwork


class TestHebbianNetwork:
    def test_initialization(self) -> None:
        net = HebbianNetwork(n_inputs=5, n_outputs=3)
        assert net.weights.shape == (5, 3)
        assert net.learning_rate == 0.01
        assert net.rule == "hebbian"

    def test_invalid_rule(self) -> None:
        with pytest.raises(ValueError, match="Unknown rule"):
            HebbianNetwork(n_inputs=5, n_outputs=3, rule="invalid")

    def test_activate(self) -> None:
        net = HebbianNetwork(n_inputs=3, n_outputs=2)
        x = np.array([1.0, 0.5, -0.5])
        y = net.activate(x)
        assert y.shape == (2,)
        np.testing.assert_array_almost_equal(y, x @ net.weights)

    def test_hebbian_weights_change(self) -> None:
        net = HebbianNetwork(n_inputs=3, n_outputs=2, learning_rate=0.1, rule="hebbian")
        w_before = net.weights.copy()
        x = np.array([1.0, 1.0, 0.0])
        net.learn(x)
        assert not np.allclose(w_before, net.weights)

    def test_oja_convergence_to_pc1(self) -> None:
        np.random.seed(42)
        cov = np.array([[2.0, 1.5], [1.5, 2.0]])
        data = np.random.multivariate_normal([0, 0], cov, 500)
        data = (data - data.mean(axis=0)) / data.std(axis=0)

        net = HebbianNetwork(n_inputs=2, n_outputs=1, learning_rate=0.01, rule="oja")
        net.train(data, epochs=100)

        w = net.weights[:, 0]
        w_norm = w / np.linalg.norm(w)

        # Compare to true PC1
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(data.T))
        pc1 = eigenvectors[:, -1]

        alignment = abs(np.dot(w_norm, pc1))
        assert alignment > 0.95, f"Alignment {alignment} too low"

    def test_competitive_winner_take_all(self) -> None:
        np.random.seed(42)
        net = HebbianNetwork(n_inputs=2, n_outputs=3, learning_rate=0.05, rule="competitive")
        x = np.array([1.0, 0.0])
        w_before = net.weights.copy()
        net.learn(x)
        # Only one column should change significantly
        changes = np.sum(np.abs(net.weights - w_before), axis=0)
        n_changed = np.sum(changes > 1e-10)
        assert n_changed == 1

    def test_train_records_weights(self) -> None:
        net = HebbianNetwork(n_inputs=2, n_outputs=1, rule="hebbian")
        data = np.random.randn(10, 2)
        epoch_weights = net.train(data, epochs=5)
        assert len(epoch_weights) == 5

    def test_get_receptive_fields(self) -> None:
        net = HebbianNetwork(n_inputs=4, n_outputs=3)
        rf = net.get_receptive_fields()
        assert rf.shape == (3, 4)
