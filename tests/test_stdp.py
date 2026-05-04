"""Tests for STDP Network."""

import numpy as np

from brain_algorithms.stdp import STDPNetwork


class TestSTDPNetwork:
    def test_initialization(self) -> None:
        net = STDPNetwork(n_pre=5, n_post=3)
        assert net.weights.shape == (5, 3)
        assert np.all(net.weights >= 0.3)
        assert np.all(net.weights <= 0.7)

    def test_stdp_window_ltp(self) -> None:
        net = STDPNetwork(n_pre=1, n_post=1, a_plus=0.01)
        dw = net.stdp_window(5.0)  # pre before post
        assert dw > 0  # Should be potentiation

    def test_stdp_window_ltd(self) -> None:
        net = STDPNetwork(n_pre=1, n_post=1, a_minus=0.012)
        dw = net.stdp_window(-5.0)  # post before pre
        assert dw < 0  # Should be depression

    def test_stdp_window_zero(self) -> None:
        net = STDPNetwork(n_pre=1, n_post=1)
        dw = net.stdp_window(0.0)
        assert dw == 0.0

    def test_stdp_window_decay(self) -> None:
        net = STDPNetwork(n_pre=1, n_post=1)
        dw_near = net.stdp_window(1.0)
        dw_far = net.stdp_window(40.0)
        assert abs(dw_near) > abs(dw_far)

    def test_weight_bounds(self) -> None:
        net = STDPNetwork(n_pre=2, n_post=1, w_max=1.0, w_min=0.0)
        # Simulate many potentiation events
        pre = np.zeros((100, 2), dtype=bool)
        post = np.zeros((100, 1), dtype=bool)
        for t in range(0, 100, 5):
            pre[t, 0] = True
            if t + 2 < 100:
                post[t + 2, 0] = True
        net.simulate(pre, post, dt=1.0)
        assert np.all(net.weights <= 1.0)
        assert np.all(net.weights >= 0.0)

    def test_causal_strengthens(self) -> None:
        np.random.seed(42)
        net = STDPNetwork(n_pre=2, n_post=1, a_plus=0.005, a_minus=0.006)
        initial = net.weights.copy()

        pre = np.zeros((500, 2), dtype=bool)
        post = np.zeros((500, 1), dtype=bool)
        # Pre[0] fires before post (causal)
        for t in range(10, 500, 20):
            pre[t, 0] = True
            if t + 3 < 500:
                post[t + 3, 0] = True

        net.simulate(pre, post, dt=1.0)
        # Causal synapse should strengthen
        assert net.weights[0, 0] > initial[0, 0]

    def test_get_stdp_curve(self) -> None:
        net = STDPNetwork(n_pre=1, n_post=1)
        dt_range, dw = net.get_stdp_curve()
        assert len(dt_range) == len(dw)
        # Positive dt → positive dw
        pos_mask = dt_range > 0
        assert np.all(dw[pos_mask] > 0)
        # Negative dt → negative dw
        neg_mask = dt_range < 0
        assert np.all(dw[neg_mask] < 0)
