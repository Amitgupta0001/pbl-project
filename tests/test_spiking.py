"""Tests for Spiking Neural Network."""

import numpy as np

from brain_algorithms.spiking import LIFNeuron, SpikingNeuralNetwork


class TestLIFNeuron:
    def test_resting_state(self) -> None:
        neuron = LIFNeuron()
        assert neuron.v == neuron.v_rest
        assert neuron.refractory_timer == 0.0

    def test_subthreshold_no_spike(self) -> None:
        neuron = LIFNeuron(tau_m=20.0, v_rest=-70.0, v_threshold=-55.0)
        # Very small current should not cause a spike
        spiked = neuron.step(0.1, dt=0.1)
        assert not spiked

    def test_suprathreshold_spike(self) -> None:
        neuron = LIFNeuron(tau_m=20.0, v_rest=-70.0, v_threshold=-55.0, r_membrane=10.0)
        # Large sustained current should eventually cause a spike
        spiked = False
        for _ in range(10000):
            if neuron.step(5.0, dt=0.1):
                spiked = True
                break
        assert spiked

    def test_refractory_period(self) -> None:
        neuron = LIFNeuron(refractory_period=5.0)
        # Force a spike
        while not neuron.step(10.0, dt=0.1):
            pass
        # Immediately after spike, should be in refractory
        assert neuron.refractory_timer > 0
        spiked = neuron.step(10.0, dt=0.1)
        assert not spiked

    def test_simulate(self) -> None:
        neuron = LIFNeuron()
        currents = np.ones(1000) * 3.0
        voltages, spikes = neuron.simulate(currents, dt=0.1)
        assert len(voltages) == 1000
        assert len(spikes) == 1000
        assert np.any(spikes)  # Should produce some spikes

    def test_reset(self) -> None:
        neuron = LIFNeuron()
        neuron.v = -50.0
        neuron.refractory_timer = 3.0
        neuron.spike_times = [1.0, 2.0]
        neuron.reset()
        assert neuron.v == neuron.v_rest
        assert neuron.refractory_timer == 0.0
        assert neuron.spike_times == []

    def test_higher_current_more_spikes(self) -> None:
        neuron1 = LIFNeuron()
        neuron2 = LIFNeuron()
        currents_low = np.ones(2000) * 2.0
        currents_high = np.ones(2000) * 5.0
        _, spikes_low = neuron1.simulate(currents_low, dt=0.1)
        _, spikes_high = neuron2.simulate(currents_high, dt=0.1)
        assert np.sum(spikes_high) > np.sum(spikes_low)


class TestSpikingNeuralNetwork:
    def test_initialization(self) -> None:
        snn = SpikingNeuralNetwork(n_neurons=5, connectivity=0.5)
        assert len(snn.neurons) == 5
        assert snn.weights.shape == (5, 5)
        # No self-connections
        for i in range(5):
            assert snn.weights[i, i] == 0.0

    def test_simulate(self) -> None:
        np.random.seed(42)
        snn = SpikingNeuralNetwork(n_neurons=3, connectivity=0.5)
        external = np.random.randn(500, 3) * 2.0
        voltages, spike_trains = snn.simulate(external, dt=0.1)
        assert voltages.shape == (500, 3)
        assert spike_trains.shape == (500, 3)

    def test_excitatory_inhibitory_ratio(self) -> None:
        np.random.seed(42)
        snn = SpikingNeuralNetwork(n_neurons=100, connectivity=0.3)
        nonzero = snn.weights[snn.weights != 0]
        excitatory = np.sum(nonzero > 0)
        inhibitory = np.sum(nonzero < 0)
        ratio = excitatory / (excitatory + inhibitory)
        # Should be roughly 80% excitatory
        assert 0.6 < ratio < 0.95
