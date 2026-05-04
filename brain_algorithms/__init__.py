"""
Brain-Inspired Learning Algorithms
===================================

A collection of biologically-inspired neural learning algorithms that mimic
how the human brain learns, adapts, and organizes information.

Algorithms included:
- HebbianNetwork: Classical Hebbian learning ("neurons that fire together, wire together")
- SpikingNeuralNetwork: Leaky Integrate-and-Fire neuron model
- SelfOrganizingMap: Competitive learning for topology-preserving mappings
- STDPNetwork: Spike-Timing Dependent Plasticity
- NeuralNetwork: Multi-layer perceptron with backpropagation
"""

from brain_algorithms.backprop import NeuralNetwork
from brain_algorithms.hebbian import HebbianNetwork
from brain_algorithms.som import SelfOrganizingMap
from brain_algorithms.spiking import LIFNeuron, SpikingNeuralNetwork
from brain_algorithms.stdp import STDPNetwork

__version__ = "1.0.0"
__all__ = [
    "HebbianNetwork",
    "LIFNeuron",
    "SpikingNeuralNetwork",
    "SelfOrganizingMap",
    "STDPNetwork",
    "NeuralNetwork",
]
