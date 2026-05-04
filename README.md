# 🧠 Brain-Inspired Learning Algorithms

A comprehensive Python library implementing biologically-inspired neural learning algorithms that mimic how the human brain learns, adapts, and organizes information.

## Overview

The human brain is the most powerful learning machine known to exist. This project implements five key learning algorithms inspired by neuroscience research, built from scratch using only NumPy — no deep learning frameworks required.

| Algorithm | Biological Inspiration | Key Principle |
|-----------|----------------------|---------------|
| **Hebbian Learning** | Synaptic plasticity | "Neurons that fire together, wire together" |
| **Spiking Neural Network** | Action potentials | Discrete spike-based computation |
| **Self-Organizing Map** | Cortical topography | Topology-preserving competitive learning |
| **STDP** | Synaptic timing rules | Spike timing drives learning |
| **Backpropagation** | Layered cortical architecture | Hierarchical feature extraction |

## Installation

```bash
# Clone the repository
git clone https://github.com/Amitgupta0001/pbl-project.git
cd pbl-project

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e ".[dev]"
```

## Algorithms

### 1. Hebbian Learning

Donald Hebb's postulate (1949): *"When an axon of cell A is near enough to excite cell B and repeatedly takes part in firing it, A's efficiency in firing B is increased."*

```python
from brain_algorithms import HebbianNetwork

# Classical Hebbian learning
net = HebbianNetwork(n_inputs=10, n_outputs=3, learning_rate=0.01, rule="hebbian")
net.train(data, epochs=100)

# Oja's rule (normalized, converges to principal components)
net_oja = HebbianNetwork(n_inputs=10, n_outputs=1, rule="oja")
net_oja.train(data, epochs=100)  # Weight vector → PC1

# Competitive learning (winner-take-all)
net_comp = HebbianNetwork(n_inputs=10, n_outputs=5, rule="competitive")
net_comp.train(data, epochs=100)  # Learns cluster prototypes
```

**Learning Rules:**
- `hebbian`: Classical — strengthens co-active connections
- `oja`: Normalized Hebbian — converges to principal components (prevents unbounded growth)
- `competitive`: Winner-take-all — only most active neuron updates

### 2. Spiking Neural Network (Leaky Integrate-and-Fire)

Biological neurons communicate through discrete electrical impulses (spikes). The LIF model captures essential dynamics:

```
τ_m · dV/dt = -(V - V_rest) + R · I(t)
```

When V ≥ V_threshold → **SPIKE** → V resets

```python
from brain_algorithms import LIFNeuron, SpikingNeuralNetwork

# Single neuron simulation
neuron = LIFNeuron(tau_m=20.0, v_rest=-70.0, v_threshold=-55.0)
voltages, spikes = neuron.simulate(input_currents, dt=0.1)

# Network of connected spiking neurons
snn = SpikingNeuralNetwork(n_neurons=100, connectivity=0.3)
voltages, spike_trains = snn.simulate(external_currents, dt=0.1)
```

**Features:**
- Configurable membrane time constant, threshold, refractory period
- 80/20 excitatory/inhibitory ratio (matching biology)
- Sparse random connectivity

### 3. Self-Organizing Map (SOM)

Inspired by the topographic organization of the cerebral cortex. Similar inputs map to nearby neurons on a 2D grid.

```python
from brain_algorithms import SelfOrganizingMap

som = SelfOrganizingMap(map_height=10, map_width=10, n_features=3)
som.train(data, n_iterations=5000)

# Find Best Matching Unit for a new input
bmu = som.find_bmu(new_input)

# Visualize cluster boundaries
u_matrix = som.get_u_matrix()
```

**Features:**
- Gaussian neighborhood function with exponential decay
- Quantization error tracking
- U-Matrix computation for cluster boundary visualization

### 4. Spike-Timing Dependent Plasticity (STDP)

A biological learning rule based on the precise timing of neural spikes:

- **Pre fires BEFORE post** → Synapse strengthens (LTP)
- **Pre fires AFTER post** → Synapse weakens (LTD)

```python
from brain_algorithms import STDPNetwork

network = STDPNetwork(n_pre=100, n_post=10, a_plus=0.01, a_minus=0.012)
final_weights = network.simulate(pre_spike_trains, post_spike_trains, dt=1.0)

# Visualize the STDP learning window
dt_range, dw = network.get_stdp_curve()
```

**Features:**
- Exponential STDP learning window
- Eligibility trace-based efficient implementation
- Configurable LTP/LTD asymmetry
- Weight bounds enforcement

### 5. Backpropagation Neural Network

Multi-layer perceptron with gradient descent — inspired by the hierarchical feature extraction of the visual cortex.

```python
from brain_algorithms import NeuralNetwork

nn = NeuralNetwork(
    layer_sizes=[784, 128, 64, 10],
    activation="relu",
    learning_rate=0.01,
    momentum=0.9,
)

history = nn.train(x_train, y_train, epochs=100, batch_size=32,
                   x_val=x_val, y_val=y_val)

predictions = nn.predict(x_test)
print(nn.summary())
```

**Features:**
- Xavier/He weight initialization
- Sigmoid, tanh, ReLU activations
- Softmax output with cross-entropy loss
- Mini-batch SGD with momentum
- L2 regularization

## Running Examples

```bash
# Run individual demos
python examples/demo_hebbian.py
python examples/demo_spiking.py
python examples/demo_som.py
python examples/demo_stdp.py
python examples/demo_backprop.py
```

Each demo produces detailed console output showing the algorithm in action. Install `matplotlib` for visualizations:

```bash
pip install matplotlib
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_hebbian.py -v
```

## Project Structure

```
brain-learning-algorithms/
├── brain_algorithms/           # Core library
│   ├── __init__.py            # Package exports
│   ├── hebbian.py             # Hebbian learning (classical, Oja's, competitive)
│   ├── spiking.py             # Spiking neural network (LIF model)
│   ├── som.py                 # Self-Organizing Map (Kohonen network)
│   ├── stdp.py                # Spike-Timing Dependent Plasticity
│   ├── backprop.py            # Backpropagation neural network
│   └── visualization.py       # Plotting utilities
├── examples/                   # Demo scripts
│   ├── demo_hebbian.py
│   ├── demo_spiking.py
│   ├── demo_som.py
│   ├── demo_stdp.py
│   └── demo_backprop.py
├── tests/                      # Test suite
│   ├── test_hebbian.py
│   ├── test_spiking.py
│   ├── test_som.py
│   ├── test_stdp.py
│   └── test_backprop.py
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## How the Brain Learns — A Quick Primer

```
Sensory Input → Feature Detection → Pattern Recognition → Memory Formation
     |               |                     |                    |
  Spiking         Hebbian              SOM / STDP         Backprop-like
  Neurons         Learning           Organization         Optimization
```

1. **Spiking neurons** convert continuous stimuli into discrete spike trains
2. **Hebbian plasticity** strengthens connections between co-active neurons
3. **STDP** refines synaptic weights based on precise spike timing
4. **Self-organization** creates topographic maps for spatial representation
5. **Hierarchical processing** (analogous to backprop) extracts increasingly abstract features

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.24
- matplotlib ≥ 3.7 (optional, for visualizations)

## License

MIT License
