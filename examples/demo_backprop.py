#!/usr/bin/env python3
"""
Demo: Backpropagation Neural Network
======================================

Demonstrates a multi-layer neural network learning to classify
patterns, inspired by the layered architecture of the brain's
visual cortex.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_algorithms import NeuralNetwork


def make_spiral_data(
    n_points: int = 100, n_classes: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Generate spiral dataset — a classic non-linearly separable problem."""
    x = np.zeros((n_points * n_classes, 2))
    y = np.zeros(n_points * n_classes, dtype=int)

    for c in range(n_classes):
        idx = range(n_points * c, n_points * (c + 1))
        r = np.linspace(0.0, 1, n_points)
        theta = np.linspace(c * 4, (c + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
        x[idx] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        y[idx] = c

    return x, y


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    encoded = np.zeros((len(labels), n_classes))
    encoded[np.arange(len(labels)), labels] = 1
    return encoded


def main() -> None:
    np.random.seed(42)

    print("=" * 60)
    print("  Backpropagation Neural Network Demo")
    print("=" * 60)

    # --- XOR Problem ---
    print("\n1. XOR Problem (non-linearly separable)")
    x_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = one_hot(np.array([0, 1, 1, 0]), 2)

    nn_xor = NeuralNetwork(
        layer_sizes=[2, 8, 2],
        activation="sigmoid",
        learning_rate=0.5,
        momentum=0.9,
    )
    nn_xor.train(x_xor, y_xor, epochs=500, batch_size=4, verbose=False)

    print("   Input    | Predicted | True")
    print("   " + "-" * 35)
    predictions = nn_xor.predict(x_xor)
    for i in range(4):
        pred = np.argmax(predictions[i])
        true = np.argmax(y_xor[i])
        print(f"   {x_xor[i]}  |     {pred}     |   {true}")

    xor_acc = nn_xor._accuracy(x_xor, y_xor)
    print(f"   Accuracy: {xor_acc * 100:.0f}%")

    # --- Spiral Classification ---
    print("\n2. Spiral Classification (3 classes)")
    x_spiral, y_spiral_int = make_spiral_data(n_points=100, n_classes=3)
    y_spiral = one_hot(y_spiral_int, 3)

    # Shuffle and split
    indices = np.random.permutation(len(x_spiral))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    x_train, y_train = x_spiral[train_idx], y_spiral[train_idx]
    x_val, y_val = x_spiral[val_idx], y_spiral[val_idx]

    nn_spiral = NeuralNetwork(
        layer_sizes=[2, 64, 32, 3],
        activation="relu",
        learning_rate=0.01,
        momentum=0.9,
        l2_lambda=0.001,
    )

    print(f"\n   {nn_spiral.summary()}\n")

    history = nn_spiral.train(
        x_train, y_train,
        epochs=200,
        batch_size=32,
        x_val=x_val,
        y_val=y_val,
        verbose=True,
    )

    train_acc = nn_spiral._accuracy(x_train, y_train)
    val_acc = nn_spiral._accuracy(x_val, y_val)
    print(f"\n   Final Train Accuracy: {train_acc * 100:.1f}%")
    print(f"   Final Val Accuracy:   {val_acc * 100:.1f}%")

    # --- Binary Pattern Recognition ---
    print("\n3. Binary Pattern Recognition")
    patterns = {
        "horizontal": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float),
        "vertical": np.array([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=float),
        "diagonal": np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float),
    }

    # Create training data with noise
    x_patterns = []
    y_patterns = []
    for label_idx, (name, pattern) in enumerate(patterns.items()):
        for _ in range(50):
            noisy = pattern + np.random.randn(9) * 0.1
            x_patterns.append(noisy)
            y_patterns.append(label_idx)

    x_pat = np.array(x_patterns)
    y_pat = one_hot(np.array(y_patterns), 3)

    nn_pat = NeuralNetwork(
        layer_sizes=[9, 16, 3],
        activation="sigmoid",
        learning_rate=0.1,
    )
    nn_pat.train(x_pat, y_pat, epochs=100, batch_size=16, verbose=False)

    print("   Testing with noisy patterns:")
    for name, pattern in patterns.items():
        noisy_test = pattern + np.random.randn(9) * 0.15
        pred = nn_pat.predict(noisy_test.reshape(1, -1))
        pred_class = np.argmax(pred)
        confidence = pred[0, pred_class]
        pred_name = list(patterns.keys())[pred_class]
        print(f"   {name:12s} → predicted: {pred_name:12s} (confidence: {confidence:.2f})")

    print("\n" + "=" * 60)
    print("  Multi-layer networks with backpropagation can learn")
    print("  complex non-linear decision boundaries, inspired by")
    print("  the hierarchical feature extraction in visual cortex.")
    print("=" * 60)

    # Visualization
    try:
        from brain_algorithms.visualization import plot_training_history

        print("\nGenerating training history plots...")
        plot_training_history(history, title="Spiral Classification Training")
    except ImportError:
        print("\n(Install matplotlib for visualizations: pip install matplotlib)")


if __name__ == "__main__":
    main()
