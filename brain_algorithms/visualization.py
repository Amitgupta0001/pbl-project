"""
Visualization Utilities for Brain-Inspired Learning Algorithms
===============================================================

Provides plotting functions for all algorithm outputs including
weight evolution, spike rasters, SOM maps, STDP curves, and
training histories.
"""

import numpy as np

try:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_hebbian_weights(
    network: "HebbianNetwork",  # noqa: F821
    title: str = "Hebbian Network Weights",
    save_path: str | None = None,
) -> None:
    """Visualize the learned weight patterns of a Hebbian network."""
    _check_matplotlib()

    weights = network.get_receptive_fields()
    n_outputs = weights.shape[0]
    cols = min(n_outputs, 5)
    rows = (n_outputs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if n_outputs == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i in range(n_outputs):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        w = weights[i]
        side = int(np.sqrt(len(w)))
        if side * side == len(w):
            ax.imshow(w.reshape(side, side), cmap="RdBu_r", aspect="equal")
        else:
            ax.bar(range(len(w)), w, color="steelblue")
        ax.set_title(f"Neuron {i}")
        ax.axis("off")

    # Hide unused subplots
    for i in range(n_outputs, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_spike_raster(
    spike_trains: np.ndarray,
    dt: float = 0.1,
    title: str = "Spike Raster Plot",
    save_path: str | None = None,
) -> None:
    """Plot a raster plot of spike trains.

    Parameters
    ----------
    spike_trains : np.ndarray
        Boolean array of shape (n_steps, n_neurons).
    dt : float
        Time step in ms.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 4))
    n_steps, n_neurons = spike_trains.shape
    time_ms = np.arange(n_steps) * dt

    for neuron_idx in range(n_neurons):
        spike_times = time_ms[spike_trains[:, neuron_idx]]
        ax.scatter(
            spike_times,
            np.full_like(spike_times, neuron_idx),
            s=2,
            c="black",
            marker="|",
            linewidths=0.5,
        )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron Index")
    ax.set_title(title)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_membrane_potential(
    voltages: np.ndarray,
    dt: float = 0.1,
    neuron_indices: list[int] | None = None,
    title: str = "Membrane Potential",
    save_path: str | None = None,
) -> None:
    """Plot membrane potential traces for selected neurons."""
    _check_matplotlib()

    if voltages.ndim == 1:
        voltages = voltages.reshape(-1, 1)

    n_steps = voltages.shape[0]
    time_ms = np.arange(n_steps) * dt

    if neuron_indices is None:
        neuron_indices = list(range(min(5, voltages.shape[1])))

    fig, axes = plt.subplots(
        len(neuron_indices), 1, figsize=(12, 2.5 * len(neuron_indices)), sharex=True
    )
    if len(neuron_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, neuron_indices):
        ax.plot(time_ms, voltages[:, idx], linewidth=0.8, color="navy")
        ax.set_ylabel(f"Neuron {idx}\n(mV)")
        ax.axhline(y=-55, color="red", linestyle="--", alpha=0.5, label="Threshold")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_som_map(
    som: "SelfOrganizingMap",  # noqa: F821
    data: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    title: str = "Self-Organizing Map",
    save_path: str | None = None,
) -> None:
    """Visualize SOM weights and data mapping."""
    _check_matplotlib()

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    # U-Matrix
    ax1 = fig.add_subplot(gs[0])
    u_matrix = som.get_u_matrix()
    im = ax1.imshow(u_matrix, cmap="bone_r", interpolation="nearest")
    ax1.set_title("U-Matrix (Distance Map)")
    plt.colorbar(im, ax=ax1, fraction=0.046)

    # Weight visualization (first 3 features as RGB if possible)
    ax2 = fig.add_subplot(gs[1])
    if som.n_features >= 3:
        w_norm = som.weights[:, :, :3].copy()
        for c in range(3):
            w_min, w_max = w_norm[:, :, c].min(), w_norm[:, :, c].max()
            if w_max > w_min:
                w_norm[:, :, c] = (w_norm[:, :, c] - w_min) / (w_max - w_min)
        ax2.imshow(w_norm, interpolation="nearest")
    else:
        ax2.imshow(som.weights[:, :, 0], cmap="viridis", interpolation="nearest")
    ax2.set_title("Weight Vectors")

    # Data mapping
    ax3 = fig.add_subplot(gs[2])
    if data is not None:
        bmus = som.map_data(data)
        if labels is not None:
            scatter = ax3.scatter(
                bmus[:, 1], bmus[:, 0],
                c=labels, cmap="tab10", alpha=0.6, s=15, edgecolors="none"
            )
            plt.colorbar(scatter, ax=ax3, fraction=0.046)
        else:
            ax3.scatter(
                bmus[:, 1], bmus[:, 0],
                alpha=0.3, s=10, color="steelblue"
            )
    ax3.set_xlim(-0.5, som.map_width - 0.5)
    ax3.set_ylim(som.map_height - 0.5, -0.5)
    ax3.set_title("Data Mapping")
    ax3.set_xlabel("Map X")
    ax3.set_ylabel("Map Y")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_stdp_curve(
    network: "STDPNetwork",  # noqa: F821
    title: str = "STDP Learning Window",
    save_path: str | None = None,
) -> None:
    """Plot the STDP weight change curve."""
    _check_matplotlib()

    dt_range, dw = network.get_stdp_curve()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dt_range, dw, linewidth=2, color="darkred")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.fill_between(dt_range, dw, where=(dw > 0), alpha=0.3, color="green", label="LTP")
    ax.fill_between(dt_range, dw, where=(dw < 0), alpha=0.3, color="red", label="LTD")
    ax.set_xlabel("Δt = t_post - t_pre (ms)")
    ax.set_ylabel("ΔW (Weight Change)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(
    history: dict,
    title: str = "Training History",
    save_path: str | None = None,
) -> None:
    """Plot training loss and accuracy curves."""
    _check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["loss"]) + 1)

    ax1.plot(epochs, history["loss"], label="Train Loss", color="navy")
    if history.get("val_loss"):
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="coral")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, history["accuracy"], label="Train Accuracy", color="navy")
    if history.get("val_accuracy"):
        ax2.plot(epochs, history["val_accuracy"], label="Val Accuracy", color="coral")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
