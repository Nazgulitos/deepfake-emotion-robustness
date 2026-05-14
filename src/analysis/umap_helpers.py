"""UMAP projection helpers for Exp. 10 feature-space visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fit_umap(
    x: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
) -> np.ndarray:
    """Fit UMAP and return the 2-D embedding.

    Args:
        x: Feature matrix (n_samples, n_features).
        n_components: UMAP output dimensions.
        n_neighbors: UMAP neighbourhood size.
        min_dist: UMAP minimum distance parameter.
        metric: Distance metric.
        seed: Random seed.

    Returns:
        Embedding array (n_samples, n_components).
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install umap-learn: pip install umap-learn") from exc

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(x)


def save_umap_scatter(
    embedding: np.ndarray,
    color_col: pd.Series,
    output_path: Path,
    title: str = "UMAP",
    palette: str = "tab10",
    alpha: float = 0.6,
    point_size: int = 8,
) -> None:
    """Save a 2-D UMAP scatter coloured by color_col to output_path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    categories = color_col.astype(str).unique()
    cmap = plt.get_cmap(palette, len(categories))
    cat_to_idx = {c: i for i, c in enumerate(sorted(categories))}

    colors = [cmap(cat_to_idx[c]) for c in color_col.astype(str)]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=point_size, alpha=alpha, linewidths=0)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(cat_to_idx[c]),
                   markersize=6, label=c)
        for c in sorted(categories)
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()

    tmp = output_path.with_name(output_path.stem + ".tmp.png")
    plt.savefig(tmp, dpi=150, bbox_inches="tight")
    plt.close()
    tmp.rename(output_path)
