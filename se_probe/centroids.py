"""
Shared utilities for loading and processing centroid files.

Centroids are mean activations per layer, stored as binary blobs in parquet files.
This module provides functions for loading, decoding, and averaging centroids
across different groupings (noise types, SNR levels, layers).
"""
import os
from glob import glob
from typing import List, Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "load_centroids",
    "decode_centroid",
    "decode_centroids",
    "average_centroids",
    "filter_centroids",
    "build_diffusion_result_df",
]


def load_centroids(centroids_dir: str, pattern: str = "centroid*.parquet",
                   verbose: bool = True) -> pd.DataFrame:
    """
    Load all centroid parquet files from a directory.

    Two on-disk schemas exist and are normalised to one: the per-utterance sweep
    stores ``noise_name`` / ``centroid`` (bytes), the 824-utterance re-extraction
    stores ``noise`` / ``centroid_bytes``. The returned frame always carries
    ``layer, snr, noise_name, centroid``.

    Args:
        centroids_dir: Directory containing centroid parquet files.
        pattern: Glob pattern of the files to read.

    Returns:
        DataFrame with all centroids concatenated.

    Raises:
        ValueError: If no centroid files are found.
    """
    files = sorted(glob(os.path.join(centroids_dir, pattern)))

    if not files:
        raise ValueError(f"No centroid files found in {centroids_dir}")

    if verbose:
        print(f"Found {len(files)} centroid files")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.rename(columns={"noise": "noise_name", "centroid_bytes": "centroid"})


def decode_centroid(centroid_bytes: bytes, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Decode a single centroid from bytes to numpy array.

    Args:
        centroid_bytes: Binary centroid data.
        dtype: Data type for decoding (default: float32).

    Returns:
        Numpy array of centroid values.
    """
    return np.frombuffer(centroid_bytes, dtype=dtype)


def decode_centroids(centroid_series: pd.Series, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Decode a series of centroids to a stacked numpy array.

    Args:
        centroid_series: Series of binary centroid data.
        dtype: Data type for decoding (default: float32).

    Returns:
        Numpy array of shape (n_centroids, feature_dim).
    """
    return np.vstack([np.frombuffer(c, dtype=dtype) for c in centroid_series])


def average_centroids(
    df: pd.DataFrame,
    group_by: Union[str, List[str]],
    noise_types: Optional[List[str]] = None,
    layer_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Average centroids by specified grouping columns.

    Args:
        df: DataFrame with centroid data. Must have 'centroid' column.
        group_by: Column(s) to group by (e.g., 'snr', ['layer', 'snr']).
        noise_types: If provided, filter to only these noise types.
        layer_filter: If provided, filter to only these layers.

    Returns:
        DataFrame with averaged centroids, one per group.
        Includes 'n_averaged' column with count of centroids averaged.
    """
    # Apply filters
    filtered_df = df.copy()

    if noise_types is not None:
        filtered_df = filtered_df[filtered_df['noise_name'].isin(noise_types)]

    if layer_filter is not None:
        filtered_df = filtered_df[filtered_df['layer'].isin(layer_filter)]

    if len(filtered_df) == 0:
        return pd.DataFrame()

    # Ensure group_by is a list
    if isinstance(group_by, str):
        group_by = [group_by]

    # Group and average
    averaged_rows = []
    for group_key, group in filtered_df.groupby(group_by):
        # Decode and average centroids
        centroids = decode_centroids(group['centroid'])
        avg_centroid = centroids.mean(axis=0).astype(np.float32)

        # Build result row
        # pandas >= 2 yields 1-tuples for a single-key groupby
        result = dict(zip(group_by, np.atleast_1d(group_key)))

        result['centroid'] = avg_centroid.tobytes()
        result['n_averaged'] = len(group)

        averaged_rows.append(result)

    return pd.DataFrame(averaged_rows)


def build_diffusion_result_df(
    psi: np.ndarray,
    eigenvalues: np.ndarray,
    metadata_df: pd.DataFrame,
    metadata_columns: List[str],
) -> pd.DataFrame:
    """
    Build a DataFrame with diffusion map results.

    Args:
        psi: Diffusion coordinates array of shape (n_points, n_components).
        eigenvalues: Eigenvalues array.
        metadata_df: DataFrame with metadata for each point (must be same order as psi).
        metadata_columns: Columns to include from metadata_df.

    Returns:
        DataFrame with metadata, n_components, eigenvalues, and psi_0, psi_1, ... columns.
    """
    n_points, n_components = psi.shape

    rows = []
    for i in range(n_points):
        # Start with metadata
        result = {col: metadata_df.iloc[i][col] for col in metadata_columns}

        # Add diffusion map info
        result['n_components'] = n_components
        result['eigenvalues'] = eigenvalues.tobytes()

        # Add psi coordinates
        for j in range(n_components):
            result[f'psi_{j}'] = psi[i, j]

        rows.append(result)

    return pd.DataFrame(rows)


def filter_centroids(
    df: pd.DataFrame,
    noise_types: Optional[List[str]] = None,
    layers: Optional[List[str]] = None,
    snr_range: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Filter centroids DataFrame by various criteria.

    Args:
        df: DataFrame with centroid data.
        noise_types: If provided, filter to only these noise types.
        layers: If provided, filter to only these layers.
        snr_range: If provided, filter to SNR values in (min, max) range.

    Returns:
        Filtered DataFrame.
    """
    filtered = df.copy()

    if noise_types is not None:
        filtered = filtered[filtered['noise_name'].isin(noise_types)]

    if layers is not None:
        filtered = filtered[filtered['layer'].isin(layers)]

    if snr_range is not None:
        min_snr, max_snr = snr_range
        filtered = filtered[(filtered['snr'] >= min_snr) & (filtered['snr'] <= max_snr)]

    return filtered
