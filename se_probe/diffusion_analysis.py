"""Analysis utilities for diffusion maps results.

This module provides functions for analyzing diffusion map embeddings,
computing distances, and organizing layer information for visualization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

# Representative layers for analysis (one per architectural block)
REPRESENTATIVE_LAYERS: List[str] = [
    'TCFTransformer.encoder_level1.mhca_blks.0.transformer_layers.0.norm1',
    'TCFTransformer.encoder_level2.mhca_blks.0.transformer_layers.0.norm1',
    'TCFTransformer.latent.mhca_blks.0.transformer_layers.0.norm1',
    'TCFTransformer.decoder_level2.mhca_blks.0.transformer_layers.0.norm1',
    'TCFTransformer.decoder_level1.mhca_blks.0.transformer_layers.0.norm1',
    'TCFTransformer.mag_refinement.mhca_blks.0.transformer_layers.0.norm1',
]

# Human-readable block names (corresponding to REPRESENTATIVE_LAYERS)
BLOCK_NAMES: List[str] = [
    "Enc-L1",
    "Enc-L2",
    "Latent",
    "Dec-L2",
    "Dec-L1",
    "Refinement",
]

# Architectural block ordering for encoder/decoder layers
BLOCK_ORDER: List[str] = ["Enc-L1", "Enc-L2", "Dec-L2", "Dec-L1"]


# =============================================================================
# Functions
# =============================================================================

def create_psi_column(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate psi_* columns into a single 'psi' array column.

    Args:
        df: DataFrame with columns named 'psi_0', 'psi_1', etc.

    Returns:
        DataFrame with a new 'psi' column containing concatenated arrays.
    """
    psi_columns = [col for col in df.columns if col.startswith("psi_")]
    df['psi'] = df[psi_columns].apply(
        lambda row: np.concatenate([np.atleast_1d(x) for x in row.values]),
        axis=1
    )
    return df


def compute_distance_from_ref_snr(
    df: pd.DataFrame,
    ref_snr: Optional[float] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Compute Euclidean distance from reference SNR for each row.

    For each row, computes the Euclidean distance between its psi vector
    and the psi vector of the reference SNR for the same layer.
    NaN values in psi vectors are replaced with 0.

    Args:
        df: DataFrame with 'layer', 'snr', and 'psi' columns.
        ref_snr: Reference SNR value. If None, uses max SNR in the data.
        verbose: If True, print information about the reference.

    Returns:
        DataFrame with a new 'dist' column containing distances.
    """
    if ref_snr is None:
        ref_snr = df['snr'].max()

    if verbose:
        print(f"Using SNR={ref_snr} as reference")

    # Create a reference dataframe with only ref_snr rows
    ref_subset = df[df['snr'] == ref_snr]

    if verbose:
        print(f"Found {len(ref_subset)} reference rows at SNR={ref_snr}")

    # Create reference dict indexed by layer only
    ref_dict = ref_subset.set_index(['layer'])['psi'].to_dict()

    if verbose:
        print(f"Reference dict has {len(ref_dict)} unique layer keys")

    def get_distance(row):
        layer_key = row['layer']
        if layer_key not in ref_dict:
            return np.nan
        ref_psi = np.nan_to_num(np.array(ref_dict[layer_key]), nan=0.0)
        curr_psi = np.nan_to_num(np.array(row['psi']), nan=0.0)
        return np.linalg.norm(curr_psi - ref_psi)

    df['dist'] = df.apply(get_distance, axis=1)
    return df


def compute_layer_distance_matrix(
    df: pd.DataFrame
) -> Dict[float, Dict[str, Any]]:
    """Compute pairwise Euclidean distances between all layers for each SNR.

    For each SNR value, computes a distance matrix where entry (i, j) is the
    Euclidean distance between layer i's psi vector and layer j's psi vector.

    Args:
        df: DataFrame with 'snr', 'layer', and 'psi' columns.

    Returns:
        Dictionary with SNR values as keys. Each value is a dict containing:
            - 'matrix': numpy array of shape (n_layers, n_layers) with distances
            - 'layers': list of layer names corresponding to matrix indices
    """
    snr_values = sorted(df['snr'].unique())
    layers = sorted(df['layer'].unique())

    distance_matrices = {}

    for snr in snr_values:
        # Get data for this SNR
        snr_subset = df[df['snr'] == snr]

        # Create psi dictionary for this SNR
        psi_dict = snr_subset.set_index('layer')['psi'].to_dict()

        # Initialize distance matrix
        n_layers = len(layers)
        dist_matrix = np.zeros((n_layers, n_layers))

        # Compute pairwise distances
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate(layers):
                if layer1 in psi_dict and layer2 in psi_dict:
                    psi1 = np.nan_to_num(np.array(psi_dict[layer1]), nan=0.0)
                    psi2 = np.nan_to_num(np.array(psi_dict[layer2]), nan=0.0)
                    dist_matrix[i, j] = np.linalg.norm(psi1 - psi2)
                else:
                    dist_matrix[i, j] = np.nan

        distance_matrices[snr] = {
            'matrix': dist_matrix,
            'layers': layers
        }

    return distance_matrices


def get_layer_order_key(layer_name: str) -> Tuple[int, str]:
    """Return a sort key to order layers architecturally.

    Orders layers as: encoder_level1, encoder_level2, decoder_level2, decoder_level1.
    This matches the U-Net architecture where encoder flows down and decoder flows up.

    Args:
        layer_name: Full layer name string.

    Returns:
        Tuple of (priority, layer_name) for sorting.
    """
    if 'encoder_level1' in layer_name:
        return (0, layer_name)  # enc1
    elif 'encoder_level2' in layer_name:
        return (1, layer_name)  # enc2
    elif 'decoder_level2' in layer_name:
        return (2, layer_name)  # dec2
    elif 'decoder_level1' in layer_name:
        return (3, layer_name)  # dec1
    else:
        return (4, layer_name)  # other layers
