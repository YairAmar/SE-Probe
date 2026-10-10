"""Analysis utilities for diffusion maps results.

This module provides functions for analyzing diffusion map embeddings,
computing distances, and organizing layer information for visualization.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# =============================================================================
# 824-scale diffusion analysis (paper Sec. "A Geometric View")
# =============================================================================

#: Diffusion time of the per-block view (distances across SNR within a block).
T_PER_LAYER = 0.5
#: Diffusion time of the architecture view (distances across layers at one SNR).
T_ARCHITECTURE = 5
#: Cumulative eigenvalue-energy cutoff that sets the embedding dimension.
CUTOFF = 0.99

#: The encoder and decoder blocks whose arc lengths the paper compares.
ENCODER_BLOCKS = ["Enc-L1", "Enc-L2"]
DECODER_BLOCKS = ["Dec-L2", "Dec-L1"]


def psi_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack the ``psi_*`` columns of ``df`` into a ``(n_rows, n_components)`` array,
    with NaN (absent higher components) replaced by 0."""
    cols = sorted((c for c in df.columns if c.startswith("psi_")),
                  key=lambda c: int(c.split("_")[1]))
    return np.nan_to_num(df[cols].to_numpy(dtype=float), nan=0.0)


def embed_block(centroids: pd.DataFrame, layer: str, diffusion_time: float = T_PER_LAYER,
                cutoff: float = CUTOFF, device: str = "cpu") -> pd.DataFrame:
    """Per-block (per-layer) view: embed **all** ``(noise, snr)`` centroids of one
    layer jointly, then average the diffusion coordinates over noises per SNR.

    The order matters: averaging the centroids over noises *before* embedding is a
    different estimator and does not reproduce the published figures. Returns a
    DataFrame indexed by SNR with ``psi_*`` columns.
    """
    from se_probe.centroids import decode_centroids
    from se_probe.diffusion_maps import diffusion_map_torch

    ld = centroids[centroids["layer"] == layer].sort_values(["noise_name", "snr"]).reset_index(drop=True)
    if ld.empty:
        raise KeyError(f"layer {layer!r} absent from the centroid table")
    X = decode_centroids(ld["centroid"])
    psi = diffusion_map_torch(X, cutoff=cutoff, diffusion_time=diffusion_time, device=device)
    ld = ld.assign(**{f"psi_{i}": psi[:, i] for i in range(psi.shape[1])})
    pc = [f"psi_{i}" for i in range(psi.shape[1])]
    return ld.groupby("snr")[pc].mean().sort_index()


def block_trajectory_from_psi(psi_df: pd.DataFrame, layer: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(snrs, M)`` for one layer of a stored per-layer psi table: ``M[i]`` is the
    diffusion-coordinate vector at ``snrs[i]``, averaged over the noise environments."""
    d = psi_df[psi_df["layer"] == layer]
    if d.empty:
        raise KeyError(f"layer {layer!r} absent from the psi table")
    snrs = np.sort(d["snr"].unique())
    rows = [psi_matrix(d[d["snr"] == s]).mean(axis=0) for s in snrs]
    return snrs, np.vstack(rows)


def arc_length(points: np.ndarray, n_dims: int = 2) -> float:
    """Length of the polyline through ``points`` in their first ``n_dims`` coordinates."""
    P = np.asarray(points, float)[:, :n_dims]
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def distances_from_reference(points: np.ndarray, ref_index: int) -> np.ndarray:
    """Euclidean distance of every row of ``points`` from the row ``ref_index``."""
    P = np.asarray(points, float)
    return np.linalg.norm(P - P[ref_index], axis=1)


def block_statistics(snrs: np.ndarray, M: np.ndarray, ref_snr: Optional[float] = None) -> Dict[str, float]:
    """Spearman ordering of the distance from the reference SNR, its monotonicity,
    the arc length in the first two coordinates and the maximum distance."""
    from scipy import stats

    snrs = np.asarray(snrs, float)
    ref = int(np.argmax(snrs)) if ref_snr is None else int(np.where(snrs == ref_snr)[0][0])
    dist = distances_from_reference(M, ref)
    rho = stats.spearmanr(snrs, dist).statistic
    order = np.argsort(snrs)
    return dict(n_components=int(M.shape[1]), spearman_rho=float(rho), abs_rho=float(abs(rho)),
                monotone_strict=bool(np.all(np.diff(dist[order]) < 0)),
                arc_len_2d=arc_length(M), max_dist_from_ref=float(dist.max()))


def arc_length_ratio(per_block: pd.DataFrame, encoder=ENCODER_BLOCKS, decoder=DECODER_BLOCKS,
                     col: str = "arc_len_2d") -> Dict[str, float]:
    """Mean decoder-block over mean encoder-block arc length (the paper's ``3.03x``)."""
    t = per_block.set_index("block")
    e = float(t.loc[list(encoder), col].mean())
    d = float(t.loc[list(decoder), col].mean())
    return dict(encoder_mean=e, decoder_mean=d, ratio=d / e)


def group_distances(D: np.ndarray, group_a: Sequence[int], group_b: Sequence[int],
                    normalize: bool = True) -> Dict[str, float]:
    """Between-group and within-group mean distances of a pairwise matrix ``D``.

    With ``normalize`` the matrix is first min-max scaled to ``[0, 1]`` (the way each
    panel of the layer-distance figure is rendered; with a zero diagonal this is
    ``D / D.max()``), so values are comparable *within* a panel only.
    """
    D = np.asarray(D, float)
    M = (D - D.min()) / (D.max() - D.min()) if normalize else D
    a, b = list(group_a), list(group_b)
    between = M[np.ix_(a, b)].mean()
    within_a = M[np.ix_(a, a)][np.triu_indices(len(a), 1)].mean()
    within_b = M[np.ix_(b, b)][np.triu_indices(len(b), 1)].mean()
    return dict(between=float(between), within_a=float(within_a), within_b=float(within_b),
                between_over_within=float(between / (0.5 * (within_a + within_b))))


def architecture_distance_matrices(arch_psi: pd.DataFrame, layers: Sequence[str]) -> Dict[float, np.ndarray]:
    """``{snr: (n_layers, n_layers)}`` raw Euclidean distance matrices between the
    embedded layers of a stored architecture-view psi table, in ``layers`` order."""
    out = {}
    for snr in sorted(arch_psi["snr"].unique()):
        d = arch_psi[arch_psi["snr"] == snr].set_index("layer").loc[list(layers)]
        P = psi_matrix(d)
        out[float(snr)] = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return out


__all__ = [n for n in dir() if not n.startswith("_")]
