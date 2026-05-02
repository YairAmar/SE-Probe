from typing import Dict, Union, List
import torch

__all__ = ["pool_muse_activations", "pool_muse_activations_mean", "select_first_segment"]


def pool_muse_activations(
    activations: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Pool MUSE activations by averaging over the temporal dimension.
    Result shape: (C, F) where C=channels, F=frequency bins.

    Uses first-segment only pooling (hardcoded) to match the old repo behavior
    where frequency bins are the "tokens" for CKA.

    Args:
        activations: Dict of activation tensors from MUSE model.
    """
    pooled_activations = {}
    for k, v in activations.items():
        if v.ndim == 5:
            # Shape: (S, B, C, T, F) - S=segments, B=batch, C=channels, T=time, F=freq
            # Only use first segment (hardcoded)
            if v.shape[0] > 1:
                v = v[:1]
            v = v.mean(dim=[0, 1, 3])  # Mean over S, B, T -> (C, F)
        else:
            # Shape: (B, C, T, F) - B=batch, C=channels, T=time, F=freq
            v = v.mean(dim=[0, 2])     # Mean over B, T -> (C, F)
        pooled_activations[k] = v
    return pooled_activations


def pool_muse_activations_mean(
    activations: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Pool MUSE activations for reverb: average over all segments except the last.
    Result shape: (C, F) where C=channels, F=frequency bins.

    For reverb analysis, we exclude the last segment because it may contain
    zero-padded content that doesn't represent real reverberant signal.

    Args:
        activations: Dict of activation tensors from MUSE model.
    """
    pooled = {}
    for k, v in activations.items():
        if v.ndim == 5:
            # Shape: (S, B, C, T, F)
            if v.shape[0] > 1:
                v = v[:-1]  # Drop last segment
            v = v.mean(dim=[0, 1, 3])  # Mean over S, B, T -> (C, F)
        else:
            # Shape: (B, C, T, F)
            v = v.mean(dim=[0, 2])  # Mean over B, T -> (C, F)
        pooled[k] = v
    return pooled


def select_first_segment(
    activations: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Select first segment from MUSE activations without temporal pooling.
    Result shape: (C, T, F) where C=channels, T=time, F=frequency bins.

    Useful for visualization where you need the full temporal dimension.

    Args:
        activations: Dict of activation tensors from MUSE model.
    """
    selected_activations = {}
    for k, v in activations.items():
        if v.ndim == 5:
            # Shape: (S, B, C, T, F) -> select first segment, squeeze batch
            v = v[0, 0]  # (C, T, F)
        else:
            # Shape: (B, C, T, F) -> squeeze batch
            v = v[0]  # (C, T, F)
        selected_activations[k] = v
    return selected_activations
