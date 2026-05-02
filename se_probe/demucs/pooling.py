"""
Demucs-specific activation pooling functions.

Demucs activations have two shapes:
  - Conv layers: [B, C, T] -> squeeze B -> transpose to [T, C]
  - LSTM (BLSTM): [T, B, C] -> squeeze B -> [T, C]

Result shape: (T, C) — time frames as dim-0 (tokens), channels as dim-1 (features).
Note: unlike MUSE/MPSENet which use (C, F) with channels as tokens, Demucs
uses time frames as tokens because T >> C and the gram matrix is built over
the features dimension (C×C is small, T×T would OOM).
"""

from typing import Dict, List, Union
import torch

__all__ = ["pool_demucs_activations"]


def _pool_single(name: str, act: torch.Tensor) -> torch.Tensor:
    """Pool a single Demucs activation tensor to [T, C]."""
    if act.ndim < 2:
        return act

    if 'lstm' in name:
        # LSTM output: [T, B, C] -> squeeze B -> [T, C]
        if act.ndim == 3:
            return act.squeeze(1)
        return act
    else:
        # Conv layers: [B, C, T] -> squeeze B -> [C, T] -> transpose -> [T, C]
        if act.ndim == 3:
            return act.squeeze(0).transpose(0, 1)
        elif act.ndim == 2:
            # Already [C, T] -> transpose to [T, C]
            return act.transpose(0, 1)
        return act


def pool_demucs_activations(
    activations: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Pool Demucs activations to [T, C] format for CKA computation.

    Conv layers [B, C, T]: squeeze batch, transpose to [T, C].
    LSTM layers [T, B, C]: squeeze batch to [T, C].

    Result shape: (T, C) — time frames as dim-0, channels as dim-1.

    Args:
        activations: Dict of activation tensors from Demucs model.

    Returns:
        Dict of pooled tensors, each with shape [T, C].
    """
    pooled = {}
    for name, act in activations.items():
        if isinstance(act, list):
            if len(act) == 1:
                pooled[name] = _pool_single(name, act[0])
            else:
                segments = [_pool_single(name, a) for a in act]
                pooled[name] = torch.cat(segments, dim=0)
        else:
            pooled[name] = _pool_single(name, act)

    return pooled
