from typing import Dict, Union, List
import torch

__all__ = ["pool_mpsenet_activations", "pool_mpsenet_activations_mean", "select_first_segment"]


def _pool_single(name: str, act: torch.Tensor) -> torch.Tensor:
    """Pool a single activation tensor over the temporal dimension.

    Returns shape [C, F']: channels as dim-0, frequency bins as dim-1.
    This matches the MUSE convention where channels are "tokens" (rows)
    and the spatial dimension is "features" (columns) for CKA.
    """
    if act.ndim < 2:
        return act
    if "time_transformer" in name:
        # Shape: [F', T, C] -> mean over T (dim=1) -> [F', C] -> transpose -> [C, F']
        return act.mean(dim=1).T
    else:
        # freq_transformer: Shape: [T, F', C] -> mean over T (dim=0) -> [F', C] -> transpose -> [C, F']
        return act.mean(dim=0).T


def pool_mpsenet_activations(
    activations: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Pool MPSENet activations by averaging over the temporal dimension.

    For time_transformer layers with shape [F', T, C]: mean over T -> transpose -> [C, F'].
    For freq_transformer layers with shape [T, F', C]: mean over T -> transpose -> [C, F'].

    When a layer is called multiple times (list of tensors with different temporal
    sizes), each segment is pooled individually and results are averaged.

    Result shape: (C, F') — channels as dim-0, freq bins as dim-1 (matching MUSE convention).

    Args:
        activations: Dict of activation tensors from MPSENet model.
    """
    pooled = {}
    for name, act in activations.items():
        if isinstance(act, list):
            if len(act) == 1:
                pooled[name] = _pool_single(name, act[0])
            else:
                # Pool each segment separately, then average
                segments = [_pool_single(name, a) for a in act]
                pooled[name] = torch.stack(segments).mean(dim=0)
        else:
            pooled[name] = _pool_single(name, act)

    return pooled


def pool_mpsenet_activations_mean(
    activations: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Pool MPSENet activations for reverb: average over all windows except the last.

    For reverb analysis, we exclude the last window because it may contain
    zero-padded content that doesn't represent real reverberant signal.

    Args:
        activations: Dict of activation tensors from MPSENet model.
    """
    pooled = {}
    for name, act in activations.items():
        if isinstance(act, list) and len(act) > 1:
            segments = [_pool_single(name, a) for a in act[:-1]]
            pooled[name] = torch.stack(segments).mean(dim=0)
        elif isinstance(act, list):
            pooled[name] = _pool_single(name, act[0])
        else:
            pooled[name] = _pool_single(name, act)
    return pooled


def select_first_segment(
    activations: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Return activations without temporal pooling, for visualization.

    If activations are stored as lists (from multi-segment processing),
    selects the first element.

    Args:
        activations: Dict of activation tensors from MPSENet model.
    """
    selected = {}
    for name, act in activations.items():
        if isinstance(act, list):
            act = act[0]
        selected[name] = act
    return selected
