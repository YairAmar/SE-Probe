import torch
from typing import Optional, Union

from se_probe.activation_extraction import ActivationsExtractor
from se_probe.device import get_device
from se_probe.mpsenet.consts import (
    PRETRAINED_SOURCE,
    LAYERS as MPSENET_LAYERS,
)
from se_probe.mpsenet.pooling import pool_mpsenet_activations, pool_mpsenet_activations_mean, select_first_segment

__all__ = ["MPSENetE2E", "load_mpsenet_model", "load_mpsenet_activation_extractor", "load_mpsenet_activation_extractor_reverb"]


class MPSENetE2E:
    """
    End-to-end wrapper for MPSENet that accepts (1, T) or (T,) waveform tensors.

    Proxies attribute access to the underlying MPSENet model so that
    named_modules() returns layer names without a wrapper prefix
    (e.g., 'TSTransformer.0.time_transformer.norm1', not 'model.TSTransformer...').
    """

    def __init__(self, model):
        object.__setattr__(self, '_model', model)

    def __call__(self, audio):
        if audio.dim() > 1:
            audio = audio.squeeze(0)
        return self._model(audio)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_model'), name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_model'), name, value)


def load_mpsenet_model(
    device: Optional[Union[torch.device, str]] = None,
) -> MPSENetE2E:
    """
    Load the MPSENet model from HuggingFace and wrap for end-to-end use.

    Args:
        device: Device to load the model on. ``None`` autodetects.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    from MPSENet import MPSENet
    model = MPSENet.from_pretrained(PRETRAINED_SOURCE)
    model.to(device)
    return MPSENetE2E(model)


def load_mpsenet_activation_extractor(
    device: Optional[Union[torch.device, str]] = None,
    with_pooling: bool = True,
) -> ActivationsExtractor:
    """
    Load the MPSENet activation extractor.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        with_pooling: If True, pool activations for CKA (output shape: F', C).
                     If False, return raw activations for visualization.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    model = load_mpsenet_model(device=device)
    pooling_fn = pool_mpsenet_activations if with_pooling else select_first_segment
    return ActivationsExtractor(
        model=model,
        relevant_layers=MPSENET_LAYERS,
        pooling_fn=pooling_fn,
    )


def load_mpsenet_activation_extractor_reverb(
    device: Optional[Union[torch.device, str]] = None,
) -> ActivationsExtractor:
    """
    Load the MPSENet activation extractor for reverb analysis.
    Uses mean pooling over all windows except the last.

    Args:
        device: Device to load the model on. ``None`` autodetects.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    model = load_mpsenet_model(device=device)
    return ActivationsExtractor(
        model=model,
        relevant_layers=MPSENET_LAYERS,
        pooling_fn=pool_mpsenet_activations_mean,
    )
