"""
Demucs DNS64 model loading and activation extraction.
"""

import torch
from typing import Optional, Union

from se_probe.activation_extraction import ActivationsExtractor
from se_probe.device import get_device
from se_probe.demucs.consts import LAYERS as DEMUCS_LAYERS
from se_probe.demucs.pooling import pool_demucs_activations

__all__ = ["load_demucs_model", "load_demucs_activation_extractor", "load_demucs_activation_extractor_reverb"]


class DemucsE2E:
    """
    End-to-end wrapper for Demucs that accepts (1, T) or (T,) waveform tensors.

    Proxies attribute access to the underlying Demucs model so that
    named_modules() returns layer names without a wrapper prefix.
    """

    def __init__(self, model):
        object.__setattr__(self, '_model', model)

    def __call__(self, audio):
        # Demucs expects [B, T] or [B, C, T]; keep batch dim intact
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return self._model(audio)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_model'), name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_model'), name, value)


def load_demucs_model(
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: Optional[str] = None,
) -> DemucsE2E:
    """
    Load the Demucs DNS64 model and wrap for end-to-end use.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        checkpoint_path: Path to a fine-tuned checkpoint (a dict holding a
            ``'generator'`` state_dict compatible with the DNS64 architecture, as
            written by ``training/train_demucs.py``). ``None`` uses the pretrained
            DNS64 weights.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    from denoiser import pretrained
    model = pretrained.dns64()
    model.to(device)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict.get('generator', state_dict))
    return DemucsE2E(model)


def load_demucs_activation_extractor(
    device: Optional[Union[torch.device, str]] = None,
    with_pooling: bool = True,
) -> ActivationsExtractor:
    """
    Load the Demucs activation extractor.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        with_pooling: If True, pool activations for CKA (output shape: T, C).
                     If False, return raw activations.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    model = load_demucs_model(device=device)
    pooling_fn = pool_demucs_activations if with_pooling else None
    return ActivationsExtractor(
        model=model,
        relevant_layers=DEMUCS_LAYERS,
        pooling_fn=pooling_fn,
    )


def load_demucs_activation_extractor_reverb(
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: Optional[str] = None,
) -> ActivationsExtractor:
    """
    Load the Demucs DNS64 activation extractor for reverb analysis.

    Mirrors the MP-SENet/MUSE reverb loaders. Demucs has no separate mean-pooled
    reverb pooler, so it uses the standard [T, C] pooling (time frames as tokens,
    channels as features) for CKA.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        checkpoint_path: Fine-tuned checkpoint (see :func:`load_demucs_model`);
            ``None`` probes the pretrained DNS64 checkpoint.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    model = load_demucs_model(device=device, checkpoint_path=checkpoint_path)
    return ActivationsExtractor(
        model=model,
        relevant_layers=DEMUCS_LAYERS,
        pooling_fn=pool_demucs_activations,
    )
