import torch
from typing import Optional, Union

from se_probe.activation_extraction import ActivationsExtractor
from se_probe.device import get_device
from se_probe.mpsenet.consts import (
    PRETRAINED_SOURCE,
    LAYERS as MPSENET_LAYERS,
)
from se_probe.mpsenet.pooling import pool_mpsenet_activations, pool_mpsenet_activations_mean, select_first_segment

__all__ = ["MPSENetE2E", "remap_trainer_keys", "load_mpsenet_model", "load_mpsenet_activation_extractor", "load_mpsenet_activation_extractor_reverb"]


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


# Key-name remap: the dereverberation trainer's local ``MPNet`` (``training/models/
# mpsenet_generator.py``) names its modules differently from the pretrained
# ``MPSENet`` class probed here. Mapping (trainer -> here):
#   encoder. -> dense_encoder. ; enhancer. -> TSTransformer.
#   decoder.mask_decoder. -> mask_decoder. ; decoder.phase_decoder. -> phase_decoder.
# Longer prefixes are listed first so ``decoder.*`` wins over a bare ``encoder.`` match.
_TRAINER_KEY_REMAP = (
    ("decoder.mask_decoder.", "mask_decoder."),
    ("decoder.phase_decoder.", "phase_decoder."),
    ("enhancer.", "TSTransformer."),
    ("encoder.", "dense_encoder."),
)


def remap_trainer_keys(state_dict: dict, reference_keys) -> dict:
    """Rename trainer-style state_dict keys to the pretrained ``MPSENet`` names.

    A checkpoint whose keys are already a subset of ``reference_keys`` is returned
    untouched, so pretrained-format checkpoints keep working.
    """
    reference = set(reference_keys)
    if not set(state_dict) - reference:
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        for src, dst in _TRAINER_KEY_REMAP:
            if key.startswith(src):
                key = dst + key[len(src):]
                break
        remapped[key] = value
    return remapped


def load_mpsenet_model(
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: Optional[str] = None,
) -> MPSENetE2E:
    """
    Load the MPSENet model from HuggingFace and wrap for end-to-end use.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        checkpoint_path: Path to a fine-tuned checkpoint (a dict holding a
            ``'generator'`` state_dict, as written by ``training/train_mpsenet.py``,
            or a bare state_dict). Trainer-style key names are remapped
            automatically. ``None`` uses the pretrained DNS weights.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    from MPSENet import MPSENet
    model = MPSENet.from_pretrained(PRETRAINED_SOURCE)
    model.to(device)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = state_dict.get('generator', state_dict)
        model.load_state_dict(remap_trainer_keys(state_dict, model.state_dict().keys()))
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
    checkpoint_path: Optional[str] = None,
) -> ActivationsExtractor:
    """
    Load the MPSENet activation extractor for reverb analysis.
    Uses mean pooling over all windows except the last.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        checkpoint_path: Fine-tuned checkpoint (see :func:`load_mpsenet_model`);
            ``None`` probes the pretrained DNS checkpoint.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    model = load_mpsenet_model(device=device, checkpoint_path=checkpoint_path)
    return ActivationsExtractor(
        model=model,
        relevant_layers=MPSENET_LAYERS,
        pooling_fn=pool_mpsenet_activations_mean,
    )
