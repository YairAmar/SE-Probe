import json
import os
from pathlib import Path
from typing import Optional, Union

import torch

from se_probe.activation_extraction import ActivationsExtractor
from se_probe.device import get_device
from se_probe.muse.consts import CHECKPOINT_FILE, CONFIG_FILE
from se_probe.muse.consts import LAYERS as MUSE_LAYERS
from se_probe.muse.models.generator import MUSE
from se_probe.muse.models.pooling import (
    pool_muse_activations,
    pool_muse_activations_mean,
    select_first_segment,
)

__all__ = ["load_muse_model", "load_muse_activation_extractor", "load_muse_activation_extractor_reverb"]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


def load_muse_model(
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: str = None,
) -> torch.nn.Module:
    """
    Load the MUSE model from checkpoint.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        checkpoint_path: Path to a specific checkpoint file. If None, uses default CHECKPOINT_FILE.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device

    with open(CONFIG_FILE) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    # Always use single_segment_mode=True (first segment only)
    model = MUSE(h, single_segment_mode=True)
    model = model.to(device)

    ckpt = checkpoint_path if checkpoint_path is not None else CHECKPOINT_FILE
    state_dict = load_checkpoint(Path(ckpt), device)
    model.load_state_dict(state_dict['generator'])

    return model


def load_muse_activation_extractor(
    device: Optional[Union[torch.device, str]] = None,
    with_pooling: bool = True,
    checkpoint_path: str = None,
) -> ActivationsExtractor:
    """
    Load the MUSE activation extractor.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        with_pooling: If True, use first-segment pooling for CKA (output shape: C, F).
                     If False, select first segment without pooling (output shape: C, T, F).
        checkpoint_path: Path to a specific checkpoint file. If None, uses default.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    model = load_muse_model(device=device, checkpoint_path=checkpoint_path)
    pooling_fn = pool_muse_activations if with_pooling else select_first_segment
    return ActivationsExtractor(model=model, relevant_layers=MUSE_LAYERS, pooling_fn=pooling_fn)


def load_muse_activation_extractor_reverb(
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: str = None,
) -> ActivationsExtractor:
    """
    Load the MUSE activation extractor for reverb analysis.
    Uses mean pooling over all segments except the last.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        checkpoint_path: Path to a specific checkpoint file. If None, uses default.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    model = load_muse_model(device=device, checkpoint_path=checkpoint_path)
    return ActivationsExtractor(model=model, relevant_layers=MUSE_LAYERS, pooling_fn=pool_muse_activations_mean)
