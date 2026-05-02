import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable

from se_probe.device import get_device

__all__ = [
    "ActivationsExtractor",
    "get_activations",
    "extract_activations_on_audios",
    "load_muse_activation_extractor",
    "load_mpsenet_activation_extractor",
    "load_demucs_activation_extractor",
    "load_muse_activation_extractor_reverb",
    "load_mpsenet_activation_extractor_reverb",
]


def _make_activation_hook(name, chunk_activations):
    """
    Creates a hook function for forward_hook that appends output activations to the dict.
    """
    def hook(module, input, output):
        # Handle tuple outputs (some modules return tuples)
        if isinstance(output, tuple):
            if len(output) > 0:
                chunk_activations[name].append(output[0].detach())
        elif isinstance(output, torch.Tensor):
            chunk_activations[name].append(output.detach())
        else:
            try:
                if hasattr(output, 'detach'):
                    chunk_activations[name].append(output.detach())
            except Exception:
                pass
    return hook

def get_activations(
    model: torch.nn.Module,
    audio: torch.Tensor,
    target_layers: Optional[List[str]],
) -> Tuple[Dict[str, Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor]:
    """
    Extract raw activations from a single WAV file without any pooling/averaging.
    Notebook-friendly version that preserves all dimensions.

    Args:
        model: The model to extract activations from
        audio: Audio tensor of shape (1, T) or (batch, T)
        target_layers: List of layer names to extract. If None or empty, extracts ALL layers.

    Returns:
        Tuple of:
        - Dictionary of raw activations preserving all dimensions.
          Values are either a single torch.Tensor (if layer called once or all calls had same shape)
          or a List[torch.Tensor] (if layer called multiple times with different shapes).
        - Model output tensor (enhanced audio).
    """
    model.eval()
    with torch.no_grad():
        # Filter to only layers that exist in the model
        available_modules = {name: module for name, module in model.named_modules()}
        existing_layers = [name for name in target_layers if name in available_modules]

        # Register hooks only for existing layers
        chunk_activations = {name: [] for name in existing_layers}
        hook_handles = []
        for name in existing_layers:
            handle = available_modules[name].register_forward_hook(
                _make_activation_hook(name, chunk_activations)
            )
            hook_handles.append(handle)

        try:
            model_output = model(audio)
        finally:
            for handle in hook_handles:
                handle.remove()

        # Process activations (only layers that were called)
        activations = {}
        for name in existing_layers:
            if len(chunk_activations[name]) == 0:
                continue

            valid_activations = [a for a in chunk_activations[name] if a is not None]
            if len(valid_activations) == 0:
                continue

            if len(valid_activations) == 1:
                activations[name] = valid_activations[0]
            elif all(isinstance(a, torch.Tensor) for a in valid_activations):
                # Check if all tensors have the same shape before stacking
                first_shape = valid_activations[0].shape
                all_same_shape = all(a.shape == first_shape for a in valid_activations)

                if all_same_shape:
                    activations[name] = torch.stack(valid_activations, dim=0)
                else:
                    # If shapes differ, return as a list
                    activations[name] = valid_activations
            else:
                activations[name] = valid_activations[0]

    return activations, model_output

class ActivationsExtractor:
    """
    Extracts specific activations from a model given a waveform.
    Optionally applies model-specific pooling via a pooling function.

    Usage:
        # Without pooling (raw activations for visualization)
        extractor = ActivationsExtractor(model, relevant_layers)
        activations, enhanced_audio = extractor(wav_np)

        # With model-specific pooling (for CKA computation)
        from se_probe.muse.models.pooling import pool_muse_activations
        extractor = ActivationsExtractor(model, relevant_layers, pooling_fn=pool_muse_activations)
        activations, enhanced_audio = extractor(wav_np)

    Args:
        model: The model to extract activations from
        relevant_layers: List of layer names to extract activations from
        pooling_fn: Optional callable that takes a dict of activations and returns pooled activations.
                    If None, returns raw activations without pooling.
                    Signature: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] -> Dict[str, torch.Tensor]
    """
    def __init__(
        self,
        model: torch.nn.Module,
        relevant_layers: List[str],
        pooling_fn: Optional[Callable[[Dict[str, Union[torch.Tensor, List[torch.Tensor]]]], Dict[str, torch.Tensor]]] = None
    ):
        self.model = model
        self.relevant_layers = relevant_layers
        self.pooling_fn = pooling_fn

    def __call__(self, wav_np):

        target_device = next(self.model.parameters()).device
        with torch.no_grad():
            if isinstance(wav_np, np.ndarray):
                wav_tensor = torch.from_numpy(wav_np).to(target_device)
            elif isinstance(wav_np, torch.Tensor):
                wav_tensor = wav_np
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            if wav_tensor.device != target_device:
                wav_tensor = wav_tensor.to(target_device)

            # Extract raw activations and model output in single forward pass
            activations, enhanced_audio = get_activations(
                audio=wav_tensor,
                model=self.model,
                target_layers=self.relevant_layers,
            )

            # Process enhanced audio output
            if isinstance(enhanced_audio, tuple):
                enhanced_audio = enhanced_audio[0]
            if isinstance(enhanced_audio, torch.Tensor):
                enhanced_audio = enhanced_audio.squeeze().detach().cpu().numpy()

            # Apply pooling if pooling function is provided
            if self.pooling_fn is not None:
                activations = self.pooling_fn(activations)

            return activations, enhanced_audio

def extract_activations_on_audios(
    mixed_audios: List[np.ndarray],
    layer_names: List[str],
    activation_extractor,
) -> Tuple[Dict[str, torch.Tensor], List[np.ndarray]]:
    """
    Extract activations for all mixed audios (sequential processing).

    Args:
        mixed_audios: List of audio signals.
        layer_names: List of layer names to extract.
        activation_extractor: The activation extractor instance.

    Returns:
        Tuple of:
        - Dict: {layer_name: activations}, stacked on batch dimension.
        - List of enhanced audio arrays, one per input.
    """
    all_enhanced_audios: List[np.ndarray] = []
    all_wet_activations_batch: Dict[str, List[torch.Tensor]] = {}
    available_layer_names: List[str] = None

    target_device = next(activation_extractor.model.parameters()).device
    for mixed_audio in mixed_audios:
        wav_tensor = torch.from_numpy(mixed_audio).unsqueeze(0).to(target_device)
        activations, enhanced = activation_extractor(wav_tensor)

        # Determine available layers from first audio
        if available_layer_names is None:
            available_layer_names = [lname for lname in layer_names if lname in activations]
            all_wet_activations_batch = {lname: [] for lname in available_layer_names}

        # Extract enhanced audio
        enhanced_np = np.asarray(enhanced).squeeze()
        orig_len = len(mixed_audio)
        if len(enhanced_np) > orig_len:
            enhanced_np = enhanced_np[:orig_len]
        all_enhanced_audios.append(enhanced_np)

        # Collect activations - keep on GPU for efficient CKA computation
        for lname in available_layer_names:
            if lname in activations:
                all_wet_activations_batch[lname].append(activations[lname].unsqueeze(0))

        # Clean up temporary tensors after each iteration
        del wav_tensor, activations

    # Stack all activations (still on GPU)
    for lname in available_layer_names:
        all_wet_activations_batch[lname] = torch.cat(all_wet_activations_batch[lname], dim=0)

    return all_wet_activations_batch, all_enhanced_audios


def load_muse_activation_extractor(device: Optional[Union[str, torch.device]] = None, with_pooling: bool = True, checkpoint_path: str = None) -> 'ActivationsExtractor':
    """
    Load the MUSE activation extractor.

    Args:
        device: Device to load the model on. ``None`` (default) autodetects via
                :func:`se_probe.device.get_device`. Accepts a string ('cuda',
                'mps', 'cpu') or :class:`torch.device`.
        with_pooling: If True, use first-segment pooling for CKA (output shape: C, F).
                     If False, select first segment without pooling (output shape: C, T, F).
        checkpoint_path: Path to a specific checkpoint file. If None, uses default.

    Returns:
        ActivationsExtractor configured for MUSE.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    from se_probe.muse.model import load_muse_activation_extractor as _load_muse
    return _load_muse(device=device, with_pooling=with_pooling, checkpoint_path=checkpoint_path)


def load_mpsenet_activation_extractor(device: Optional[Union[str, torch.device]] = None, with_pooling: bool = True) -> 'ActivationsExtractor':
    """
    Load the MPSENet activation extractor.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        with_pooling: If True, pool activations for CKA (output shape: C, F').
                     If False, return raw activations for visualization.

    Returns:
        ActivationsExtractor configured for MPSENet.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    from se_probe.mpsenet.model import load_mpsenet_activation_extractor as _load_mpsenet
    return _load_mpsenet(device=device, with_pooling=with_pooling)


def load_demucs_activation_extractor(device: Optional[Union[str, torch.device]] = None, with_pooling: bool = True) -> 'ActivationsExtractor':
    """
    Load the Demucs DNS64 activation extractor.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        with_pooling: If True, pool activations for CKA (output shape: T, C).
                     If False, return raw activations.

    Returns:
        ActivationsExtractor configured for Demucs.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    from se_probe.demucs.model import load_demucs_activation_extractor as _load_demucs
    return _load_demucs(device=device, with_pooling=with_pooling)


def load_muse_activation_extractor_reverb(device: Optional[Union[str, torch.device]] = None, checkpoint_path: str = None) -> 'ActivationsExtractor':
    """
    Load the MUSE activation extractor for reverb analysis.
    Uses mean pooling over all segments except the last.

    Args:
        device: Device to load the model on. ``None`` autodetects.
        checkpoint_path: Path to a specific checkpoint file. If None, uses default.

    Returns:
        ActivationsExtractor configured for MUSE reverb analysis.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    from se_probe.muse.model import load_muse_activation_extractor_reverb as _load_muse_reverb
    return _load_muse_reverb(device=device, checkpoint_path=checkpoint_path)


def load_mpsenet_activation_extractor_reverb(device: Optional[Union[str, torch.device]] = None) -> 'ActivationsExtractor':
    """
    Load the MPSENet activation extractor for reverb analysis.
    Uses mean pooling over all windows except the last.

    Args:
        device: Device to load the model on. ``None`` autodetects.

    Returns:
        ActivationsExtractor configured for MPSENet reverb analysis.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    from se_probe.mpsenet.model import load_mpsenet_activation_extractor_reverb as _load_mpsenet_reverb
    return _load_mpsenet_reverb(device=device)
