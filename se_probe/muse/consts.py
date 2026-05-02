"""
MUSE model constants.

Model paths can be configured via environment variables:
- MUSE_CHECKPOINT: Path to MUSE checkpoint file
- MUSE_CONFIG: Path to MUSE config JSON file
"""
import os
from pathlib import Path

__all__ = [
    "CHECKPOINT_FILE",
    "CONFIG_FILE",
    "TRANSFORMER_BLOCKS",
    "LAYER_TYPES",
    "LAYERS",
    "NORM1_LAYERS",
    "NORM2_LAYERS",
    "FFN_PROJECT_OUT_LAYERS",
    "ARCHITECTURE_BLOCKS",
    "ARCHITECTURE_LAYERS",
]

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Model checkpoint and config paths
CHECKPOINT_FILE = os.environ.get(
    "MUSE_CHECKPOINT",
    str(PROJECT_ROOT / "pretrained_models/muse/g_best")
)
CONFIG_FILE = os.environ.get(
    "MUSE_CONFIG",
    str(PROJECT_ROOT / "pretrained_models/muse/config.json")
)

# All transformer blocks in the model
TRANSFORMER_BLOCKS = [
    "TCFTransformer.encoder_level1.mhca_blks.0.transformer_layers.0",
    "TCFTransformer.encoder_level1.mhca_blks.0.transformer_layers.1",
    "TCFTransformer.encoder_level1.mhca_blks.0.transformer_layers.2",
    "TCFTransformer.encoder_level1.mhca_blks.0.transformer_layers.3",
    "TCFTransformer.encoder_level2.mhca_blks.0.transformer_layers.0",
    "TCFTransformer.encoder_level2.mhca_blks.0.transformer_layers.1",
    "TCFTransformer.encoder_level2.mhca_blks.0.transformer_layers.2",
    "TCFTransformer.encoder_level2.mhca_blks.0.transformer_layers.3",
    "TCFTransformer.latent.mhca_blks.0.transformer_layers.0",
    "TCFTransformer.latent.mhca_blks.0.transformer_layers.1",
    "TCFTransformer.latent.mhca_blks.0.transformer_layers.2",
    "TCFTransformer.latent.mhca_blks.0.transformer_layers.3",
    "TCFTransformer.decoder_level2.mhca_blks.0.transformer_layers.0",
    "TCFTransformer.decoder_level2.mhca_blks.0.transformer_layers.1",
    "TCFTransformer.decoder_level2.mhca_blks.0.transformer_layers.2",
    "TCFTransformer.decoder_level2.mhca_blks.0.transformer_layers.3",
    "TCFTransformer.decoder_level1.mhca_blks.0.transformer_layers.0",
    "TCFTransformer.decoder_level1.mhca_blks.0.transformer_layers.1",
    "TCFTransformer.decoder_level1.mhca_blks.0.transformer_layers.2",
    "TCFTransformer.decoder_level1.mhca_blks.0.transformer_layers.3",
    "TCFTransformer.mag_refinement.mhca_blks.0.transformer_layers.0",
    "TCFTransformer.mag_refinement.mhca_blks.0.transformer_layers.1",
    "TCFTransformer.mag_refinement.mhca_blks.0.transformer_layers.2",
    "TCFTransformer.mag_refinement.mhca_blks.0.transformer_layers.3",
]

# Layer types to probe
LAYER_TYPES = ["norm1", "norm2", "ffn.project_out"]

# Generate all layer combinations
LAYERS = [f"{block}.{layer_type}" for block in TRANSFORMER_BLOCKS for layer_type in LAYER_TYPES]

# Convenience lists for filtering by type
NORM1_LAYERS = [f"{block}.norm1" for block in TRANSFORMER_BLOCKS]
NORM2_LAYERS = [f"{block}.norm2" for block in TRANSFORMER_BLOCKS]
FFN_PROJECT_OUT_LAYERS = [f"{block}.ffn.project_out" for block in TRANSFORMER_BLOCKS]

# Blocks for architecture diffusion maps (encoder/decoder only, excluding latent and refinement)
ARCHITECTURE_BLOCKS = ["encoder_level1", "encoder_level2", "decoder_level2", "decoder_level1"]

# Ordered layers for architecture diffusion maps analysis (norm1 only, encoder/decoder blocks)
# 16 layers total: 4 blocks × 4 transformer layers × norm1
ARCHITECTURE_LAYERS = [
    f"TCFTransformer.{block}.mhca_blks.0.transformer_layers.{i}.norm1"
    for block in ARCHITECTURE_BLOCKS
    for i in range(4)
]
