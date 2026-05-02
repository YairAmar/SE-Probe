"""
MPSENet model constants and layer definitions.

Extended layer set with 56 layers (8 blocks x 7 layer types) for maximum
flexibility in activation analysis. Earlier iterations exposed only 32 layers
(norm1/norm2/norm3/attention).

Note: attention.out_proj is excluded because nn.MultiheadAttention uses
F.multi_head_attention_forward internally (functional form), so out_proj's
forward hook never fires.
"""

__all__ = [
    "PRETRAINED_SOURCE",
    "TRANSFORMER_BLOCKS",
    "LAYER_TYPES",
    "LAYERS",
    "NORM1_LAYERS",
    "NORM2_LAYERS",
    "NORM3_LAYERS",
    "ATT_LAYERS",
    "FFN_LAYERS",
    "FFN_GRU_LAYERS",
    "FFN_LINEAR_LAYERS",
]

PRETRAINED_SOURCE = 'JacobLinCool/MP-SENet-DNS'

block_nums = [0, 1, 2, 3]
block_types = ['time', 'freq']

TRANSFORMER_BLOCKS = [
    f"TSTransformer.{n}.{t}_transformer"
    for n in block_nums for t in block_types
]

LAYER_TYPES = [
    "norm1", "attention",
    "norm2", "ffn", "ffn.gru", "ffn.linear", "norm3",
]

# 8 blocks x 7 layer types = 56 layers
LAYERS = [f"{block}.{lt}" for block in TRANSFORMER_BLOCKS for lt in LAYER_TYPES]

# Convenience sublists
NORM1_LAYERS = [f"{b}.norm1" for b in TRANSFORMER_BLOCKS]
NORM2_LAYERS = [f"{b}.norm2" for b in TRANSFORMER_BLOCKS]
NORM3_LAYERS = [f"{b}.norm3" for b in TRANSFORMER_BLOCKS]
ATT_LAYERS = [f"{b}.attention" for b in TRANSFORMER_BLOCKS]
FFN_LAYERS = [f"{b}.ffn" for b in TRANSFORMER_BLOCKS]
FFN_GRU_LAYERS = [f"{b}.ffn.gru" for b in TRANSFORMER_BLOCKS]
FFN_LINEAR_LAYERS = [f"{b}.ffn.linear" for b in TRANSFORMER_BLOCKS]
