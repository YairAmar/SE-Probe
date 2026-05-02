"""
Demucs DNS64 model constants and layer definitions.

Demucs is a 1D convolutional U-Net:
  - 5 encoder stages (Conv1d + ReLU + Conv1d [+ GLU in stage 0])
  - BLSTM bottleneck
  - 5 decoder stages (Conv1d + ConvTranspose1d + ReLU, except last stage has no ReLU)

Activations are [B, C, T] for conv layers and [T, B, C] for the LSTM.
~30 probed sub-layers total.
"""

__all__ = ["LAYERS", "BLOCK_OUTPUT_LAYERS"]

LAYERS = [
    # Encoder stage 0: Conv1d, ReLU, Conv1d, Sequential
    # Note: GLU (encoder.0.3) excluded — single nn.GLU instance is shared across
    # all encoder+decoder stages, causing its hook to fire 10 times with mixed shapes.
    'encoder.0.0', 'encoder.0.1', 'encoder.0.2', 'encoder.0',
    # Encoder stage 1: Conv1d, ReLU, Conv1d, Sequential
    'encoder.1.0', 'encoder.1.1', 'encoder.1.2', 'encoder.1',
    # Encoder stage 2
    'encoder.2.0', 'encoder.2.1', 'encoder.2.2', 'encoder.2',
    # Encoder stage 3
    'encoder.3.0', 'encoder.3.1', 'encoder.3.2', 'encoder.3',
    # Encoder stage 4
    'encoder.4.0', 'encoder.4.1', 'encoder.4.2', 'encoder.4',
    # BLSTM bottleneck
    'lstm',
    # Decoder stage 0: Conv1d, ConvTranspose1d, ReLU, Sequential
    'decoder.0.0', 'decoder.0.2', 'decoder.0.3', 'decoder.0',
    # Decoder stage 1
    'decoder.1.0', 'decoder.1.2', 'decoder.1.3', 'decoder.1',
    # Decoder stage 2
    'decoder.2.0', 'decoder.2.2', 'decoder.2.3', 'decoder.2',
    # Decoder stage 3
    'decoder.3.0', 'decoder.3.2', 'decoder.3.3', 'decoder.3',
    # Decoder stage 4 (no ReLU after final ConvTranspose1d)
    'decoder.4.0', 'decoder.4.2', 'decoder.4',
]

# Block-level output layers (analogous to norm1 in transformer models):
# one per encoder stage, bottleneck, and decoder stage.
BLOCK_OUTPUT_LAYERS = [
    'encoder.0', 'encoder.1', 'encoder.2', 'encoder.3', 'encoder.4',
    'lstm',
    'decoder.0', 'decoder.1', 'decoder.2', 'decoder.3', 'decoder.4',
]
