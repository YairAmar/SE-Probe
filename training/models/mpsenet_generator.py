"""MPSENet generator model (self-contained, no asteroid dependency).

Copied from Meta-Interface-Asteroid/asteroid/models/mp_senet.py with
BaseEncoderEnhancerDecoder replaced by plain nn.Module.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MPNet(nn.Module):
    """Multi-Phase Network for speech enhancement.

    Args:
        dense_channel (int): Number of channels in dense blocks
        n_fft (int): FFT size
        sigmoid_beta (float): Beta parameter for learnable sigmoid
        num_tsblocks (int, optional): Number of transformer blocks. Defaults to 4.
    """
    def __init__(self,
                 dense_channel: int,
                 n_fft: int,
                 sigmoid_beta: float,
                 num_tsblocks: int = 4) -> None:
        super().__init__()
        self.num_tscblocks = num_tsblocks
        self.dense_channel = dense_channel

        self.encoder = DenseEncoder(dense_channel=dense_channel, in_channel=2)

        self.enhancer = nn.ModuleList([])
        for i in range(num_tsblocks):
            self.enhancer.append(TSTransformerBlock(dense_channel=dense_channel))

        self.decoder = MPDecoder(dense_channel=dense_channel,
                                 n_fft=n_fft,
                                 sigmoid_beta=sigmoid_beta,
                                 out_channel=1)

    def forward_enhancer(self, tf_rep: Tensor) -> Tensor:
        for i in range(self.num_tscblocks):
            tf_rep = self.enhancer[i](tf_rep)
        return tf_rep

    def forward(self, noisy_amp: Tensor, noisy_pha: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the network.

        Args:
            noisy_amp (Tensor): Noisy amplitude spectrum [B, F, T]
            noisy_pha (Tensor): Noisy phase spectrum [B, F, T]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Denoised amplitude, phase, and complex spectrum
        """
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)  # [B, 2, T, F]
        x = self.encoder(x)
        x = self.forward_enhancer(x)
        denoised_amp, denoised_pha, denoised_com = self.decoder(x, noisy_amp)
        return denoised_amp, denoised_pha, denoised_com


class TSTransformerBlock(nn.Module):
    def __init__(self, dense_channel: int) -> None:
        super().__init__()
        self.time_transformer = TransformerBlock(d_model=dense_channel, n_heads=4)
        self.freq_transformer = TransformerBlock(d_model=dense_channel, n_heads=4)

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 bidirectional: bool = True,
                 dropout: float = 0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)
        x = x + self.dropout1(xt)
        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)
        x = self.norm3(x)
        return x


class FFN(nn.Module):
    def __init__(self,
                 d_model: int,
                 bidirectional: bool = True,
                 dropout: float = 0) -> None:
        super().__init__()
        self.gru = nn.GRU(d_model, d_model*2, 1, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(d_model*2*2, d_model)
        else:
            self.linear = nn.Linear(d_model*2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 r: int = 1) -> None:
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self,
                 dense_channel: int,
                 kernel_size: Tuple[int, int] = (2, 3),
                 depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                nn.Conv2d(dense_channel*(i+1), dense_channel, kernel_size, dilation=(dilation, 1)),
                nn.InstanceNorm2d(dense_channel, affine=True),
                nn.PReLU(dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self,
                 dense_channel: int,
                 in_channel: int) -> None:
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))
        self.dense_block = DenseBlock(dense_channel=dense_channel, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self,
                 dense_channel: int,
                 n_fft: int,
                 sigmoid_beta: float,
                 out_channel: int = 1) -> None:
        super().__init__()
        self.dense_block = DenseBlock(dense_channel=dense_channel, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
            nn.Conv2d(dense_channel, out_channel, (1, 2))
        )
        self.lsigmoid = LearnableSigmoid2d(n_fft//2+1, beta=sigmoid_beta)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self,
                 dense_channel: int,
                 out_channel: int = 1) -> None:
        super().__init__()
        self.dense_block = DenseBlock(dense_channel=dense_channel, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 2))

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]
        return x


class MPDecoder(nn.Module):
    def __init__(self,
                 dense_channel: int,
                 n_fft: int,
                 sigmoid_beta: float,
                 out_channel: int) -> None:
        super().__init__()
        self.phase_decoder = PhaseDecoder(dense_channel=dense_channel,
                                          out_channel=out_channel)
        self.mask_decoder = MaskDecoder(dense_channel=dense_channel,
                                        n_fft=n_fft,
                                        sigmoid_beta=sigmoid_beta,
                                        out_channel=out_channel)

    def forward(self, x: Tensor, noisy_amp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack((denoised_amp*torch.cos(denoised_pha),
                                    denoised_amp*torch.sin(denoised_pha)), dim=-1)
        return denoised_amp, denoised_pha, denoised_com


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features: int, beta: float = 1) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x: Tensor) -> Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


# ---------- Phase losses (from asteroid/losses/mpsenet_loss.py) ----------

def anti_wrapping_function(x: Tensor) -> Tensor:
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_losses(phase_r: Tensor, phase_g: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Anti-wrapping phase loss (IP + GD + IAF).

    Uses torch.diff for group-delay and instantaneous-angular-frequency losses.
    """
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))
    return ip_loss, gd_loss, iaf_loss
