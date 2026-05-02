import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed
from .MUSE_net import Multi_transformer

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))

class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(h.dense_channel*(i+1), h.dense_channel, kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

        self.dense_block = DenseBlock(h, depth=4) # [b, h.dense_channel, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2) ,padding=(0, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.mask_conv = nn.Sequential(
            # nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2), padding=(0, 1)),

            # pw-linear
            nn.Conv2d(h.dense_channel, h.dense_channel * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            # dw
            nn.Conv2d(h.dense_channel, h.dense_channel, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1),
                      groups=h.dense_channel, bias=False, ),
            # nn.Conv2d(h.dense_channel, h.dense_channel, (1, 1), (2, 1)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(h.n_fft//2+1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.phase_conv = nn.Sequential(
            # nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),

            # pw-linear
            nn.Conv2d(h.dense_channel, h.dense_channel * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            # nn.Conv2d(h.dense_channel, h.dense_channel, (1, 1), (2, 1)),
            # dw
            nn.Conv2d(h.dense_channel, h.dense_channel, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1),
                      groups=h.dense_channel, bias=False, ),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


class MUSE(nn.Module):
    def __init__(self, h, single_segment_mode: bool = False):
        super(MUSE, self).__init__()
        self.h = h
        self.single_segment_mode = single_segment_mode
        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.TCFTransformer = Multi_transformer(dense_channel=h.dense_channel)
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)
        mag, pha = self.TCFTransformer(x)
        denoised_mag = (noisy_mag * self.mask_decoder(mag)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(pha).permute(0, 3, 2, 1).squeeze(-1)
        denoised_com = torch.stack((denoised_mag*torch.cos(denoised_pha),
                                    denoised_mag*torch.sin(denoised_pha)), dim=-1)

        return denoised_mag, denoised_pha, denoised_com

    def __call__(self, x):
        # Always process full audio for enhanced output
        # In single_segment_mode, pooling will only use first segment's activations
        return self.process_audio_segment(x, x.device)

    def process_audio_segment(self, noisy_wav, device):
        segment_size = self.h.segment_size
        n_fft = self.h.n_fft
        hop_size = self.h.hop_size
        win_size = self.h.win_size
        compress_factor = self.h.compress_factor
        sampling_rate = self.h.sampling_rate

        norm_factor = torch.sqrt(noisy_wav.shape[1] / torch.sum(noisy_wav ** 2.0)).to(device)
        noisy_wav = noisy_wav * norm_factor
        orig_size = noisy_wav.size(1)
        if noisy_wav.size(1) >= segment_size:
            num_segments = noisy_wav.size(1) // segment_size
            last_segment_size = noisy_wav.size(1) % segment_size
            if last_segment_size > 0:
                last_segment = noisy_wav[:, -segment_size:]
                noisy_wav = noisy_wav[:, :-last_segment_size]
                segments = torch.split(noisy_wav, segment_size, dim=1)
                segments = list(segments)
                segments.append(last_segment)
                reshapelast=1
            else:
                segments = torch.split(noisy_wav, segment_size, dim=1)
                reshapelast = 0

        else:
            padded_zeros = torch.zeros(1, segment_size - noisy_wav.size(1)).to(device)
            # print(padded_zeros.size())
            # print(noisy_wav.size())
            noisy_wav = torch.cat((noisy_wav, padded_zeros), dim=1)
            segments = [noisy_wav]
            reshapelast = 0

        processed_segments = []

        for i, segment in enumerate(segments):
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(segment, n_fft, hop_size, win_size, compress_factor)
            amp_g, pha_g, com_g = self.forward(noisy_amp.to(device, non_blocking=True), noisy_pha.to(device, non_blocking=True))
            audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_g = audio_g / norm_factor
            audio_g = audio_g.squeeze()
            if reshapelast == 1 and i == len(segments) - 2:
                audio_g = audio_g[ :-(segment_size-last_segment_size)]

            processed_segments.append(audio_g)


        processed_audio = torch.cat(processed_segments, dim=-1)
        processed_audio = processed_audio[:orig_size]

        return processed_audio




def phase_losses(phase_r, phase_g, h):

    dim_freq = h.n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))

    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            h.sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1

    return pesq_score

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    mag = torch.abs(stft_spec)
    pha = torch.angle(stft_spec)
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav