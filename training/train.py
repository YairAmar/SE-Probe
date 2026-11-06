"""Fine-tune MUSE for dereverberation.

Loads a pretrained MUSE generator (trained on VB+DEMAND denoising) and
fine-tunes it end-to-end on reverberant speech generated on-the-fly by
convolving clean utterances with RIRs.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import csv
import json
import random as stdlib_random
import numpy as np
import torch
import torch.nn.functional as F
try:  # experiment tracking is optional at import time (tests import this module)
    import wandb
except ImportError:  # pragma: no cover
    wandb = None
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pesq import pesq as compute_pesq
from pystoi import stoi
try:
    from speechmos import dnsmos
except ImportError:  # pragma: no cover
    dnsmos = None
from torch.utils.data import DataLoader

from env import AttrDict, build_env
from datasets.dataset import get_dataset_filelist, mag_pha_stft, mag_pha_istft
from datasets.reverb_dataset import ReverbDataset, ReverbValDataset
from models.generator import MUSE, phase_losses
from models.discriminator import MetricDiscriminator, batch_pesq
from utils import scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Selective freezing (ported from the Muse-Reverb-FN trainer, and the three
# profile-guided arms of the Job A experiment). All names are matched against
# `generator.named_parameters()` of models.generator.MUSE.
# ---------------------------------------------------------------------------

# --unfreeze <block>: every parameter owned by one of MUSE's six stages,
# including its patch-embedding / down- / up-sampling / output heads.
BLOCK_TO_PREFIXES = {
    'encoder_l1': ['patch_embed_encoder_level1.', 'encoder_level1.', 'down1_2.'],
    'encoder_l2': ['patch_embed_encoder_level2.', 'encoder_level2.', 'down2_3.'],
    'latent':     ['patch_embed_latent.', 'latent.'],
    'decoder_l2': ['up3_2.', 'reduce_chan_level2.', 'patch_embed_decoder_level2.', 'decoder_level2.'],
    'decoder_l1': ['up2_1.', 'reduce_chan_level1.', 'patch_embed_decoder_level1.', 'decoder_level1.'],
    'refinement': ['mag_patch_embed_refinement.', 'mag_refinement.', 'mag_output.',
                   'pha_patch_embed_refinement.', 'pha_refinement.', 'pha_output.'],
}

# --unfreeze <block>.<layer>: a single transformer layer (0-3) inside a stage.
BLOCK_TO_LAYER_PREFIXES = {
    'encoder_l1': ['encoder_level1.'],
    'encoder_l2': ['encoder_level2.'],
    'latent':     ['latent.'],
    'decoder_l2': ['decoder_level2.'],
    'decoder_l1': ['decoder_level1.'],
    'refinement': ['mag_refinement.', 'pha_refinement.'],
}
NUM_LAYERS_PER_BLOCK = 4

# --freeze_arm: the three profile-guided fine-tuning arms of Job A. A parameter
# is FROZEN when its name contains any of the listed substrings; everything
# else stays trainable. `full_ft` freezes nothing.
#   freeze_encoder : the profile-robust encoder stages (76% of parameters trainable)
#   freeze_decoder : the profile-sensitive decoder stages and the mask/phase heads
#                    (71% trainable). Note that mag_refinement is NOT frozen by this
#                    arm and latent is frozen by neither.
FREEZE_ARM_KEYS = {
    'full_ft': (),
    'freeze_encoder': ('dense_encoder', 'TCFTransformer.encoder_level'),
    'freeze_decoder': ('TCFTransformer.decoder_level', 'mask_decoder', 'phase_decoder'),
}


def validate_unfreeze(spec):
    """Validate an --unfreeze value; raise ValueError with a helpful message."""
    valid_blocks = list(BLOCK_TO_PREFIXES.keys()) + ['all']
    if '.' in spec:
        block, layer_str = spec.rsplit('.', 1)
        if block not in BLOCK_TO_LAYER_PREFIXES:
            raise ValueError(f"--unfreeze: invalid block '{block}'. "
                             f"Choose from: {list(BLOCK_TO_LAYER_PREFIXES.keys())}")
        try:
            layer_idx = int(layer_str)
        except ValueError:
            raise ValueError(f"--unfreeze: layer index must be an integer, got '{layer_str}'")
        if not (0 <= layer_idx < NUM_LAYERS_PER_BLOCK):
            raise ValueError(f"--unfreeze: layer index must be 0-{NUM_LAYERS_PER_BLOCK - 1}, "
                             f"got {layer_idx}")
    elif spec not in valid_blocks:
        raise ValueError(f"--unfreeze: invalid value '{spec}'. Choose from: {valid_blocks} "
                         "or use block.layer syntax (e.g. encoder_l1.0)")


def apply_unfreeze(generator, unfreeze, unfreeze_boundary=False):
    """Freeze every generator parameter, then re-enable the selected stage / layer.

    unfreeze: 'all', a block name from BLOCK_TO_PREFIXES, or '<block>.<layer>'.
    unfreeze_boundary: additionally unfreeze dense_encoder / mask_decoder / phase_decoder.
    Returns (n_trainable, n_total) parameter counts.
    """
    validate_unfreeze(unfreeze)
    for p in generator.parameters():
        p.requires_grad = False

    if unfreeze == 'all':
        for p in generator.parameters():
            p.requires_grad = True
    elif '.' in unfreeze:
        block, layer_str = unfreeze.rsplit('.', 1)
        layer_idx = int(layer_str)
        for name, p in generator.named_parameters():
            for sp in BLOCK_TO_LAYER_PREFIXES[block]:
                if name.startswith(f'TCFTransformer.{sp}mhca_blks.0.transformer_layers.{layer_idx}.'):
                    p.requires_grad = True
    else:
        prefixes = BLOCK_TO_PREFIXES[unfreeze]
        for name, p in generator.named_parameters():
            if any(name.startswith(f'TCFTransformer.{pfx}') for pfx in prefixes):
                p.requires_grad = True

    if unfreeze_boundary:
        for name, p in generator.named_parameters():
            if name.startswith(('dense_encoder.', 'mask_decoder.', 'phase_decoder.')):
                p.requires_grad = True

    total = sum(p.numel() for p in generator.parameters())
    trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    return trainable, total


def apply_freeze_arm(generator, arm):
    """Configure requires_grad per Job A arm (full_ft / freeze_encoder / freeze_decoder).

    Returns (n_trainable, n_frozen) parameter counts.
    """
    if arm not in FREEZE_ARM_KEYS:
        raise ValueError(f"--freeze_arm: invalid value '{arm}'. "
                         f"Choose from: {list(FREEZE_ARM_KEYS)}")
    keys = FREEZE_ARM_KEYS[arm]
    trainable = frozen = 0
    for name, p in generator.named_parameters():
        p.requires_grad_(not any(k in name for k in keys))
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    return trainable, frozen


def process_audio(noisy_audio, generator, device, segment_size, n_fft, hop_size, win_size, compress_factor):
    """Segmented full-signal inference on a 1D tensor. Returns 1D CPU tensor."""
    orig_size = noisy_audio.size(0)
    noisy_audio = noisy_audio.unsqueeze(0)

    if noisy_audio.size(1) >= segment_size:
        last_segment_size = noisy_audio.size(1) % segment_size
        if last_segment_size > 0:
            last_segment = noisy_audio[:, -segment_size:]
            noisy_audio_trimmed = noisy_audio[:, :-last_segment_size]
            segments = list(torch.split(noisy_audio_trimmed, segment_size, dim=1))
            segments.append(last_segment)
            reshapelast = 1
        else:
            segments = list(torch.split(noisy_audio, segment_size, dim=1))
            reshapelast = 0
    else:
        padded = torch.zeros(1, segment_size - noisy_audio.size(1)).to(device)
        noisy_audio = torch.cat((noisy_audio, padded), dim=1)
        segments = [noisy_audio]
        reshapelast = 0

    processed = []
    for i, segment in enumerate(segments):
        noisy_amp, noisy_pha, _ = mag_pha_stft(segment, n_fft, hop_size, win_size, compress_factor)
        amp_g, pha_g, _ = generator(noisy_amp.to(device), noisy_pha.to(device))
        audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor).squeeze()
        if reshapelast == 1 and i == len(segments) - 2:
            audio_g = audio_g[:-(segment_size - last_segment_size)]
        processed.append(audio_g)

    return torch.cat(processed, dim=-1)[:orig_size].cpu()


def make_spectrogram_fig(clean_np, noisy_np, enhanced_np, sr, n_fft, hop_size):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, audio, title in zip(axes, [clean_np, noisy_np, enhanced_np],
                                 ['Clean', 'Reverberant', 'Enhanced']):
        spec = np.abs(np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(
                np.pad(audio, (0, n_fft - len(audio) % n_fft if len(audio) % n_fft else 0)),
                n_fft)[::hop_size] * np.hanning(n_fft)
        ))
        ax.imshow(20 * np.log10(spec.T + 1e-8), aspect='auto', origin='lower',
                  extent=[0, len(audio) / sr, 0, sr / 2])
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (Hz)')
    fig.tight_layout()
    return fig


def train(a, h):
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    generator = MUSE(h).to(device)
    discriminator = MetricDiscriminator().to(device)

    print(generator)
    print("Generator parameters:", sum(p.numel() for p in generator.parameters()))
    os.makedirs(a.checkpoint_path, exist_ok=True)
    print("Checkpoints directory:", a.checkpoint_path)

    steps = 0
    state_dict_do = None
    last_epoch = -1

    # Resume if a fine-tuning checkpoint already exists
    cp_g = scan_checkpoint(a.checkpoint_path, 'g_') if os.path.isdir(a.checkpoint_path) else None
    cp_do = scan_checkpoint(a.checkpoint_path, 'do_') if os.path.isdir(a.checkpoint_path) else None

    if cp_g is not None and cp_do is not None:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        print(f"Resumed from {cp_g} (step {steps}, epoch {last_epoch})")
    elif a.pretrained_checkpoint:
        state_dict_g = load_checkpoint(a.pretrained_checkpoint, device)
        generator.load_state_dict(state_dict_g['generator'])
        print(f"Loaded pretrained generator from {a.pretrained_checkpoint}")

    # Selective freezing (applied right after the weights are loaded)
    if a.unfreeze is not None:
        trainable, total = apply_unfreeze(generator, a.unfreeze, a.unfreeze_boundary)
        print(f"Parameters: {trainable}/{total} trainable ({100 * trainable / total:.1f}%) "
              f"[--unfreeze {a.unfreeze}{' --unfreeze_boundary' if a.unfreeze_boundary else ''}]")
    if a.freeze_arm is not None:
        trainable, frozen = apply_freeze_arm(generator, a.freeze_arm)
        print(f"freeze arm={a.freeze_arm}: trainable={trainable / 1e6:.2f}M "
              f"frozen={frozen / 1e6:.2f}M "
              f"({100 * trainable / (trainable + frozen):.1f}%)")
    if a.freeze_discriminator:
        for p in discriminator.parameters():
            p.requires_grad = False
        print("Discriminator frozen")

    lr = a.lr if a.lr else h.learning_rate
    # Only trainable parameters reach the optimizers (identical to the full
    # fine-tune when nothing is frozen).
    g_params = [p for p in generator.parameters() if p.requires_grad]
    d_params = [p for p in discriminator.parameters() if p.requires_grad]
    optim_g = torch.optim.AdamW(g_params if g_params else [torch.nn.Parameter(torch.empty(0))],
                                lr, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(d_params if d_params else [torch.nn.Parameter(torch.empty(0))],
                                lr, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_indexes, validation_indexes = get_dataset_filelist(a)

    # RIR setup
    rir_meta = {}
    with open(a.rir_metadata, "r") as f:
        for row in csv.DictReader(f):
            rir_meta[row["filename"]] = row["wav_path"]

    with open(a.rir_split, "r") as f:
        rir_split = json.load(f)
    train_rir_fns = sorted([fn for fn, s in rir_split.items() if s == "train"])

    rng = stdlib_random.Random(h.seed)
    shuffled = list(train_rir_fns)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * a.val_rir_ratio))
    val_rir_fns = shuffled[:n_val]
    actual_train_fns = shuffled[n_val:]

    train_rir_paths = [rir_meta[fn] for fn in actual_train_fns]
    val_rir_paths = [rir_meta[fn] for fn in val_rir_fns]
    print(f"RIR split: {len(train_rir_paths)} train, {len(val_rir_paths)} val holdout")

    trainset = ReverbDataset(training_indexes, a.input_clean_wavs_dir, a.rir_dir,
                             train_rir_paths,
                             h.segment_size, h.n_fft, h.hop_size, h.win_size, h.sampling_rate,
                             h.compress_factor, split=True, shuffle=True, device=device)

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True,
                              persistent_workers=h.num_workers > 0)

    val_clean_dir = a.val_clean_wavs_dir if a.val_clean_wavs_dir else a.input_clean_wavs_dir
    validset = ReverbValDataset(validation_indexes, val_clean_dir,
                                a.rir_dir, val_rir_paths, h.sampling_rate)
    validation_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False,
                                   batch_size=1, pin_memory=True, drop_last=True,
                                   persistent_workers=h.num_workers > 0)

    total_params = sum(p.numel() for p in generator.parameters())
    wandb.init(
        project="muse-dereverb-ft",
        name=a.run_name or "full-ft",
        config={**dict(h), "lr": lr, "training_epochs": a.training_epochs,
                "pretrained_checkpoint": a.pretrained_checkpoint,
                "unfreeze": a.unfreeze, "unfreeze_boundary": a.unfreeze_boundary,
                "freeze_arm": a.freeze_arm, "freeze_discriminator": a.freeze_discriminator,
                "seed": h.seed,
                "total_params": total_params,
                "trainable_params": sum(p.numel() for p in generator.parameters() if p.requires_grad)},
    )

    generator.train()
    discriminator.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        start = time.time()
        print(f"Epoch: {epoch + 1}")

        for batch in train_loader:
            start_b = time.time()
            clean_audio, clean_mag, clean_pha, clean_com, _, noisy_mag, noisy_pha = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            clean_mag = clean_mag.to(device, non_blocking=True)
            clean_pha = clean_pha.to(device, non_blocking=True)
            clean_com = clean_com.to(device, non_blocking=True)
            noisy_mag = noisy_mag.to(device, non_blocking=True)
            noisy_pha = noisy_pha.to(device, non_blocking=True)
            one_labels = torch.ones(h.batch_size).to(device, non_blocking=True)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
            audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            batch_pesq_score = batch_pesq(list(clean_audio.cpu().numpy()),
                                          list(audio_g.detach().cpu().numpy()))

            # Discriminator
            if not a.freeze_discriminator:
                optim_d.zero_grad()
                metric_r = discriminator(clean_mag, clean_mag)
                metric_g = discriminator(clean_mag, mag_g.detach())
                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten()) \
                    if batch_pesq_score is not None else 0
                loss_disc_all = loss_disc_r + loss_disc_g
                loss_disc_all.backward()
                optim_d.step()
            else:
                loss_disc_all = torch.tensor(0.0)

            # Generator
            optim_g.zero_grad()
            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, h)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2
            metric_g = discriminator(clean_mag, mag_g)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
            loss_gen_all = loss_metric * 0.05 + loss_mag * 0.9 + loss_pha * 0.3 + loss_com * 0.1
            loss_gen_all.backward()
            optim_g.step()

            if steps % a.stdout_interval == 0:
                print(f'Steps {steps:d}, Gen {loss_gen_all:.3f}, Disc {loss_disc_all:.3f}, '
                      f'Mag {loss_mag:.3f}, Pha {loss_pha:.3f}, Com {loss_com:.3f}, '
                      f's/b {time.time() - start_b:.3f}')

            if steps % a.checkpoint_interval == 0 and steps != 0:
                save_checkpoint(f"{a.checkpoint_path}/g_{steps:08d}",
                                {'generator': generator.state_dict()})
                save_checkpoint(f"{a.checkpoint_path}/do_{steps:08d}",
                                {'discriminator': discriminator.state_dict(),
                                 'optim_g': optim_g.state_dict(),
                                 'optim_d': optim_d.state_dict(),
                                 'steps': steps, 'epoch': epoch})

            if steps % a.summary_interval == 0:
                wandb.log({
                    "Training/Generator Loss": loss_gen_all.item(),
                    "Training/Discriminator Loss": loss_disc_all.item() if torch.is_tensor(loss_disc_all) else loss_disc_all,
                    "Training/Magnitude Loss": loss_mag.item(),
                    "Training/Phase Loss": loss_pha.item(),
                    "Training/Complex Loss": loss_com.item(),
                    "epoch": epoch + 1,
                }, step=steps)

            if steps % a.validation_interval == 0 and steps != 0:
                _run_validation(generator, validation_loader, validset, h, device,
                                steps, epoch)

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if a.save_every_epoch:
            ed = os.path.join(a.checkpoint_path, f'epoch_{epoch + 1}')
            os.makedirs(ed, exist_ok=True)
            save_checkpoint(os.path.join(ed, f'g_{steps:08d}'),
                            {'generator': generator.state_dict()})
            save_checkpoint(os.path.join(ed, f'do_{steps:08d}'),
                            {'discriminator': discriminator.state_dict(),
                             'optim_g': optim_g.state_dict(),
                             'optim_d': optim_d.state_dict(),
                             'steps': steps, 'epoch': epoch})

        print(f'Time taken for epoch {epoch + 1}: {int(time.time() - start)} sec\n')


def _run_validation(generator, validation_loader, validset, h, device, steps, epoch):
    torch.cuda.empty_cache()
    generator.eval()
    val_mag, val_pha, val_com = 0, 0, 0
    with torch.no_grad():
        for j, batch in enumerate(validation_loader):
            clean_audio, noisy_audio = batch
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            clean_audio = clean_audio.to(device, non_blocking=True)
            audio_g = process_audio(noisy_audio.squeeze(0), generator, device,
                                    h.segment_size, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            cmag, cpha, ccom = mag_pha_stft(clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            mg, pg, cg = mag_pha_stft(audio_g.unsqueeze(0), h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            cmag, cpha, ccom = cmag.to(device), cpha.to(device), ccom.to(device)
            mg, pg, cg = mg.to(device), pg.to(device), cg.to(device)

            val_mag += F.mse_loss(cmag.squeeze(), mg.squeeze()).item()
            ip, gd, iaf = phase_losses(cpha, pg, h)
            val_pha += (ip + gd + iaf).item()
            val_com += F.mse_loss(ccom.squeeze(), cg.squeeze()).item()
        n_val = j + 1

        # 10-sample subset for PESQ/STOI/DNSMOS + audio logging
        n_sub = min(10, len(validset))
        pesqs, stois, dnss, pesqs_n, stois_n, dnss_n = [], [], [], [], [], []
        wandb_logs = {}
        for j in range(n_sub):
            clean_wav, noisy_wav = validset[j]
            enh_wav = process_audio(noisy_wav.to(device), generator, device,
                                    h.segment_size, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            c, n, e = clean_wav.numpy(), noisy_wav.numpy(), enh_wav.numpy()
            sr = h.sampling_rate

            try:    p = compute_pesq(sr, c, e, 'wb')
            except: p = float('nan')
            try:    s = stoi(c, e, sr, extended=False)
            except: s = float('nan')
            try:    d = dnsmos.run(e, sr)["ovrl_mos"]
            except: d = float('nan')
            pesqs.append(p); stois.append(s); dnss.append(d)

            try:    p_n = compute_pesq(sr, c, n, 'wb')
            except: p_n = float('nan')
            try:    s_n = stoi(c, n, sr, extended=False)
            except: s_n = float('nan')
            try:    d_n = dnsmos.run(n, sr)["ovrl_mos"]
            except: d_n = float('nan')
            pesqs_n.append(p_n); stois_n.append(s_n); dnss_n.append(d_n)

            if j < 5:
                cap = (f"sample_{j} | PESQ={p:.2f}/{p_n:.2f} STOI={s:.3f}/{s_n:.3f} "
                       f"DNSMOS={d:.2f}/{d_n:.2f}")
                pre = f"Validation/sample_{j}"
                cc = c / max(np.abs(c).max(), 1e-8)
                nn = n / max(np.abs(n).max(), 1e-8)
                ee = e / max(np.abs(e).max(), 1e-8)
                wandb_logs[f"{pre}/clean"] = wandb.Audio(cc, sample_rate=sr, caption=f"clean | {cap}")
                wandb_logs[f"{pre}/reverb"] = wandb.Audio(nn, sample_rate=sr, caption=f"reverb | {cap}")
                wandb_logs[f"{pre}/enhanced"] = wandb.Audio(ee, sample_rate=sr, caption=f"enhanced | {cap}")
                fig = make_spectrogram_fig(c, n, e, sr, h.n_fft, h.hop_size)
                wandb_logs[f"{pre}/spectrogram"] = wandb.Image(fig, caption=cap)
                plt.close(fig)

        print(f'Steps {steps}: PESQ {np.nanmean(pesqs):.3f} (in {np.nanmean(pesqs_n):.3f}), '
              f'STOI {np.nanmean(stois):.3f} (in {np.nanmean(stois_n):.3f}), '
              f'DNSMOS {np.nanmean(dnss):.3f} (in {np.nanmean(dnss_n):.3f})')

        wandb.log({
            "Validation/Magnitude Loss": val_mag / n_val,
            "Validation/Phase Loss": val_pha / n_val,
            "Validation/Complex Loss": val_com / n_val,
            "Validation/PESQ": float(np.nanmean(pesqs)),
            "Validation/STOI": float(np.nanmean(stois)),
            "Validation/DNSMOS": float(np.nanmean(dnss)),
            "Validation/PESQ_input": float(np.nanmean(pesqs_n)),
            "Validation/STOI_input": float(np.nanmean(stois_n)),
            "Validation/DNSMOS_input": float(np.nanmean(dnss_n)),
            "epoch": epoch + 1,
            **wandb_logs,
        }, step=steps)
    generator.train()


def main():
    print('Initializing fine-tuning process...')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', required=True)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--checkpoint_path', default='checkpoints/dereverb')

    parser.add_argument('--input_clean_wavs_dir', required=True)
    parser.add_argument('--input_training_file', default='VoiceBank+DEMAND/training.txt')
    parser.add_argument('--input_validation_file', default='VoiceBank+DEMAND/test.txt')
    parser.add_argument('--val_clean_wavs_dir', default=None)
    parser.add_argument('--input_noisy_wavs_dir', default=None,
                        help='Unused for reverb fine-tuning; kept for compat with get_dataset_filelist')
    parser.add_argument('--val_noisy_wavs_dir', default=None)

    parser.add_argument('--rir_dir', required=True)
    parser.add_argument('--rir_metadata', required=True)
    parser.add_argument('--rir_split', required=True)
    parser.add_argument('--val_rir_ratio', default=0.05, type=float)

    parser.add_argument('--pretrained_checkpoint', default='paper_result/g_best')
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--training_epochs', default=50, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=850, type=int)
    parser.add_argument('--save_every_epoch', default=True,
                        action=argparse.BooleanOptionalAction)

    # Selective freezing (see BLOCK_TO_PREFIXES / FREEZE_ARM_KEYS at the top)
    parser.add_argument('--unfreeze', default=None, type=str,
                        help='Train only this part of the generator. Blocks: all|encoder_l1|'
                             'encoder_l2|latent|decoder_l2|decoder_l1|refinement. '
                             'Single layers: encoder_l1.0|encoder_l1.1|...|refinement.3')
    parser.add_argument('--unfreeze_boundary', action='store_true',
                        help='With --unfreeze: also train dense_encoder/mask_decoder/phase_decoder')
    parser.add_argument('--freeze_arm', default=None,
                        choices=['full_ft', 'freeze_encoder', 'freeze_decoder'],
                        help='Profile-guided fine-tuning arm of the Job A experiment')
    parser.add_argument('--freeze_discriminator', action='store_true',
                        help='Freeze the MetricDiscriminator')
    parser.add_argument('--seed', default=None, type=int,
                        help='Override the config seed (the arms used 1234, 2345, 3456)')

    a = parser.parse_args()

    if a.unfreeze is not None:
        try:
            validate_unfreeze(a.unfreeze)
        except ValueError as e:
            parser.error(str(e))
    if a.unfreeze is not None and a.freeze_arm is not None:
        parser.error('--unfreeze and --freeze_arm are mutually exclusive')

    with open(a.config) as f:
        h = AttrDict(json.loads(f.read()))
    build_env(a.config, 'config.json', a.checkpoint_path)

    if a.seed is not None:
        h.seed = a.seed
    stdlib_random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        torch.cuda.manual_seed_all(h.seed)
        h.num_gpus = 1
    else:
        h.num_gpus = 0

    train(a, h)


if __name__ == '__main__':
    main()
