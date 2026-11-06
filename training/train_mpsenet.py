"""MPSENet fine-tuning script for dereverberation.

Mirrors train.py structure but with MPSENet-specific:
  - STFT computed in training step (not in dataset)
  - Consistency loss (ISTFT -> STFT round-trip)
  - Time-domain L1 loss
  - PL checkpoint loading (strip 'model.' / 'discriminator_loss_func.discriminator.' prefixes)
  - Block-level freezing for MPSENet architecture
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
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pesq import pesq as compute_pesq
from pystoi import stoi
from speechmos import dnsmos
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from datasets.dataset import mag_pha_stft, mag_pha_istft, get_dataset_filelist


def mag_pha_stft_safe(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    """mag_pha_stft with epsilon-clamped magnitude for gradient-safe compression.

    torch.pow(mag, compress_factor) has infinite gradient at mag=0 when
    compress_factor < 1.  Clamping avoids NaN gradients in the backward pass
    (needed for the ISTFT -> STFT consistency round-trip).
    """
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    mag = torch.abs(stft_spec).clamp(min=1e-9)
    pha = torch.angle(stft_spec)
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com
from models.mpsenet_generator import MPNet, phase_losses
from models.mpsenet_discriminator import MetricDiscriminator, batch_pesq
from utils import scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

# MPSENet block-level freezing map
BLOCK_TO_PREFIXES = {
    'encoder':       ['encoder.'],
    'enhancer':      ['enhancer.'],
    'enhancer_0':    ['enhancer.0.'],
    'enhancer_1':    ['enhancer.1.'],
    'enhancer_2':    ['enhancer.2.'],
    'enhancer_3':    ['enhancer.3.'],
    'mask_decoder':  ['decoder.mask_decoder.'],
    'phase_decoder': ['decoder.phase_decoder.'],
    'decoder':       ['decoder.'],
}


def si_sdr(reference, estimate, eps=1e-8):
    """Scale-invariant SDR (dB) between 1D numpy arrays of equal length."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference
    noise = estimate - target
    return float(10.0 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def process_audio(noisy_audio, generator, device, segment_size, n_fft, hop_size, win_size, compress_factor):
    """Segmented full-signal inference on a 1D tensor. Returns 1D CPU tensor."""
    orig_size = noisy_audio.size(0)
    noisy_audio = noisy_audio.unsqueeze(0)  # [1, T]

    if noisy_audio.size(1) >= segment_size:
        num_segments = noisy_audio.size(1) // segment_size
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
        padded_zeros = torch.zeros(1, segment_size - noisy_audio.size(1)).to(device)
        noisy_audio = torch.cat((noisy_audio, padded_zeros), dim=1)
        segments = [noisy_audio]
        reshapelast = 0

    processed_segments = []
    for i, segment in enumerate(segments):
        noisy_amp, noisy_pha, noisy_com = mag_pha_stft(segment, n_fft, hop_size, win_size, compress_factor)
        amp_g, pha_g, com_g = generator(noisy_amp.to(device), noisy_pha.to(device))
        audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
        audio_g = audio_g.squeeze()
        if reshapelast == 1 and i == len(segments) - 2:
            audio_g = audio_g[:-(segment_size - last_segment_size)]
        processed_segments.append(audio_g)

    processed_audio = torch.cat(processed_segments, dim=-1)
    return processed_audio[:orig_size].cpu()


def make_spectrogram_fig(clean_np, noisy_np, enhanced_np, sr, n_fft, hop_size):
    """3-panel spectrogram figure (clean / noisy / enhanced)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, audio, title in zip(axes, [clean_np, noisy_np, enhanced_np],
                                 ['Clean', 'Noisy', 'Enhanced']):
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


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = MPNet(
        dense_channel=h.dense_channel,
        n_fft=h.n_fft,
        sigmoid_beta=h.beta,
        num_tsblocks=h.num_tsblocks,
    ).to(device)
    discriminator = MetricDiscriminator(dim=64, in_channel=2).to(device)

    if rank == 0:
        print(generator)
        num_params = sum(p.numel() for p in generator.parameters())
        print("Generator Parameters : ", num_params)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    steps = 0
    state_dict_do = None
    last_epoch = -1

    # Resume from existing fine-tuning checkpoints if available
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
        ckpt = torch.load(a.pretrained_checkpoint, map_location=device)
        if 'state_dict' in ckpt:
            # PyTorch Lightning checkpoint — strip prefixes
            sd = ckpt['state_dict']
            gen_sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
            disc_sd = {k[len('discriminator_loss_func.discriminator.'):]: v
                       for k, v in sd.items() if k.startswith('discriminator_loss_func.discriminator.')}
            generator.load_state_dict(gen_sd)
            if disc_sd:
                discriminator.load_state_dict(disc_sd)
            print(f"Loaded PL checkpoint from {a.pretrained_checkpoint} "
                  f"(gen: {len(gen_sd)} keys, disc: {len(disc_sd)} keys)")
        elif 'generator' in ckpt:
            # Our own checkpoint format
            generator.load_state_dict(ckpt['generator'])
            print(f"Loaded pretrained generator from {a.pretrained_checkpoint}")
        else:
            raise ValueError(f"Unrecognized checkpoint format: {list(ckpt.keys())[:10]}")

    # Selective freezing
    if a.unfreeze is not None:
        for p in generator.parameters():
            p.requires_grad = False

        if a.unfreeze == 'all':
            for p in generator.parameters():
                p.requires_grad = True
        else:
            prefixes = BLOCK_TO_PREFIXES[a.unfreeze]
            for name, p in generator.named_parameters():
                if any(name.startswith(pfx) for pfx in prefixes):
                    p.requires_grad = True

        if a.unfreeze_boundary:
            for name, p in generator.named_parameters():
                if name.startswith(('encoder.', 'decoder.')):
                    p.requires_grad = True

        total = sum(p.numel() for p in generator.parameters())
        trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        if rank == 0:
            print(f"Parameters: {trainable}/{total} trainable ({100*trainable/total:.1f}%)")

    # Freeze discriminator if requested
    if a.freeze_discriminator:
        for p in discriminator.parameters():
            p.requires_grad = False
        if rank == 0:
            print("Discriminator frozen")

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank], find_unused_parameters=True).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank], find_unused_parameters=True).to(device)

    lr = a.lr if a.lr else h.learning_rate
    g_params = list(filter(lambda p: p.requires_grad, generator.parameters()))
    d_params = list(filter(lambda p: p.requires_grad, discriminator.parameters()))
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

    if a.mode == 'reverb':
        from datasets.reverb_dataset import MPSENetReverbDataset
        # Load RIR metadata (filename -> wav_path mapping)
        rir_meta = {}
        with open(a.rir_metadata, "r") as f:
            for row in csv.DictReader(f):
                rir_meta[row["filename"]] = row["wav_path"]

        # Load RIR split and create train/val holdout
        with open(a.rir_split, "r") as f:
            rir_split = json.load(f)
        train_rir_fns = sorted([fn for fn, s in rir_split.items() if s == "train"])

        # Holdout val RIRs from train set
        rng = stdlib_random.Random(h.seed)
        shuffled = list(train_rir_fns)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * a.val_rir_ratio))
        val_rir_fns = shuffled[:n_val]
        actual_train_fns = shuffled[n_val:]

        # Map filenames to wav_paths for dataset loading
        train_rir_paths = [rir_meta[fn] for fn in actual_train_fns]
        val_rir_paths = [rir_meta[fn] for fn in val_rir_fns]
        if rank == 0:
            print(f"RIR split: {len(train_rir_fns)} train total -> "
                  f"{len(train_rir_paths)} train, {len(val_rir_paths)} val holdout")

        trainset = MPSENetReverbDataset(training_indexes, a.input_clean_wavs_dir, a.rir_dir,
                                        train_rir_paths,
                                        h.segment_size, h.sampling_rate,
                                        split=True,
                                        shuffle=False if h.num_gpus > 1 else True, device=device)
    else:
        raise ValueError("MPSENet training currently only supports --mode reverb")

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True if h.num_workers > 0 else False)
    if rank == 0:
        if a.mode == 'reverb':
            from datasets.reverb_dataset import MPSENetReverbValDataset
            val_clean_dir = a.val_clean_wavs_dir if a.val_clean_wavs_dir else a.input_clean_wavs_dir
            validset = MPSENetReverbValDataset(validation_indexes, val_clean_dir,
                                               a.rir_dir, val_rir_paths, h.sampling_rate)
        else:
            raise ValueError("MPSENet training currently only supports --mode reverb")

        validation_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True,
                                       persistent_workers=True if h.num_workers > 0 else False)

        total_params = sum(p.numel() for p in generator.parameters())
        trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        run_name = a.unfreeze if a.unfreeze else a.mode
        wandb.init(
            project="MPSENet-Reverb-FN",
            name=run_name,
            config={
                **dict(h),
                "mode": a.mode,
                "unfreeze": a.unfreeze,
                "unfreeze_boundary": a.unfreeze_boundary,
                "freeze_discriminator": a.freeze_discriminator,
                "lr": lr,
                "training_epochs": a.training_epochs,
                "pretrained_checkpoint": a.pretrained_checkpoint,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_pct": 100 * trainable_params / total_params if total_params > 0 else 0,
            },
        )

    generator.train()
    discriminator.train()

    n_fft = h.n_fft
    hop_size = h.hop_size
    win_size = h.win_size
    compress_factor = h.compress_factor

    # nan-guard skip counters (see the non-finite-gradient guards below)
    n_g_step = n_g_skip = n_d_step = n_d_skip = 0

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):

            if rank == 0:
                start_b = time.time()

            # MPSENet dataset returns (reverb_audio, clean_audio, noise_residual)
            reverb_audio, clean_audio, _ = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            reverb_audio = reverb_audio.to(device, non_blocking=True)

            # STFT in training step (MPSENet does not pre-compute STFT in dataset)
            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, n_fft, hop_size, win_size, compress_factor)
            reverb_mag, reverb_pha, _ = mag_pha_stft(reverb_audio, n_fft, hop_size, win_size, compress_factor)

            # Generator forward
            mag_g, pha_g, com_g = generator(reverb_mag, reverb_pha)

            # ISTFT to get enhanced audio
            audio_g = mag_pha_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)

            # Consistency round-trip: STFT of enhanced audio (use safe version to
            # avoid NaN gradients from pow(mag, 0.3) at mag≈0)
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft_safe(audio_g, n_fft, hop_size, win_size, compress_factor)

            one_labels = torch.ones(clean_audio.shape[0]).to(device, non_blocking=True)

            # PESQ for discriminator
            audio_list_r = list(clean_audio.cpu().numpy())
            audio_list_g = list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)

            # --- Discriminator ---
            if not a.freeze_discriminator:
                optim_d.zero_grad()
                metric_r = discriminator(clean_mag, clean_mag)
                metric_g = discriminator(clean_mag, mag_g_hat.detach())
                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

                if batch_pesq_score is not None:
                    loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
                else:
                    loss_disc_g = 0

                loss_disc_all = loss_disc_r + loss_disc_g
                loss_disc_all.backward()
                # Guard: a single NaN/Inf gradient makes clip_grad_norm_ scale ALL
                # grads by NaN; stepping then poisons every weight. Skip such steps.
                d_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
                n_d_step += 1
                if torch.isfinite(d_norm):
                    optim_d.step()
                else:
                    optim_d.zero_grad()
                    n_d_skip += 1
                    if rank == 0:
                        print(f"[nan-guard] skipped D step at step {steps} (grad_norm={d_norm})")
            else:
                loss_disc_all = torch.tensor(0.0)

            # --- Generator (gradient accumulation over a.grad_accum micro-batches) ---
            if i % a.grad_accum == 0:
                optim_g.zero_grad()

            # L2 Magnitude Loss
            loss_mag = F.mse_loss(clean_mag, mag_g)
            # Anti-wrapping Phase Loss
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
            loss_pha = loss_ip + loss_gd + loss_iaf
            # L2 Complex Loss
            loss_com = F.mse_loss(clean_com, com_g) * 2
            # L2 Consistency Loss (ISTFT -> STFT round-trip)
            loss_stft = F.mse_loss(com_g, com_g_hat) * 2
            # Time-domain L1 Loss
            loss_time = F.l1_loss(clean_audio, audio_g)
            # Metric Loss (discriminator uses mag_g_hat from consistency round-trip)
            metric_g = discriminator(clean_mag, mag_g_hat)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

            loss_gen_all = (loss_mag * 0.9 + loss_pha * 0.3 + loss_com * 0.1
                            + loss_stft * 0.1 + loss_metric * 0.05 + loss_time * 0.2)
            # Scale so accumulated grad equals the mean over the effective batch
            # (micro_batch * num_gpus * grad_accum). Step only at window end.
            (loss_gen_all / a.grad_accum).backward()
            if (i + 1) % a.grad_accum == 0:
                # Guard: skip the step if the accumulated gradient is non-finite
                # (e.g. NaN gradient from atan2(0,0) in the phase decoder), which
                # would otherwise poison every weight via clip_grad_norm_.
                g_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
                n_g_step += 1
                if torch.isfinite(g_norm):
                    optim_g.step()
                else:
                    n_g_skip += 1
                    if rank == 0:
                        print(f"[nan-guard] skipped G step at step {steps} (grad_norm={g_norm})")
                optim_g.zero_grad()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        metric_error = loss_metric.item()
                        mag_error = loss_mag.item()
                        pha_error = loss_pha.item()
                        com_error = loss_com.item() / 2  # undo the *2 for display
                        stft_error = loss_stft.item() / 2
                        time_error = loss_time.item()
                    g_skip_pct = 100.0 * n_g_skip / max(1, n_g_step)
                    d_skip_pct = 100.0 * n_d_skip / max(1, n_d_step)
                    print('Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, '
                          'Metric: {:4.3f}, Mag: {:4.3f}, Pha: {:4.3f}, '
                          'Com: {:4.3f}, STFT: {:4.3f}, Time: {:4.3f}, s/b: {:4.3f}, '
                          'nan_skip G={:d}/{:d}({:.2f}%) D={:d}/{:d}({:.2f}%)'.format(
                              steps, loss_gen_all, loss_disc_all,
                              metric_error, mag_error, pha_error,
                              com_error, stft_error, time_error,
                              time.time() - start_b,
                              n_g_skip, n_g_step, g_skip_pct,
                              n_d_skip, n_d_step, d_skip_pct))

                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'discriminator': (discriminator.module if h.num_gpus > 1 else discriminator).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # W&B logging
                if steps % a.summary_interval == 0:
                    wandb.log({
                        "Training/Generator Loss": loss_gen_all.item(),
                        "Training/Discriminator Loss": loss_disc_all.item() if torch.is_tensor(loss_disc_all) else loss_disc_all,
                        "Training/Metric Loss": metric_error,
                        "Training/Magnitude Loss": mag_error,
                        "Training/Phase Loss": pha_error,
                        "Training/Complex Loss": com_error,
                        "Training/Consistency Loss": stft_error,
                        "Training/Time Loss": time_error,
                        "epoch": epoch + 1,
                    }, step=steps)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    torch.cuda.empty_cache()
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    segment_size = h.segment_size
                    gen_model = generator.module if h.num_gpus > 1 else generator
                    with torch.no_grad():
                        # Full-set loss computation
                        for j, batch in enumerate(validation_loader):
                            clean_audio_v, noisy_audio_v = batch
                            noisy_audio_v = noisy_audio_v.to(device, non_blocking=True)
                            clean_audio_v = clean_audio_v.to(device, non_blocking=True)

                            audio_g_v = process_audio(
                                noisy_audio_v.squeeze(0), gen_model, device,
                                segment_size, n_fft, hop_size, win_size, compress_factor)

                            clean_mag_v, clean_pha_v, clean_com_v = mag_pha_stft(
                                clean_audio_v, n_fft, hop_size, win_size, compress_factor)
                            clean_mag_v = clean_mag_v.to(device)
                            clean_pha_v = clean_pha_v.to(device)
                            clean_com_v = clean_com_v.to(device)

                            mag_g_v, pha_g_v, com_g_v = mag_pha_stft(
                                audio_g_v.unsqueeze(0), n_fft, hop_size, win_size, compress_factor)
                            mag_g_v = mag_g_v.to(device)
                            pha_g_v = pha_g_v.to(device)
                            com_g_v = com_g_v.to(device)

                            val_mag_err_tot += F.mse_loss(clean_mag_v.squeeze(), mag_g_v.squeeze()).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha_v, pha_g_v)
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(clean_com_v.squeeze(), com_g_v.squeeze()).item()

                        n_val = j + 1
                        val_mag_err = val_mag_err_tot / n_val
                        val_pha_err = val_pha_err_tot / n_val
                        val_com_err = val_com_err_tot / n_val

                        # Subset for PESQ/STOI/DNSMOS + audio logging
                        n_subset = min(10, len(validset))
                        pesq_scores = []
                        stoi_scores = []
                        sisdr_scores = []
                        dnsmos_scores = []
                        pesq_noisy_scores = []
                        stoi_noisy_scores = []
                        sisdr_noisy_scores = []
                        dnsmos_noisy_scores = []
                        wandb_logs = {}
                        for j in range(n_subset):
                            clean_wav, noisy_wav = validset[j]
                            enhanced_wav = process_audio(
                                noisy_wav.to(device), gen_model, device,
                                segment_size, n_fft, hop_size, win_size, compress_factor)

                            clean_np = clean_wav.numpy()
                            noisy_np = noisy_wav.numpy()
                            enhanced_np = enhanced_wav.numpy()
                            sr = h.sampling_rate

                            try:
                                p = compute_pesq(sr, clean_np, enhanced_np, 'wb')
                            except Exception:
                                p = float('nan')
                            try:
                                s = stoi(clean_np, enhanced_np, sr, extended=False)
                            except Exception:
                                s = float('nan')
                            try:
                                d = dnsmos.run(enhanced_np, sr)
                                d_ovrl = d["ovrl_mos"]
                            except Exception:
                                d_ovrl = float('nan')
                            try:
                                si = si_sdr(clean_np, enhanced_np)
                            except Exception:
                                si = float('nan')
                            pesq_scores.append(p)
                            stoi_scores.append(s)
                            sisdr_scores.append(si)
                            dnsmos_scores.append(d_ovrl)

                            try:
                                p_n = compute_pesq(sr, clean_np, noisy_np, 'wb')
                            except Exception:
                                p_n = float('nan')
                            try:
                                s_n = stoi(clean_np, noisy_np, sr, extended=False)
                            except Exception:
                                s_n = float('nan')
                            try:
                                d_n = dnsmos.run(noisy_np, sr)
                                d_n_ovrl = d_n["ovrl_mos"]
                            except Exception:
                                d_n_ovrl = float('nan')
                            try:
                                si_n = si_sdr(clean_np, noisy_np)
                            except Exception:
                                si_n = float('nan')
                            pesq_noisy_scores.append(p_n)
                            stoi_noisy_scores.append(s_n)
                            sisdr_noisy_scores.append(si_n)
                            dnsmos_noisy_scores.append(d_n_ovrl)

                            if j < 5:
                                caption = (f"sample_{j} | PESQ={p:.2f}/{p_n:.2f} "
                                           f"STOI={s:.3f}/{s_n:.3f} "
                                           f"DNSMOS={d_ovrl:.2f}/{d_n_ovrl:.2f}")
                                prefix = f"Validation/sample_{j}"
                                clean_clipped = clean_np / max(np.abs(clean_np).max(), 1e-8)
                                noisy_clipped = noisy_np / max(np.abs(noisy_np).max(), 1e-8)
                                enhanced_clipped = enhanced_np / max(np.abs(enhanced_np).max(), 1e-8)
                                wandb_logs[f"{prefix}/clean"] = wandb.Audio(
                                    clean_clipped, sample_rate=sr, caption=f"clean | {caption}")
                                wandb_logs[f"{prefix}/noisy"] = wandb.Audio(
                                    noisy_clipped, sample_rate=sr, caption=f"noisy | {caption}")
                                wandb_logs[f"{prefix}/enhanced"] = wandb.Audio(
                                    enhanced_clipped, sample_rate=sr, caption=f"enhanced | {caption}")
                                fig = make_spectrogram_fig(clean_np, noisy_np, enhanced_np,
                                                          sr, n_fft, hop_size)
                                wandb_logs[f"{prefix}/spectrogram"] = wandb.Image(
                                    fig, caption=caption)
                                plt.close(fig)
                                wandb_logs[f"{prefix}/PESQ"] = p
                                wandb_logs[f"{prefix}/STOI"] = s
                                wandb_logs[f"{prefix}/DNSMOS_ovrl"] = d_ovrl
                                wandb_logs[f"{prefix}/PESQ_noisy"] = p_n
                                wandb_logs[f"{prefix}/STOI_noisy"] = s_n
                                wandb_logs[f"{prefix}/DNSMOS_ovrl_noisy"] = d_n_ovrl

                        mean_pesq = float(np.nanmean(pesq_scores))
                        mean_stoi = float(np.nanmean(stoi_scores))
                        mean_sisdr = float(np.nanmean(sisdr_scores))
                        mean_dnsmos = float(np.nanmean(dnsmos_scores))
                        mean_pesq_noisy = float(np.nanmean(pesq_noisy_scores))
                        mean_stoi_noisy = float(np.nanmean(stoi_noisy_scores))
                        mean_sisdr_noisy = float(np.nanmean(sisdr_noisy_scores))
                        mean_dnsmos_noisy = float(np.nanmean(dnsmos_noisy_scores))
                        print('Steps : {:d}, PESQ: {:4.3f} (input {:4.3f}), STOI: {:4.3f} (input {:4.3f}), '
                              'SI-SDR: {:4.3f} (input {:4.3f}), DNSMOS: {:4.3f} (input {:4.3f}), s/b : {:4.3f}'.format(
                                  steps, mean_pesq, mean_pesq_noisy, mean_stoi, mean_stoi_noisy,
                                  mean_sisdr, mean_sisdr_noisy, mean_dnsmos, mean_dnsmos_noisy,
                                  time.time() - start_b))

                        wandb.log({
                            "Validation/Magnitude Loss": val_mag_err,
                            "Validation/Phase Loss": val_pha_err,
                            "Validation/Complex Loss": val_com_err,
                            "Validation/PESQ": mean_pesq,
                            "Validation/STOI": mean_stoi,
                            "Validation/SI-SDR": mean_sisdr,
                            "Validation/DNSMOS": mean_dnsmos,
                            "Validation/PESQ_input": mean_pesq_noisy,
                            "Validation/STOI_input": mean_stoi_noisy,
                            "Validation/SI-SDR_input": mean_sisdr_noisy,
                            "Validation/DNSMOS_input": mean_dnsmos_noisy,
                            "epoch": epoch + 1,
                            **wandb_logs,
                        }, step=steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if a.save_every_epoch and rank == 0:
            epoch_dir = os.path.join(a.checkpoint_path, f'epoch_{epoch+1}')
            os.makedirs(epoch_dir, exist_ok=True)
            save_checkpoint(os.path.join(epoch_dir, f'g_{steps:08d}'),
                            {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
            save_checkpoint(os.path.join(epoch_dir, f'do_{steps:08d}'),
                            {'discriminator': (discriminator.module if h.num_gpus > 1 else discriminator).state_dict(),
                             'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                             'epoch': epoch})

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing MPSENet Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_clean_wavs_dir', default='./VB_DEMAND_16K/clean_train')
    parser.add_argument('--input_noisy_wavs_dir', default='./VB_DEMAND_16K/noisy_train')
    parser.add_argument('--input_training_file', default='VoiceBank+DEMAND/training.txt')
    parser.add_argument('--input_validation_file', default='VoiceBank+DEMAND/test.txt')
    parser.add_argument('--checkpoint_path', default='checkpoints_mpsenet')
    parser.add_argument('--config', default='config_mpsenet.json')
    parser.add_argument('--training_epochs', default=200, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)

    # Fine-tuning arguments
    parser.add_argument('--mode', default='reverb', choices=['reverb'])
    parser.add_argument('--pretrained_checkpoint', default=None, type=str,
                        help='Path to pretrained checkpoint (PL .ckpt or our format)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Override learning_rate from config')
    parser.add_argument('--grad_accum', default=1, type=int,
                        help='Generator gradient-accumulation steps. Effective global '
                             'batch = (config batch_size) * grad_accum.')
    parser.add_argument('--unfreeze', default=None, type=str,
                        help='Block to unfreeze: all|encoder|enhancer|enhancer_0|...|enhancer_3|'
                             'mask_decoder|phase_decoder|decoder')
    parser.add_argument('--unfreeze_boundary', action='store_true',
                        help='Also unfreeze encoder + decoder (boundary modules)')
    parser.add_argument('--freeze_discriminator', action='store_true',
                        help='Freeze MetricDiscriminator')
    parser.add_argument('--rir_dir', default='data/rirs_clipped', type=str,
                        help='Root directory of raw RIR dataset')
    parser.add_argument('--rir_metadata', default='data/rir_metadata.csv', type=str,
                        help='RIR metadata CSV (filename, wav_path, family, rt60, ...)')
    parser.add_argument('--rir_split', default='data/rir_split.json', type=str)
    parser.add_argument('--save_every_epoch', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Save checkpoint at end of each epoch (disable with --no-save_every_epoch)')
    parser.add_argument('--val_rir_ratio', default=0.05, type=float,
                        help='Fraction of train RIRs to hold out for validation (default: 0.05)')
    parser.add_argument('--val_clean_wavs_dir', default=None, type=str,
                        help='Validation clean wavs dir (defaults to --input_clean_wavs_dir)')
    parser.add_argument('--val_noisy_wavs_dir', default=None, type=str,
                        help='Validation noisy wavs dir (defaults to --input_noisy_wavs_dir)')

    a = parser.parse_args()

    # Validate --unfreeze
    if a.unfreeze is not None:
        valid_blocks = list(BLOCK_TO_PREFIXES.keys()) + ['all']
        if a.unfreeze not in valid_blocks:
            parser.error(f"--unfreeze: invalid value '{a.unfreeze}'. "
                         f"Choose from: {valid_blocks}")

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        master_port = os.environ.get('MASTER_PORT')
        if master_port:
            h.dist_config['dist_url'] = f'tcp://localhost:{master_port}'
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
