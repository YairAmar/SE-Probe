"""Time-domain DEREVERBERATION fine-tuning for Demucs (denoiser DNS64).

Mirrors the MUSE / MP-SENet reverb fine-tuning protocol, but in the time domain:

  * Model:      denoiser ``dns64()`` (pretrained DNS64), all params unfrozen.
                ``Demucs.forward`` internally pads to ``valid_length`` and crops
                back, so we just pass the waveform; no external padding needed.
  * Loss:       L1(enhanced, clean) + Multi-Resolution STFT loss
                (spectral-convergence + log-STFT-magnitude over FFT {512,1024,2048}).
                The MRSTFT module is lifted verbatim from the installed
                ``denoiser.stft_loss`` package and combined exactly as in
                ``denoiser.solver`` (``loss += sc_loss + mag_loss``).
  * Optim:      AdamW(betas=[0.8,0.99], lr=1e-4), ExponentialLR(gamma=0.99)
                stepped per epoch, batch 28, 50 epochs (MUSE-protocol parity).
  * Checkpoint: per-epoch ``{checkpoint_path}/epoch_{N}/g_{step:08d}`` holding
                ``{'generator': model.state_dict()}`` -- reloads into a fresh
                ``dns64()`` (the probing side loads dns64() then applies this).
  * Val:        PESQ (wb) + STOI on a held-out set each epoch; best-PESQ epoch
                is selectable post-hoc from the logged history.

RIR train/val holdout replicates the MP-SENet trainer: hold out ``val_rir_ratio``
of the train RIRs (seed = config seed) for validation, leaving the RIR-split
"test" entries untouched for final evaluation.
"""

import argparse
import csv
import json
import os
import random as stdlib_random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pesq import pesq as compute_pesq
from pystoi import stoi as compute_stoi

from denoiser.pretrained import dns64
from denoiser.stft_loss import MultiResolutionSTFTLoss

from datasets.reverb_waveform_dataset import DemucsReverbDataset, DemucsReverbValDataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_dataset_filelist(training_file, validation_file):
    with open(training_file, "r", encoding="utf-8") as fi:
        training_indexes = [x.split("|")[0] for x in fi.read().split("\n") if len(x) > 0]
    with open(validation_file, "r", encoding="utf-8") as fi:
        validation_indexes = [x.split("|")[0] for x in fi.read().split("\n") if len(x) > 0]
    return training_indexes, validation_indexes


def build_rir_holdout(a, seed):
    """Return (train_rir_paths, val_rir_paths). Holds out ``val_rir_ratio`` of
    the train-split RIRs for validation (seeded), same as the MP-SENet trainer."""
    rir_meta = {}
    with open(a.rir_metadata, "r") as f:
        for row in csv.DictReader(f):
            rir_meta[row["filename"]] = row["wav_path"]

    with open(a.rir_split, "r") as f:
        rir_split = json.load(f)
    train_rir_fns = sorted([fn for fn, s in rir_split.items() if s == "train"])

    rng = stdlib_random.Random(seed)
    shuffled = list(train_rir_fns)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * a.val_rir_ratio))
    val_rir_fns = shuffled[:n_val]
    actual_train_fns = shuffled[n_val:]

    train_rir_paths = [rir_meta[fn] for fn in actual_train_fns]
    val_rir_paths = [rir_meta[fn] for fn in val_rir_fns]
    print(f"RIR split: {len(train_rir_fns)} train total -> "
          f"{len(train_rir_paths)} train, {len(val_rir_paths)} val holdout")
    return train_rir_paths, val_rir_paths


def validate(model, validset, device, sr, n_subset):
    """Compute mean PESQ (wb) + STOI over a small subset of full-length utts."""
    model.eval()
    pesq_scores, stoi_scores = [], []
    pesq_noisy, stoi_noisy = [], []
    with torch.no_grad():
        for j in range(min(n_subset, len(validset))):
            clean_wav, reverb_wav = validset[j]
            enhanced = model(reverb_wav.unsqueeze(0).to(device)).squeeze().cpu()
            clean_np = clean_wav.numpy()
            reverb_np = reverb_wav.numpy()
            enh_np = enhanced.numpy()
            try:
                pesq_scores.append(compute_pesq(sr, clean_np, enh_np, "wb"))
            except Exception:
                pesq_scores.append(float("nan"))
            try:
                stoi_scores.append(compute_stoi(clean_np, enh_np, sr, extended=False))
            except Exception:
                stoi_scores.append(float("nan"))
            try:
                pesq_noisy.append(compute_pesq(sr, clean_np, reverb_np, "wb"))
            except Exception:
                pesq_noisy.append(float("nan"))
            try:
                stoi_noisy.append(compute_stoi(clean_np, reverb_np, sr, extended=False))
            except Exception:
                stoi_noisy.append(float("nan"))
    model.train()
    return (float(np.nanmean(pesq_scores)), float(np.nanmean(stoi_scores)),
            float(np.nanmean(pesq_noisy)), float(np.nanmean(stoi_noisy)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_finetune_demucs.json")
    parser.add_argument("--checkpoint_path", default="checkpoints/dereverb_demucs")
    parser.add_argument("--input_clean_wavs_dir", default="data/VB_DEMAND_16K/clean_train")
    parser.add_argument("--input_training_file", default="VoiceBank+DEMAND/training.txt")
    parser.add_argument("--input_validation_file", default="VoiceBank+DEMAND/test.txt")
    parser.add_argument("--val_clean_wavs_dir", default="data/VB_DEMAND_16K/clean_test")
    parser.add_argument("--rir_dir", default="data/rirs_clipped")
    parser.add_argument("--rir_metadata", default="data/rir_metadata.csv")
    parser.add_argument("--rir_split", default="data/rir_split.json")
    parser.add_argument("--val_rir_ratio", type=float, default=0.05)
    parser.add_argument("--training_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None, help="override config learning_rate")
    parser.add_argument("--val_subset", type=int, default=30)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="smoke-test cap on optimizer steps per epoch")
    parser.add_argument("--no_wandb", action="store_true")
    a = parser.parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    lr = a.lr if a.lr is not None else h.learning_rate
    torch.manual_seed(h.seed)
    np.random.seed(h.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ----
    training_indexes, validation_indexes = get_dataset_filelist(
        a.input_training_file, a.input_validation_file)
    train_rir_paths, val_rir_paths = build_rir_holdout(a, h.seed)

    trainset = DemucsReverbDataset(
        training_indexes, a.input_clean_wavs_dir, a.rir_dir, train_rir_paths,
        h.segment_size, h.sampling_rate, shuffle=True, seed=h.seed)
    train_loader = DataLoader(
        trainset, num_workers=h.num_workers, shuffle=True, batch_size=h.batch_size,
        pin_memory=True, drop_last=True,
        persistent_workers=True if h.num_workers > 0 else False)

    validset = DemucsReverbValDataset(
        validation_indexes, a.val_clean_wavs_dir, a.rir_dir, val_rir_paths, h.sampling_rate)

    # ---- model / optim / loss ----
    model = dns64().to(device)          # pretrained DNS64, all params trainable
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"dns64 loaded: {total_params/1e6:.2f}M params (all trainable)")

    optim = torch.optim.AdamW(model.parameters(), lr, betas=[h.adam_b1, h.adam_b2])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=h.lr_decay)
    # MRSTFT lifted verbatim from denoiser.stft_loss; factors from config.
    mrstftloss = MultiResolutionSTFTLoss(
        factor_sc=h.stft_sc_factor, factor_mag=h.stft_mag_factor).to(device)

    os.makedirs(a.checkpoint_path, exist_ok=True)

    use_wandb = not a.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="Demucs-Reverb-FN", name="dereverb-full-ft",
                       config={**dict(h), "lr": lr, "training_epochs": a.training_epochs,
                               "total_params": total_params})
        except Exception as e:
            print(f"wandb disabled ({e})")
            use_wandb = False

    history = []
    steps = 0
    best_pesq = -1e9
    best_epoch = -1
    for epoch in range(a.training_epochs):
        start = time.time()
        running = 0.0
        n_batches = 0
        for i, (clean, reverb) in enumerate(train_loader):
            clean = clean.to(device)              # [B, T]
            reverb = reverb.to(device)            # [B, T]
            enhanced = model(reverb.unsqueeze(1)).squeeze(1)   # [B, T]

            loss = F.l1_loss(enhanced, clean)
            sc_loss, mag_loss = mrstftloss(enhanced, clean)
            loss = loss + sc_loss + mag_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            steps += 1
            running += loss.item()
            n_batches += 1
            if steps % 50 == 0:
                print(f"epoch {epoch+1} step {steps} | loss {loss.item():.5f} "
                      f"(l1+sc+mag) | lr {optim.param_groups[0]['lr']:.2e}", flush=True)
                if use_wandb:
                    wandb.log({"train/loss": loss.item(),
                               "train/l1": F.l1_loss(enhanced, clean).item(),
                               "train/sc": sc_loss.item(), "train/mag": mag_loss.item(),
                               "lr": optim.param_groups[0]["lr"]}, step=steps)
            if a.max_train_steps is not None and n_batches >= a.max_train_steps:
                break

        scheduler.step()                          # ExponentialLR per epoch
        train_loss = running / max(1, n_batches)

        # ---- per-epoch checkpoint: {'generator': state_dict} into epoch_{N}/g_{step} ----
        epoch_dir = os.path.join(a.checkpoint_path, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        ckpt_file = os.path.join(epoch_dir, f"g_{steps:08d}")
        torch.save({"generator": model.state_dict()}, ckpt_file)

        # ---- validation ----
        pesq_e, stoi_e, pesq_n, stoi_n = validate(
            model, validset, device, h.sampling_rate, a.val_subset)
        if pesq_e > best_pesq:
            best_pesq = pesq_e
            best_epoch = epoch + 1
        metrics = {"epoch": epoch + 1, "steps": steps, "train_loss": train_loss,
                   "pesq": pesq_e, "stoi": stoi_e,
                   "pesq_reverb": pesq_n, "stoi_reverb": stoi_n,
                   "checkpoint": ckpt_file, "time_s": time.time() - start}
        history.append(metrics)
        print(f"=== Epoch {epoch+1}/{a.training_epochs} | train_loss {train_loss:.5f} | "
              f"PESQ {pesq_e:.4f} (reverb {pesq_n:.4f}) | STOI {stoi_e:.4f} "
              f"(reverb {stoi_n:.4f}) | best PESQ {best_pesq:.4f} @ ep{best_epoch} | "
              f"{time.time()-start:.1f}s | ckpt {ckpt_file}", flush=True)
        with open(os.path.join(a.checkpoint_path, "history.json"), "w") as f:
            json.dump({"best_epoch": best_epoch, "best_pesq": best_pesq,
                       "history": history}, f, indent=2)
        if use_wandb:
            wandb.log({"val/pesq": pesq_e, "val/stoi": stoi_e,
                       "val/pesq_reverb": pesq_n, "val/stoi_reverb": stoi_n,
                       "train/epoch_loss": train_loss, "epoch": epoch + 1}, step=steps)

    print(f"DONE. best PESQ {best_pesq:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
