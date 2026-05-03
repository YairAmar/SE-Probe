"""Evaluate a MUSE checkpoint on reverb and/or noise test sets.

Usage:
    python evaluate.py \
      --checkpoint paper_result/g_best \
      --test_set both \
      --config config.json \
      --output results/baseline.csv
"""

import os
import csv
import argparse
import json
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from pesq import pesq
from pystoi import stoi

from env import AttrDict
from datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import MUSE


class GPUDNSMOSEvaluator:
    """GPU-accelerated DNSMOS evaluator using onnx2torch conversion.

    Converts ONNX DNSMOS models to native PyTorch for GPU inference,
    bypassing onnxruntime CUDA provider requirements.
    Based on seint.metrics.GPUDNSMOSEvaluator.
    """

    def __init__(self, sample_rate=16000, device='cuda'):
        from onnx2torch import convert
        import speechmos

        self.sample_rate = sample_rate
        self.device = device
        self.INPUT_LENGTH = 9.01
        self.len_samples = int(self.INPUT_LENGTH * sample_rate)

        # Load and convert ONNX models to PyTorch
        pkg_dir = os.path.dirname(speechmos.__file__)
        primary_path = os.path.join(pkg_dir, "dnsmos_models", "sig_bak_ovr.onnx")
        p808_path = os.path.join(pkg_dir, "dnsmos_models", "model_v8.onnx")

        self.primary_model = convert(primary_path).to(device).eval()
        self.p808_model = convert(p808_path).to(device).eval()

        # Mel spectrogram transform on GPU
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=321,
            hop_length=160,
            n_mels=120,
            power=2.0,
        ).to(device)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80,
        ).to(device)

        # Polynomial coefficients for score mapping (same as speechmos)
        self._p_ovr = [-0.06766283, 1.11546468, 0.04602535]
        self._p_sig = [-0.08397278, 1.22083953, 0.0052439]
        self._p_bak = [-0.13166888, 1.60915514, -0.39604546]

    @staticmethod
    def _polyval(coeffs, x):
        result = 0.0
        for c in coeffs:
            result = result * x + c
        return result

    def run(self, audio, sr):
        """Score a single audio waveform. Same return dict as speechmos.dnsmos.run()."""
        if sr != self.sample_rate:
            raise ValueError(f"Sampling rate must be {self.sample_rate}.")

        # Convert to GPU tensor
        audio_t = torch.from_numpy(audio).float().to(self.device).squeeze()

        # Pad to required length
        while len(audio_t) < self.len_samples:
            audio_t = torch.cat([audio_t, audio_t])
        audio_t = audio_t[:self.len_samples]

        with torch.no_grad():
            # Primary model: raw audio -> SIG, BAK, OVR
            input_features = audio_t.unsqueeze(0)
            primary_out = self.primary_model(input_features)
            sig_raw, bak_raw, ovr_raw = primary_out[0].cpu().numpy()

            # P808 model: mel spectrogram -> P808 MOS
            mel_spec = self.mel_transform(audio_t[:-160])
            mel_spec_db = self.amp_to_db(mel_spec)
            mel_spec_norm = (mel_spec_db + 40) / 40
            p808_input = mel_spec_norm.T.unsqueeze(0)
            p808_out = self.p808_model(p808_input)
            p808_mos = p808_out.squeeze().item()

        return {
            'ovrl_mos': float(self._polyval(self._p_ovr, ovr_raw)),
            'sig_mos': float(self._polyval(self._p_sig, sig_raw)),
            'bak_mos': float(self._polyval(self._p_bak, bak_raw)),
            'p808_mos': float(p808_mos),
        }


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath), f"Checkpoint not found: {filepath}"
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


def si_sdr(ref, est):
    """Scale-Invariant Signal-to-Distortion Ratio."""
    ref = ref - ref.mean()
    est = est - est.mean()
    s_target = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8) * ref
    e_noise = est - s_target
    return 10 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8))


def process_audio(noisy_wav, model, h, device):
    """Run model inference on a full-length waveform, segment by segment."""
    segment_size = h.segment_size
    n_fft = h.n_fft
    hop_size = h.hop_size
    win_size = h.win_size
    compress_factor = h.compress_factor

    noisy_wav = torch.FloatTensor(noisy_wav).to(device)
    norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    orig_size = noisy_wav.size(1)

    if noisy_wav.size(1) >= segment_size:
        last_segment_size = noisy_wav.size(1) % segment_size
        if last_segment_size > 0:
            last_segment = noisy_wav[:, -segment_size:]
            noisy_wav_main = noisy_wav[:, :-last_segment_size]
            segments = list(torch.split(noisy_wav_main, segment_size, dim=1))
            segments.append(last_segment)
            reshapelast = 1
        else:
            segments = list(torch.split(noisy_wav, segment_size, dim=1))
            reshapelast = 0
            last_segment_size = 0
    else:
        padded_zeros = torch.zeros(1, segment_size - noisy_wav.size(1)).to(device)
        noisy_wav = torch.cat((noisy_wav, padded_zeros), dim=1)
        segments = [noisy_wav]
        reshapelast = 0
        last_segment_size = 0

    processed_segments = []
    for i, segment in enumerate(segments):
        noisy_amp, noisy_pha, _ = mag_pha_stft(segment, n_fft, hop_size, win_size, compress_factor)
        amp_g, pha_g, _ = model(noisy_amp.to(device), noisy_pha.to(device))
        audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
        audio_g = audio_g / norm_factor
        audio_g = audio_g.squeeze()
        if reshapelast == 1 and i == len(segments) - 2:
            audio_g = audio_g[:-(segment_size - last_segment_size)]
        processed_segments.append(audio_g)

    processed_audio = torch.cat(processed_segments, dim=-1)
    processed_audio = processed_audio[:orig_size]
    return processed_audio.cpu().numpy()


def evaluate_test_set(model, h, device, wavs_dir, clean_dir, metadata_path, domain, gpu_dnsmos, sr=16000):
    """Evaluate model on a test set, return per-utterance results."""
    results = []

    with open(metadata_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        filepath = row["filepath"]
        utt_id = row["utterance_id"]

        # Load noisy/reverb audio
        noisy_path = os.path.join(wavs_dir, filepath)
        noisy, _ = librosa.load(noisy_path, sr=sr)

        # Load clean reference
        clean_filename = f"{utt_id}.wav"
        clean_path = os.path.join(clean_dir, clean_filename)
        clean, _ = librosa.load(clean_path, sr=sr)

        # Run model
        enhanced = process_audio(noisy, model, h, device)

        # Align lengths
        min_len = min(len(clean), len(enhanced), len(noisy))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        noisy = noisy[:min_len]

        # Enhanced metrics
        try:
            pesq_val = pesq(sr, clean, enhanced, "wb")
        except Exception:
            pesq_val = -1.0

        try:
            stoi_val = stoi(clean, enhanced, sr, extended=False)
        except Exception:
            stoi_val = -1.0

        sisdr_val = si_sdr(clean, enhanced)

        try:
            dnsmos_enh = gpu_dnsmos.run(enhanced, sr)
        except Exception:
            dnsmos_enh = {"ovrl_mos": -1.0, "sig_mos": -1.0, "bak_mos": -1.0}

        # Input baseline metrics
        try:
            pesq_input = pesq(sr, clean, noisy, "wb")
        except Exception:
            pesq_input = -1.0

        try:
            stoi_input = stoi(clean, noisy, sr, extended=False)
        except Exception:
            stoi_input = -1.0

        sisdr_input = si_sdr(clean, noisy)

        try:
            dnsmos_inp = gpu_dnsmos.run(noisy, sr)
        except Exception:
            dnsmos_inp = {"ovrl_mos": -1.0, "sig_mos": -1.0, "bak_mos": -1.0}

        result = {
            "domain": domain,
            "utterance_id": utt_id,
            "pesq": f"{pesq_val:.4f}",
            "stoi": f"{stoi_val:.4f}",
            "si_sdr": f"{sisdr_val:.2f}",
            "dnsmos_ovrl": f"{dnsmos_enh['ovrl_mos']:.4f}",
            "dnsmos_sig": f"{dnsmos_enh['sig_mos']:.4f}",
            "dnsmos_bak": f"{dnsmos_enh['bak_mos']:.4f}",
            "pesq_input": f"{pesq_input:.4f}",
            "stoi_input": f"{stoi_input:.4f}",
            "si_sdr_input": f"{sisdr_input:.2f}",
            "dnsmos_ovrl_input": f"{dnsmos_inp['ovrl_mos']:.4f}",
            "dnsmos_sig_input": f"{dnsmos_inp['sig_mos']:.4f}",
            "dnsmos_bak_input": f"{dnsmos_inp['bak_mos']:.4f}",
        }

        # Add reverb-specific metadata
        if "rir_id" in row:
            result["rir_id"] = row["rir_id"]
        if "rt60" in row:
            result["rt60"] = row["rt60"]
        if "c50" in row:
            result["c50"] = row["c50"]

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint")
    parser.add_argument("--test_set", default="both", choices=["reverb", "noise", "both"])
    parser.add_argument("--reverb_dir", default="test_sets/reverb")
    parser.add_argument("--noise_dir", default="test_sets/noise")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        h = AttrDict(json.loads(f.read()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MUSE(h).to(device)
    state_dict = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state_dict["generator"])
    model.eval()

    gpu_dnsmos = GPUDNSMOSEvaluator(device=device)
    print(f"DNSMOS models loaded on {device}")

    all_results = []

    with torch.no_grad():
        if args.test_set in ("reverb", "both"):
            print("Evaluating on reverb test set...")
            reverb_results = evaluate_test_set(
                model, h, device,
                os.path.join(args.reverb_dir, "wavs"),
                os.path.join(args.reverb_dir, "clean_refs"),
                os.path.join(args.reverb_dir, "metadata.csv"),
                domain="reverb",
                gpu_dnsmos=gpu_dnsmos,
            )
            all_results.extend(reverb_results)
            # Summary
            pesq_vals = [float(r["pesq"]) for r in reverb_results if float(r["pesq"]) > 0]
            stoi_vals = [float(r["stoi"]) for r in reverb_results if float(r["stoi"]) > 0]
            sisdr_vals = [float(r["si_sdr"]) for r in reverb_results]
            dnsmos_vals = [float(r["dnsmos_ovrl"]) for r in reverb_results if float(r["dnsmos_ovrl"]) > 0]
            pesq_inp = [float(r["pesq_input"]) for r in reverb_results if float(r["pesq_input"]) > 0]
            stoi_inp = [float(r["stoi_input"]) for r in reverb_results if float(r["stoi_input"]) > 0]
            sisdr_inp = [float(r["si_sdr_input"]) for r in reverb_results]
            dnsmos_inp = [float(r["dnsmos_ovrl_input"]) for r in reverb_results if float(r["dnsmos_ovrl_input"]) > 0]
            print(f"  Reverb enhanced: PESQ={np.mean(pesq_vals):.3f}, STOI={np.mean(stoi_vals):.4f}, "
                  f"SI-SDR={np.mean(sisdr_vals):.2f}dB, DNSMOS={np.mean(dnsmos_vals):.3f}")
            print(f"  Reverb input:    PESQ={np.mean(pesq_inp):.3f}, STOI={np.mean(stoi_inp):.4f}, "
                  f"SI-SDR={np.mean(sisdr_inp):.2f}dB, DNSMOS={np.mean(dnsmos_inp):.3f}")
            print(f"  Reverb delta:    PESQ={np.mean(pesq_vals)-np.mean(pesq_inp):+.3f}, "
                  f"STOI={np.mean(stoi_vals)-np.mean(stoi_inp):+.4f}, "
                  f"SI-SDR={np.mean(sisdr_vals)-np.mean(sisdr_inp):+.2f}dB, "
                  f"DNSMOS={np.mean(dnsmos_vals)-np.mean(dnsmos_inp):+.3f} "
                  f"({len(reverb_results)} utterances)")

        if args.test_set in ("noise", "both"):
            print("Evaluating on noise test set...")
            noise_results = evaluate_test_set(
                model, h, device,
                os.path.join(args.noise_dir, "wavs"),
                os.path.join(args.noise_dir, "clean_refs"),
                os.path.join(args.noise_dir, "metadata.csv"),
                domain="noise",
                gpu_dnsmos=gpu_dnsmos,
            )
            all_results.extend(noise_results)
            pesq_vals = [float(r["pesq"]) for r in noise_results if float(r["pesq"]) > 0]
            stoi_vals = [float(r["stoi"]) for r in noise_results if float(r["stoi"]) > 0]
            sisdr_vals = [float(r["si_sdr"]) for r in noise_results]
            dnsmos_vals = [float(r["dnsmos_ovrl"]) for r in noise_results if float(r["dnsmos_ovrl"]) > 0]
            pesq_inp = [float(r["pesq_input"]) for r in noise_results if float(r["pesq_input"]) > 0]
            stoi_inp = [float(r["stoi_input"]) for r in noise_results if float(r["stoi_input"]) > 0]
            sisdr_inp = [float(r["si_sdr_input"]) for r in noise_results]
            dnsmos_inp = [float(r["dnsmos_ovrl_input"]) for r in noise_results if float(r["dnsmos_ovrl_input"]) > 0]
            print(f"  Noise enhanced: PESQ={np.mean(pesq_vals):.3f}, STOI={np.mean(stoi_vals):.4f}, "
                  f"SI-SDR={np.mean(sisdr_vals):.2f}dB, DNSMOS={np.mean(dnsmos_vals):.3f}")
            print(f"  Noise input:    PESQ={np.mean(pesq_inp):.3f}, STOI={np.mean(stoi_inp):.4f}, "
                  f"SI-SDR={np.mean(sisdr_inp):.2f}dB, DNSMOS={np.mean(dnsmos_inp):.3f}")
            print(f"  Noise delta:    PESQ={np.mean(pesq_vals)-np.mean(pesq_inp):+.3f}, "
                  f"STOI={np.mean(stoi_vals)-np.mean(stoi_inp):+.4f}, "
                  f"SI-SDR={np.mean(sisdr_vals)-np.mean(sisdr_inp):+.2f}dB, "
                  f"DNSMOS={np.mean(dnsmos_vals)-np.mean(dnsmos_inp):+.3f} "
                  f"({len(noise_results)} utterances)")

    # Write results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = ["domain", "utterance_id", "rir_id", "rt60", "c50",
                   "pesq", "stoi", "si_sdr", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak",
                   "pesq_input", "stoi_input", "si_sdr_input",
                   "dnsmos_ovrl_input", "dnsmos_sig_input", "dnsmos_bak_input"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Results saved to {args.output}")

    # Per-RT60-bin breakdown (if reverb results have rt60 field)
    if args.test_set in ("reverb", "both"):
        reverb_only = [r for r in all_results if r["domain"] == "reverb"]
        if reverb_only and "rt60" in reverb_only[0] and reverb_only[0]["rt60"]:
            print("\n--- Per-RT60 breakdown ---")
            rt60_bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.5)]
            for lo, hi in rt60_bins:
                rs = [r for r in reverb_only if lo <= float(r["rt60"]) < hi]
                if not rs:
                    continue
                p = np.mean([float(r["pesq"]) for r in rs if float(r["pesq"]) > 0])
                s = np.mean([float(r["stoi"]) for r in rs if float(r["stoi"]) > 0])
                d = np.mean([float(r["dnsmos_ovrl"]) for r in rs if float(r["dnsmos_ovrl"]) > 0])
                si = np.mean([float(r["si_sdr"]) for r in rs])
                print(f"  RT60=[{lo:.1f}, {hi:.1f})s: PESQ={p:.3f}  STOI={s:.4f}  "
                      f"SI-SDR={si:.2f} dB  DNSMOS={d:.3f}  (n={len(rs)})")


if __name__ == "__main__":
    main()
