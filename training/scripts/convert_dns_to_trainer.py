"""Convert JacobLinCool/MP-SENet-DNS generator weights into the trainer's
MPNet key layout, verify a strict load, and prove a lossless round-trip back
into the JacobLinCool MPSENet (seint probing) key layout.

DNS (JacobLinCool MPSENet) prefix   ->  trainer MPNet prefix
  dense_encoder.                    ->  encoder.
  TSTransformer.                    ->  enhancer.
  mask_decoder.                     ->  decoder.mask_decoder.
  phase_decoder.                    ->  decoder.phase_decoder.
"""
import sys, os, argparse
# Resolve the training/ root relative to this file so the local models/ package imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

HF_REPO = "JacobLinCool/MP-SENet-DNS"
# The exact snapshot the paper's fine-tune started from (md5 of model.safetensors:
# 7d35225dcbce21eb98edd5486c38a290). Override with --revision to pin another.
HF_REVISION = "8b78493f536df1aa53bd3bcbb2f620f705e8589c"

# DNS config.json h.* — the STFT/model hyperparams the DNS weights were trained with.
DNS = dict(dense_channel=64, n_fft=400, hop_size=100, win_size=400,
           beta=2.0, compress_factor=0.3, num_tsblocks=4)

HF2LOCAL = [
    ("dense_encoder.", "encoder."),
    ("TSTransformer.", "enhancer."),
    ("mask_decoder.", "decoder.mask_decoder."),
    ("phase_decoder.", "decoder.phase_decoder."),
]

def hf_to_local(k):
    for hf, loc in HF2LOCAL:
        if k.startswith(hf):
            return loc + k[len(hf):]
    raise KeyError(f"unmapped HF key: {k}")

def local_to_hf(k):
    # longest local prefix first so 'decoder.mask_decoder.' wins over any 'decoder.'
    for hf, loc in sorted(HF2LOCAL, key=lambda p: -len(p[1])):
        if k.startswith(loc):
            return hf + k[len(loc):]
    raise KeyError(f"unmapped local key: {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints_mpsenet/dns_base_converted.pt")
    ap.add_argument("--revision", default=HF_REVISION,
                    help="HF snapshot revision of JacobLinCool/MP-SENet-DNS")
    args = ap.parse_args()

    snap = snapshot_download(HF_REPO, revision=args.revision)
    sd_hf = load_file(os.path.join(snap, "model.safetensors"))
    print(f"[hf] loaded {len(sd_hf)} tensors from DNS safetensors")
    print(f"[hf] mask_decoder.lsigmoid.slope shape = "
          f"{tuple(sd_hf['mask_decoder.lsigmoid.slope'].shape)} (expect (201, 1) at n_fft=400)")

    # ---- forward remap: HF -> local ----
    sd_local = {hf_to_local(k): v for k, v in sd_hf.items()}

    # ---- strict load into trainer's MPNet at DNS hyperparams ----
    from models.mpsenet_generator import MPNet
    gen = MPNet(dense_channel=DNS["dense_channel"], n_fft=DNS["n_fft"],
                sigmoid_beta=DNS["beta"], num_tsblocks=DNS["num_tsblocks"])
    res = gen.load_state_dict(sd_local, strict=True)
    print(f"[load] strict=True OK  missing={list(res.missing_keys)}  "
          f"unexpected={list(res.unexpected_keys)}")

    # ---- round-trip: local (from the live model) -> HF -> compare to original ----
    sd_local_roundtrip = gen.state_dict()
    sd_hf_back = {local_to_hf(k): v for k, v in sd_local_roundtrip.items()}
    assert set(sd_hf_back.keys()) == set(sd_hf.keys()), "key set differs after round-trip"
    max_diff = max((sd_hf_back[k].float() - sd_hf[k].float()).abs().max().item()
                   for k in sd_hf)
    print(f"[roundtrip] local->HF key set matches DNS exactly; max |Δ| vs original = {max_diff:.3e}")

    # ---- also verify it loads into the actual seint MPSENet class ----
    try:
        from MPSENet import MPSENet
        probe = MPSENet.from_pretrained("JacobLinCool/MP-SENet-DNS")
        target = probe.model if hasattr(probe, "model") else probe
        res2 = target.load_state_dict(sd_hf_back, strict=True)
        print(f"[seint] MPSENet.load_state_dict(strict=True) OK  "
              f"missing={list(res2.missing_keys)}  unexpected={list(res2.unexpected_keys)}")
    except Exception as e:
        print(f"[seint] WARN could not verify against live MPSENet class: {e!r}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"generator": sd_local}, args.out)
    print(f"[save] wrote trainer-format base checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
