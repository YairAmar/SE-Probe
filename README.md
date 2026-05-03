# SE-Probe

📄 **Paper:** <https://arxiv.org/abs/2512.00482> &nbsp;·&nbsp; 📖 **Book:** <https://yairamar.github.io/SE-Probe/>

## Background

Public companion code for *"Where Does Speech Enhancement Adapt? Probing Study Under Controlled Degradation"* (Amar, Ivry, Cohen, 2026; [arXiv:2512.00482](https://arxiv.org/abs/2512.00482)). Speech enhancement networks are treated as black boxes: clean and degraded utterances are pushed through a frozen SE model, activations are extracted layer by layer, clean and degraded representations are compared by linear CKA, and the resulting curves are regressed against degradation severity (SNR or C50). Diffusion-map distances and downstream PESQ correlations cross check the picture from a different angle. The analysis is run end to end across MUSE, MP-SENet, and Demucs.

The full narrative lives in the book linked above. This README is just enough to get the code running.

## Get going

```bash
git clone https://github.com/YairAmar/SE-Probe.git && cd SE-Probe
pip install -e .
python scripts/setup.py
jupyter lab notebooks/
```

`scripts/setup.py` fetches the upstream MUSE pretrained weights, downloads the epoch-48 reverb fine-tuned checkpoint from HuggingFace (`yairamr/SE-Probe-models`), and probes the local device (CUDA, MPS, or CPU; autodetected). The six notebooks read precomputed CKA tables from `results_demo/` (about 200 KB, shipped in-tree) and run on any laptop in seconds. Add `--full-data` to `setup.py` to also pull the 3.1 GB full precomputed tables from HuggingFace (`yairamr/SE-Probe-data`). Notebooks 01 and 06 optionally re-run model inference if hardware is available.

The reverb fine-tuning training pipeline that produced the checkpoint is vendored under `training/` (originally [muse-dereverb-ft](https://github.com/YairAmar/muse-dereverb-ft)). See `training/README.md` for retraining instructions.

## Citation

```bibtex
@article{amar2026seprobe,
  title         = {Where Does Speech Enhancement Adapt? Probing Study Under Controlled Degradation},
  author        = {Amar, Yair and Ivry, Amir and Cohen, Israel},
  year          = {2026},
  eprint        = {2512.00482},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2512.00482}
}
```

## License

MIT, see `LICENSE`. Open an issue at <https://github.com/YairAmar/SE-Probe/issues>.
