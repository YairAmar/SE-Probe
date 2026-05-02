# External datasets

The SE-Probe **demo** notebooks (02–05, plus the figure cells of 01 and 06) read precomputed CKA tables from `results_demo/` and **do not need any of the datasets below**. You only need them if you want to:

- recompute CKA from raw audio,
- run end-to-end inference in notebooks 01 / 06 on new utterances, or
- regenerate `results_df/` from scratch.

Point the env vars below at local copies before running the relevant scripts. `se_probe.consts.set_paths(...)` accepts the same values programmatically (handy for tests / fixtures).

| Env var | Dataset | Used for | Download |
|---|---|---|---|
| `SEPROBE_VCTK_DIR` | VCTK-Corpus (clean speech, test speakers) | Source of clean utterances. Loader downsamples 48 → 16 kHz on the fly. | https://datashare.ed.ac.uk/handle/10283/2950 |
| `SEPROBE_DEMAND_DIR` | DEMAND multi-channel noise database | Additive noise for SNR sweeps. | https://zenodo.org/records/1227121 |
| `SEPROBE_AIR_RIR_DIR` | Aachen Impulse Response (AIR) database | Room impulse responses for the reverb experiments (notebook 06). | https://www.iks.rwth-aachen.de/forschung/tools-downloads/databases/aachen-impulse-response-database/ |

## Expected layouts

```
$SEPROBE_VCTK_DIR/
    p<speaker_id>/
        p<speaker_id>_<utt_id>.wav   # 48 kHz mono
$SEPROBE_DEMAND_DIR/
    <NOISE_NAME>.wav                 # e.g. TBUS.wav, SPSQUARE.wav (16 kHz mono)
$SEPROBE_AIR_RIR_DIR/
    air_binaural_<room>_<config>.mat # MATLAB struct with field h_air (48 kHz)
```

The exact speaker / noise / room subsets used in the paper are listed in `se_probe/consts.py` (`TEST_SPEAKERS`, `TEST_NOISES`, `AIR_TEST_ROOMS`).

## Precomputed CKA tables

The full `results_df/` (~3.1 GB of parquets, every model × every SNR × every noise × every utterance × every layer) is hosted on HuggingFace. Pull it with:

```bash
python scripts/setup.py --full-data
```

This downloads from `HF_REPO_DATA` (default `YairAmar/SE-Probe-data`).

## Quick sanity check

If you set `SEPROBE_VCTK_DIR` and `SEPROBE_DEMAND_DIR` and want to verify they work without launching a notebook:

```python
from se_probe.io import load_clean_wavs
from se_probe.data_generation import load_demand_noise
print(len(load_clean_wavs()), "clean utterances")
print(load_demand_noise("TBUS").shape, "TBUS noise samples")
```
