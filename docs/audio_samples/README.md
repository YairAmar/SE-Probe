# Audio samples

13 short waveforms (≈1.1 MB total) used as demonstration material in talks and the project page. Each pair `<NOISE>_snr<X>.wav` / `<NOISE>_snr<X>_enhanced.wav` shows the noisy input and the MUSE-enhanced output for the same utterance and SNR.

| File | Content |
|---|---|
| `clean.wav` | Clean reference utterance. |
| `SPSQUARE_snr-5.wav` / `SPSQUARE_snr-5_enhanced.wav` | Public-square noise at SNR −5 dB, before / after enhancement. |
| `SPSQUARE_snr+5.wav` / `SPSQUARE_snr+5_enhanced.wav` | Public-square noise at SNR +5 dB. |
| `SPSQUARE_snr+20.wav` / `SPSQUARE_snr+20_enhanced.wav` | Public-square noise at SNR +20 dB. |
| `TBUS_snr-5.wav` / `TBUS_snr-5_enhanced.wav` | Bus noise at SNR −5 dB. |
| `TBUS_snr+5.wav` / `TBUS_snr+5_enhanced.wav` | Bus noise at SNR +5 dB. |
| `TBUS_snr+20.wav` / `TBUS_snr+20_enhanced.wav` | Bus noise at SNR +20 dB. |

All files are 16 kHz mono PCM. Noise types come from DEMAND (`SPSQUARE`, `TBUS`); enhancement is the upstream noise-only MUSE checkpoint (no reverb fine-tuning).
