# Stable Audio 3 VAE bf16 corruption — evidence

Supporting files for the fix "don't default the Stable Audio 3 VAE to bf16 on MPS".

On Apple Silicon the SA3 audio VAE picks bf16 by default (`working_dtypes`
lists it first and `should_use_bf16()` approves it on macOS >= 14), and the
decode silently produces broadband noise. fp16 and fp32 decode correctly.
The bf16 failure also reproduces on CPU, so it is a model/format
incompatibility surfaced wherever sdpa's softmax runs in the input dtype —
not an MPS kernel bug.

## Files

| file | what |
|---|---|
| `before_default_bf16.wav` | the old MPS default (bf16) — broadband noise |
| `after_default_fp16.wav` | the new MPS default (fp16) — correct one-shot |
| `reference_cpu_fp32.wav` | ground-truth CPU fp32 decode of the same latent |
| `*_spectrogram.png` | spectrograms of the three above |
| `*.mp4` | spectrogram + audio as video, for drag-and-drop into GitHub comments (inline playback) |
| `latent_s12345.pt` | the sampled latent all three decodes share |
| `repro.py` | standalone A/B script (see header for usage) |
| `workflow_api.json` | the generating graph (API format) |

## Generation settings

`stable_audio_3_small_sfx.safetensors` + `t5gemma_b_b_ul2.safetensors`,
seed 12345, steps 8, cfg 1.0, sampler lcm, scheduler simple, 2.0 s.
Measured on an Apple Silicon Mac, macOS 26.5.2, torch 2.10.0.

## Numbers (mono mixdown, same latent for every row)

| decode | crest | crest dB | rms | corr vs cpu-fp32 |
|---|---|---|---|---|
| cpu-fp32 (reference) | 15.49 | 23.80 | 0.060 | 1.0000 |
| cpu-bf16 | 13.13 | 22.37 | 0.197 | -0.0044 |
| mps-default before fix (= bf16) | 13.07 | 22.33 | 0.197 | -0.0099 |
| mps-fp32 | 15.50 | 23.80 | 0.060 | 1.0000 |
| mps-bf16 | 13.07 | 22.33 | 0.197 | -0.0099 |
| mps-fp16 (= default after fix) | 15.69 | 23.91 | 0.060 | 0.9997 |

corr ~1.0 = same sound as the reference; corr ~0 = unrelated noise.
