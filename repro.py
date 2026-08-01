#!/usr/bin/env python3
"""A/B repro for the Stable Audio 3 VAE bf16 corruption.

Decodes the same fixed-seed latent through the SA3 audio VAE at several
(device, dtype) combos and reports crest factor + correlation vs the
CPU-fp32 reference decode.

Usage (from a ComfyUI checkout, with its venv):
    python repro.py --ckpt /path/to/stable_audio_3_small_sfx.safetensors \
                    [--comfy-root /path/to/ComfyUI] [--latent latent_s12345.pt]

With --latent (the pinned latent from this evidence branch, sampled at
seed 12345, steps 8, cfg 1.0, lcm/simple, 2.0 s, from the prompt in
workflow_api.json) no text encoder is needed. Without it, pass --te to
sample a fresh latent with t5gemma first.
"""
import argparse
import math
import os
import sys
import wave

p = argparse.ArgumentParser()
p.add_argument("--comfy-root", default=".")
p.add_argument("--ckpt", required=True)
p.add_argument("--latent", default=None)
p.add_argument("--te", default=None, help="t5gemma path; only needed without --latent")
p.add_argument("--out", default="sa3_vae_repro_out")
a = p.parse_args()

for attr in ("ckpt", "latent", "te"):
    if getattr(a, attr):
        setattr(a, attr, os.path.abspath(getattr(a, attr)))
sys.path.insert(0, os.path.abspath(a.comfy_root))
os.chdir(os.path.abspath(a.comfy_root))

import torch  # noqa: E402

import comfy.utils  # noqa: E402
import comfy.sd  # noqa: E402

out_dir = os.path.abspath(a.out)
os.makedirs(out_dir, exist_ok=True)

if a.latent:
    latent = torch.load(a.latent, weights_only=True)
else:
    import nodes  # noqa: E402
    model, _, _, _ = comfy.sd.load_checkpoint_guess_config(a.ckpt, output_vae=False, output_clip=False)
    clip = comfy.sd.load_clip(ckpt_paths=[a.te], clip_type=comfy.sd.CLIPType.STABLE_AUDIO)
    enc = nodes.CLIPTextEncode()
    prompt = ("Sci-fi blaster shot, sharp percussive mechanical transient layered under a bright "
              "descending energy zap, snappy attack, short pitched tail, dry and punchy, arcade "
              "space shooter weapon. Length: 1 seconds")
    positive = enc.encode(clip, prompt)[0]
    negative = enc.encode(clip, "")[0]
    length = round((2.0 * 44100 / 2048) / 2) * 2
    empty = {"samples": torch.zeros([1, 64, length]), "type": "audio",
             "downscale_ratio_temporal": 2048}
    latent = nodes.common_ksampler(model, 12345, 8, 1.0, "lcm", "simple",
                                   positive, negative, empty, denoise=1.0)[0]["samples"]
    latent = latent.to(torch.float32).cpu()
    torch.save(latent, os.path.join(out_dir, "latent_s12345.pt"))

sd = comfy.utils.load_torch_file(a.ckpt)
vae_sd = comfy.utils.state_dict_prefix_replace(sd, {"pretransform.model.": ""}, filter_keys=True)
del sd


def save_wav(path, wav_t, sr=44100):
    data = (wav_t.clamp(-1, 1) * 32767.0).to(torch.int16).transpose(0, 1).contiguous().numpy()
    with wave.open(path, "wb") as f:
        f.setnchannels(wav_t.shape[0])
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(data.tobytes())


def decode_with(device, dtype, tag):
    vae = comfy.sd.VAE(sd={k: v.clone() for k, v in vae_sd.items()},
                       device=torch.device(device), dtype=dtype)
    print(f"[{tag}] vae_dtype={vae.vae_dtype}", flush=True)
    torch.manual_seed(0)  # bottleneck.decode has a tiny randn regularizer; pin it
    audio = vae.decode(latent.clone()).movedim(-1, 1)
    # same normalization as vae_decode_audio() in comfy_extras/nodes_audio.py
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    return (audio / std)[0].to(torch.float32).cpu()


def crest(wav):
    mono = wav.mean(dim=0)
    peak = mono.abs().max().item()
    rms = mono.pow(2).mean().sqrt().item()
    return peak / rms, rms


def norm_corr(x, y):
    x, y = x.mean(dim=0), y.mean(dim=0)
    n = min(x.shape[-1], y.shape[-1])
    x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
    d = (x.norm() * y.norm()).item()
    return (x @ y).item() / d if d else 0.0


accel = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else None)
combos = [("cpu", torch.float32, "cpu-fp32"), ("cpu", torch.bfloat16, "cpu-bf16")]
if accel:
    combos += [(accel, None, f"{accel}-default"), (accel, torch.float32, f"{accel}-fp32"),
               (accel, torch.bfloat16, f"{accel}-bf16"), (accel, torch.float16, f"{accel}-fp16")]

results = {}
for device, dtype, tag in combos:
    wav = decode_with(device, dtype, tag)
    results[tag] = wav
    save_wav(os.path.join(out_dir, f"{tag}.wav"), wav)

ref = results["cpu-fp32"]
print(f"\n{'tag':13s} {'crest':>7s} {'crest_dB':>9s} {'rms':>9s} {'corr_vs_cpu_fp32':>17s}")
for _, _, tag in combos:
    c, rms = crest(results[tag])
    r = norm_corr(results[tag], ref)
    print(f"{tag:13s} {c:7.2f} {20 * math.log10(c):9.2f} {rms:9.5f} {r:17.4f}")
print("\ncorr ~1.0 = decodes the same sound as the reference; corr ~0 = noise")
