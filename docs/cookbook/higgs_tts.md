# Higgs Audio v3 TTS

[Higgs Audio v3 TTS](https://huggingface.co/boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999)
is a chat-native text-to-speech model from Boson AI built on a Qwen3-4B backbone. It generates
24 kHz speech through 8 discrete codebooks and supports 100+ languages, voice cloning from a
reference clip, and fine-grained inline control over emotion, style, sound effects, and prosody.
This page covers the GRPO-tuned checkpoint, which is reinforcement-tuned for tighter zero-shot
intelligibility while preserving speaker similarity.

## Prerequisites

```bash
docker pull frankleeeee/sglang-omni:dev
docker run -it --shm-size 32g --gpus all frankleeeee/sglang-omni:dev /bin/zsh
```

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v .

# Higgs TTS model is private; export your HF token before downloading.
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
hf download boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999
hf download bosonai/higgs-audio-v2-tokenizer
```

## Server Configuration

The pipeline is `preprocessing → audio_encoder → tts_engine → vocoder`. By default the
pipeline loads the audio codec from the model checkpoint directory itself.

```bash
sgl-omni serve \
  --model-path boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999 \
  --config examples/configs/higgs_tts.yaml \
  --port 8000
```

To point at a separate codec (e.g. the public
[`bosonai/higgs-audio-v2-tokenizer`](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer)),
pass it through stage args:

```bash
sgl-omni serve \
  --model-path boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999 \
  --config examples/configs/higgs_tts.yaml \
  --stage-arg preprocessing.audio_codec_path=bosonai/higgs-audio-v2-tokenizer \
  --stage-arg vocoder.audio_codec_path=bosonai/higgs-audio-v2-tokenizer \
  --port 8000
```

## Synthesizing Speech

### Zero-shot

Without a reference, the model falls back to the zero-shot prompt:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "Hello, how are you?"},
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Voice Cloning

Higgs TTS conditions on both the reference audio and its transcript. Supplying the transcript
(`text`) materially improves cloning quality over audio-only conditioning.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

```python
import requests

REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
        "temperature": 0.8,
        "top_k": 50,
        "max_new_tokens": 1024,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Streaming

Set `"stream": true` to receive base64-encoded WAV chunks over Server-Sent Events:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "stream": true
  }'
```

### Inline Control Tokens

Control tokens are embedded directly in the `input` text and take effect from the point they
appear.

#### Emotion

| Token | Description |
|---|---|
| `<\|emotion:elation\|>` | Elation / joy |
| `<\|emotion:amusement\|>` | Amusement / playful laughter |
| `<\|emotion:enthusiasm\|>` | Enthusiasm / excitement |
| `<\|emotion:determination\|>` | Determination / firmness |
| `<\|emotion:pride\|>` | Pride / confidence |
| `<\|emotion:contentment\|>` | Calm satisfaction |
| `<\|emotion:affection\|>` | Warmth / affection |
| `<\|emotion:relief\|>` | Relief |
| `<\|emotion:contemplation\|>` | Thoughtful / reflective |
| `<\|emotion:confusion\|>` | Confused |
| `<\|emotion:surprise\|>` | Surprised |
| `<\|emotion:awe\|>` | Awe / wonder |
| `<\|emotion:longing\|>` | Longing / yearning |
| `<\|emotion:arousal\|>` | Heightened desire |
| `<\|emotion:anger\|>` | Anger |
| `<\|emotion:fear\|>` | Fear |
| `<\|emotion:disgust\|>` | Disgust |
| `<\|emotion:bitterness\|>` | Bitterness |
| `<\|emotion:sadness\|>` | Sadness |
| `<\|emotion:shame\|>` | Shame |
| `<\|emotion:helplessness\|>` | Helplessness |

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "I can't believe it! <|emotion:surprise|> That's absolutely amazing!"},
)
```

#### Style

| Token | Description |
|---|---|
| `<\|style:singing\|>` | Singing |
| `<\|style:shouting\|>` | Shouting / projected voice |
| `<\|style:whispering\|>` | Whisper |

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "<|style:whispering|> Keep your voice down, they might hear us."},
)
```

#### Sound Effects

| Token | Description |
|---|---|
| `<\|sfx:cough\|>` | Cough |
| `<\|sfx:laughter\|>` | Laughter |
| `<\|sfx:crying\|>` | Crying |
| `<\|sfx:screaming\|>` | Screaming |
| `<\|sfx:burping\|>` | Burping |
| `<\|sfx:humming\|>` | Humming |
| `<\|sfx:sigh\|>` | Sigh |
| `<\|sfx:sniff\|>` | Sniff |
| `<\|sfx:sneeze\|>` | Sneeze |

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "That was hilarious! <|sfx:laughter|> I can't stop laughing."},
)
```

#### Prosody

| Token | Effect |
|---|---|
| `<\|prosody:speed_very_slow\|>` | ~0.65× speed |
| `<\|prosody:speed_slow\|>` | ~0.85× speed |
| `<\|prosody:speed_fast\|>` | ~1.2× speed |
| `<\|prosody:speed_very_fast\|>` | ~1.4× speed |
| `<\|prosody:pitch_low\|>` | ~−3 semitones |
| `<\|prosody:pitch_high\|>` | ~+2.5 semitones |
| `<\|prosody:pause\|>` | ~400–700 ms pause |
| `<\|prosody:long_pause\|>` | ~700–1500 ms pause |
| `<\|prosody:expressive_high\|>` | More expressive delivery |
| `<\|prosody:expressive_low\|>` | Flatter delivery |

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": (
            "Please listen carefully. "
            "<|prosody:pause|> "
            "<|prosody:speed_slow|> This part is very important."
        )
    },
)
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | (required) | Text to synthesize; may include inline control tokens |
| `voice` | string | `"default"` | Voice identifier (ignored when `references` is set) |
| `response_format` | string | `"wav"` | Output audio format |
| `stream` | bool | `false` | Enable streaming via SSE |
| `references` | list | `null` | Reference audio for voice cloning; each item has `audio_path` (local path or HTTP URL) and `text` (transcript) |
| `reference_codes` | list[list[int]] | `null` | Pre-encoded discrete codes, shape `[T, 8]` — alternative to `references[0].audio_path` |
| `reference_text` | string | `null` | Transcript of reference audio when supplying `reference_codes` |
| `max_new_tokens` | int | `2048` | Maximum number of generated multi-codebook steps |
| `temperature` | float | `1.0` | Sampling temperature |
| `top_p` | float | `null` | Top-p sampling |
| `top_k` | int | `null` | Top-k sampling |
| `seed` | int | `null` | Random seed for reproducibility |

## Advanced: Pre-encoded Reference Codes

For high-throughput pipelines where the same reference audio is reused across many requests
(e.g. RL rollout), encode the reference offline and pass the discrete codes via `reference_codes`.
This skips the server-side codec encode step. Shape must be `[T, num_codebooks=8]`
(pre-delay-pattern).

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "reference_codes": codes_TN,   # [T, 8] int list, pre-delay-pattern
        "reference_text": "We asked over twenty different people, and they all said it was his.",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

## Benchmark Results

WER/CER (↓) and WavLM speaker similarity (↑, ×100).

### Seed-TTS

| Lang | Higgs v3 WER ↓ | Higgs v3 SIM ↑ | Higgs v2 WER ↓ | Higgs v2 SIM ↑ | Fish S2 Pro WER ↓ | Fish S2 Pro SIM ↑ | Qwen3-TTS-1.7B WER ↓ | Qwen3-TTS-1.7B SIM ↑ | VibeVoice-7B WER ↓ | VibeVoice-7B SIM ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| en | **1.51** | 68.02 | 2.99 | 67.56 | 1.63 | 63.97 | 1.67 | 71.61 | 5.35 | 60.39 |
| zh | **1.10** | 72.91 | 1.82 | 73.21 | 1.23 | 72.40 | 1.48 | 77.04 | 2.50 | 70.71 |
| macro | **1.31** | 70.47 | 2.41 | 70.38 | 1.43 | 68.19 | 1.57 | 74.33 | 3.92 | 65.55 |

### CV3

| Lang | Higgs v3 WER ↓ | Higgs v3 SIM ↑ | Higgs v2 WER ↓ | Higgs v2 SIM ↑ | Fish S2 Pro WER ↓ | Fish S2 Pro SIM ↑ | Qwen3-TTS-1.7B WER ↓ | Qwen3-TTS-1.7B SIM ↑ | VibeVoice-7B WER ↓ | VibeVoice-7B SIM ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 4.01 | 69.57 | 7.22 | 67.01 | **3.65** | 66.83 | 3.33 | 72.04 | 11.47 | 63.74 |
| en | 3.76 | 63.49 | 6.10 | 61.74 | 3.92 | 58.60 | 14.96 | 66.39 | 8.23 | 60.60 |
| es | **2.86** | 71.77 | 5.96 | 70.28 | 2.88 | 69.64 | 3.47 | 74.48 | 6.61 | 67.05 |
| fr | 9.77 | 66.47 | 18.49 | 62.10 | **8.63** | 64.85 | 10.33 | 70.17 | 17.91 | 63.11 |
| it | **3.60** | 70.99 | 13.06 | 62.98 | 4.39 | 67.81 | 3.98 | 72.52 | 9.65 | 63.81 |
| ja | **4.92** | 70.70 | 67.25 | 61.28 | 5.11 | 68.12 | 14.29 | 72.47 | 28.55 | 66.37 |
| ko | **3.86** | 71.98 | 10.83 | 71.14 | 3.86 | 69.95 | 4.90 | 75.00 | 7.58 | 69.19 |
| ru | **5.05** | 70.87 | 52.43 | 60.57 | 5.42 | 68.46 | 11.21 | 73.42 | 9.79 | 68.37 |
| zh | 3.76 | 72.13 | 10.21 | 71.38 | **3.82** | 71.28 | 3.71 | 75.57 | 5.83 | 71.44 |
| macro | **4.67** | 69.77 | 21.28 | 65.39 | 4.63 | 67.28 | 7.80 | 72.45 | 11.74 | 65.96 |

### MiniMax-Multilingual

| Lang | Higgs v3 WER ↓ | Higgs v3 SIM ↑ | Higgs v2 WER ↓ | Higgs v2 SIM ↑ | Fish S2 Pro WER ↓ | Fish S2 Pro SIM ↑ | Qwen3-TTS-1.7B WER ↓ | Qwen3-TTS-1.7B SIM ↑ | VibeVoice-7B WER ↓ | VibeVoice-7B SIM ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| ar | **0.76** | 76.64 | 53.36 | 64.65 | 1.14 | 73.98 | 65.06 | 76.16 | 3.76 | 69.43 |
| cs | **2.59** | 80.09 | 47.12 | 73.00 | 7.18 | 76.55 | 33.86 | 75.85 | 10.95 | 75.91 |
| de | 1.31 | 75.82 | **1.18** | 69.57 | 0.54 | 70.84 | 0.62 | 76.11 | 2.52 | 68.84 |
| el | **0.81** | 79.09 | 21.49 | 76.77 | 3.53 | 79.29 | 21.04 | 81.60 | 2.17 | 79.62 |
| en | **1.61** | 81.83 | 3.45 | 81.05 | 1.24 | 78.35 | 1.38 | 79.68 | 3.23 | 78.78 |
| es | **1.01** | 75.26 | 2.19 | 76.21 | 1.39 | 73.13 | 1.07 | 81.47 | 4.65 | 75.10 |
| fi | **2.73** | 83.94 | 37.07 | 71.00 | 7.88 | 77.94 | 36.94 | 79.64 | 13.95 | 82.33 |
| fr | **3.65** | 72.81 | 7.24 | 68.57 | 4.12 | 67.94 | 3.15 | 72.08 | 7.38 | 53.32 |
| hi | **5.19** | 84.72 | 8.97 | 80.81 | 7.58 | 78.80 | 6.38 | 81.36 | 9.40 | 82.62 |
| id | **1.53** | 75.64 | 14.36 | 70.79 | 3.14 | 71.96 | 2.28 | 78.98 | 4.65 | 73.60 |
| it | **1.49** | 78.44 | 2.89 | 72.69 | 1.47 | 76.32 | 1.29 | 76.04 | 2.75 | 70.59 |
| ja | **3.04** | 76.52 | 86.78 | 72.56 | 3.05 | 74.84 | 20.09 | 78.97 | 12.17 | 73.17 |
| ko | **2.41** | 78.50 | 13.86 | 78.84 | 2.25 | 74.88 | 6.62 | 79.74 | 11.40 | 74.35 |
| nl | **1.21** | 75.59 | 9.42 | 69.69 | 1.92 | 70.72 | 3.19 | 69.42 | 4.08 | 67.38 |
| pl | **0.76** | 85.03 | 8.52 | 78.21 | 2.09 | 79.07 | 43.55 | 79.50 | 8.56 | 81.02 |
| pt | **1.39** | 78.69 | 5.59 | 72.20 | 1.68 | 77.76 | 1.50 | 77.08 | 6.64 | 79.32 |
| ro | **1.96** | 80.40 | 12.63 | 72.21 | 15.28 | 72.60 | 21.42 | 78.77 | 7.60 | 72.99 |
| ru | **3.60** | 77.43 | 26.02 | 70.33 | 4.76 | 77.18 | 4.56 | 78.60 | 5.47 | 71.20 |
| th | **3.34** | 79.43 | 215.98 | 41.36 | 7.39 | 74.84 | 311.62 | 64.85 | 6.09 | 77.74 |
| tr | **0.50** | 80.80 | 73.43 | 68.65 | 1.76 | 76.53 | 24.00 | 80.37 | 6.08 | 71.68 |
| uk | **0.80** | 75.51 | 202.46 | 57.99 | 8.00 | 68.60 | 7.36 | 74.47 | 12.86 | 69.53 |
| vi | **0.51** | 76.69 | 149.69 | 54.16 | 7.53 | 69.90 | 22.60 | 77.68 | 17.64 | 72.76 |
| zh | **0.93** | 77.93 | 1.78 | 76.50 | 0.94 | 77.05 | 1.04 | 80.42 | 1.50 | 79.55 |
| macro | **1.88** | 78.56 | 43.72 | 70.34 | 4.17 | 74.74 | 27.85 | 77.34 | 7.20 | 73.95 |

### Higgs-Multilingual (macro, 100+ languages)

| Model | WER ↓ | SIM ↑ |
|---|---|---|
| Higgs Audio v3 | **5.20** | **75.49** |
| Higgs Audio v2 | 55.62 | 63.03 |
| Fish Audio S2 Pro | 13.33 | 71.88 |
| Qwen3-TTS-1.7B | 97.80 | 73.13 |
| VibeVoice-7B | 20.81 | 71.85 |

### Throughput

Measured on seed-tts en (N=50 per concurrency level), sequential thread pool, A100 40 GB, bf16:

| Concurrency | Mean Latency | RTF (per-req) | audio\_s/s |
|---|---|---|---|
| 1 | 4 637 ms | 0.526 | 1.90 |
| 16 | 7 138 ms | 0.747 | 12.88 |
| 32 | 10 188 ms | 0.865 | 16.94 |

## Known Limitations

- **Transcript required for best cloning quality.** Omitting `text` in `references` degrades
  speaker similarity, especially for short clips.
- **Rare-word mispronunciation.** The model may mispronounce uncommon words or proper nouns.
- **Prosody drift on long generations.** Expressive control may weaken over long utterances.
- **Control token stacking instability.** Using many control tokens simultaneously can produce
  unexpected delivery.
- **Degraded quality on unsupported languages or noisy prompts.** Performance outside the
  95+ single-digit WER/CER languages is usable but less polished.
- **`reference_codes` must be pre-delay-pattern.** Shape must be `[T, num_codebooks=8]`;
  the server does not undo the delay pattern.
- **8 192-token context limit.** Split long texts into sentence-level chunks and concatenate
  the resulting WAV files to stay within the model's sequence length.
