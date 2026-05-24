# Higgs Audio v3 Generation

[Higgs Audio v3 Generation](https://huggingface.co/boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999)
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
```

The model weights are private. Export your Hugging Face token before downloading:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
huggingface-cli download boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999
huggingface-cli download bosonai/higgs-audio-v2-tokenizer   # public codec checkpoint
```

## Server Configuration

The pipeline is `preprocessing → audio_encoder → tts_engine → vocoder`. By default the
vocoder loads the audio codec from the model checkpoint directory itself.

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

Without a reference, the model generates speech in a default voice:

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

Higgs conditions on both the reference audio and its transcript. Supplying the transcript
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

### Multi-speaker Dialogue

Pass multiple entries in `references` to condition on several distinct voices within one
request. Each entry needs `audio_path` and `text`.

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": (
            "[Speaker A]: Good morning, how can I help you today?\n"
            "[Speaker B]: I'd like to check my account balance, please."
        ),
        "references": [
            {"audio_path": "speaker_a.wav", "text": "Good morning everyone."},
            {"audio_path": "speaker_b.wav", "text": "Thank you very much."},
        ],
        "temperature": 0.8,
        "top_k": 50,
    },
)
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

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | (required) | Text to synthesize; may include inline control tokens |
| `voice` | string | `"default"` | Voice identifier (ignored when `references` is set) |
| `response_format` | string | `"wav"` | Output audio format |
| `stream` | bool | `false` | Enable streaming via SSE |
| `references` | list | `null` | Reference audio for voice cloning; each item has `audio_path` (local path or HTTP URL) and `text` (transcript) |
| `reference_codes` | list[list[int]] | `null` | Pre-encoded discrete codes, shape `[T, 8]` — alternative to `references[0].audio_path` |
| `reference_text` | string | `null` | Transcript paired with `reference_codes` |
| `max_new_tokens` | int | `2048` | Maximum number of generated multi-codebook steps |
| `temperature` | float | `1.0` | Sampling temperature |
| `top_p` | float | `null` | Top-p (nucleus) sampling |
| `top_k` | int | `null` | Top-k sampling |
| `seed` | int | `null` | Random seed for reproducibility |

## Advanced: Pre-encoded Reference Codes

For high-throughput pipelines where the same reference audio is reused across many requests
(e.g. RL rollout), encode the reference offline and pass the discrete codes via `reference_codes`.
This skips the server-side encode step. Shape must be `[T, num_codebooks=8]` (pre-delay-pattern).

```python
import numpy as np
import requests

# codes_TN shape: [T, 8], dtype int, pre-delay-pattern
codes_TN: list[list[int]] = np.load("reference_codes.npy").tolist()

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "reference_codes": codes_TN,
        "reference_text": "We asked over twenty different people, and they all said it was his.",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

## Benchmark Results

WER/CER (↓) and WavLM speaker similarity (↑, ×100). Macro averages across models:

### Seed-TTS

| Lang | Higgs v3 WER ↓ | Higgs v3 SIM ↑ | Higgs v2 WER ↓ | Higgs v2 SIM ↑ | Fish S2 Pro WER ↓ | Fish S2 Pro SIM ↑ | Qwen3-TTS-1.7B WER ↓ | Qwen3-TTS-1.7B SIM ↑ | VibeVoice-7B WER ↓ | VibeVoice-7B SIM ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| en | 1.51 | 68.02 | 2.99 | 67.56 | 1.63 | 63.97 | 1.67 | 71.61 | 5.35 | 60.39 |
| zh | 1.10 | 72.91 | 1.82 | 73.21 | 1.23 | 72.40 | 1.48 | 77.04 | 2.50 | 70.71 |
| **macro** | **1.31** | **70.47** | 2.41 | 70.38 | 1.43 | 68.19 | 1.57 | 74.33 | 3.92 | 65.55 |

### CV3 (macro)

| Model | WER ↓ | SIM ↑ |
|---|---|---|
| Higgs Audio v3 | **4.67** | **69.77** |
| Higgs Audio v2 | 21.28 | 65.39 |
| Fish Audio S2 Pro | 4.63 | 67.28 |
| Qwen3-TTS-1.7B | 7.80 | 72.45 |
| VibeVoice-7B | 11.74 | 65.96 |

### MiniMax-Multilingual (macro, 23 languages)

| Model | WER ↓ | SIM ↑ |
|---|---|---|
| Higgs Audio v3 | **1.88** | **78.56** |
| Higgs Audio v2 | 43.72 | 70.34 |
| Fish Audio S2 Pro | 4.17 | 74.74 |
| Qwen3-TTS-1.7B | 27.85 | 77.34 |
| VibeVoice-7B | 7.20 | 73.95 |

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
  unexpected delivery; prefer one or two tokens per segment.
- **Degraded quality on unsupported languages or noisy prompts.** Performance outside the
  95+ single-digit WER/CER languages is usable but less polished.
- **`reference_codes` must be pre-delay-pattern.** Pass codes before the delay-pattern shift;
  the server does not undo the pattern.
- **8 192-token context limit.** Split long texts into sentence-level chunks and concatenate
  the resulting WAV files to stay within the model's sequence length.
