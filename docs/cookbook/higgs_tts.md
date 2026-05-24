# Higgs Audio v3 TTS

[Higgs Audio v3](https://huggingface.co/boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999) is a
chat-native TTS model built on a Qwen3-4B backbone. It generates 24 kHz speech through 8 discrete
codebooks and supports 100+ languages, voice cloning from a reference clip, and fine-grained
expression control via inline tokens for emotion, style, sound effects, and prosody.

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

The Higgs Audio model weights are private. Export your Hugging Face token before downloading:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
huggingface-cli download boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999
huggingface-cli download bosonai/higgs-audio-v2-tokenizer   # public codec checkpoint
```

## Server Configuration

```bash
sgl-omni serve \
  --model-path boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999 \
  --config examples/configs/higgs_tts.yaml \
  --port 8000
```

The pipeline is `preprocessing → audio_encoder → tts_engine → vocoder`. The vocoder loads
[`bosonai/higgs-audio-v2-tokenizer`](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer)
by default. Override the codec path per-stage if you have a local copy:

```bash
sgl-omni serve \
  --model-path boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999 \
  --config examples/configs/higgs_tts.yaml \
  --stage-arg preprocessing.audio_codec_path=/path/to/codec \
  --stage-arg vocoder.audio_codec_path=/path/to/codec \
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

### Inline Control Tokens

The model supports control tokens embedded directly in the `input` text. Tokens take effect
from the point they appear and persist until overridden or the utterance ends.

#### Emotion

| Token | Effect |
|---|---|
| `<\|happy\|>` | Happy / upbeat |
| `<\|sad\|>` | Sad / subdued |
| `<\|angry\|>` | Angry |
| `<\|fearful\|>` | Fearful |
| `<\|disgusted\|>` | Disgusted |
| `<\|surprised\|>` | Surprised |
| `<\|neutral\|>` | Neutral (reset to baseline) |

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "I can't believe it! <|surprised|> That's absolutely amazing!"},
)
```

#### Speaker Style

| Token | Effect |
|---|---|
| `<\|male\|>` | Male speaker characteristics |
| `<\|female\|>` | Female speaker characteristics |
| `<\|whisper\|>` | Whispered delivery |
| `<\|strong_accent\|>` | Accented speech |

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "<|whisper|> Keep your voice down, they might hear us."},
)
```

#### Sound Effects

| Token | Effect |
|---|---|
| `<\|sfx\|>` | Generic sound effect marker |
| `<\|laughter\|>` | Laughter |

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "That was so funny. <|sfx|><|laughter|> I can't stop laughing!"},
)
```

#### Prosody

| Token | Effect |
|---|---|
| `<\|emphasis:word\|>` | Stress the given word (replace `word` with the target word) |
| `<\|pace:slow\|>` | Slow down speaking rate |
| `<\|pace:fast\|>` | Speed up speaking rate |
| `<\|pause:long\|>` | Insert a long pause |
| `<\|pause:short\|>` | Insert a short pause |

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": (
            "This is <|emphasis:very|> important. "
            "<|pause:short|> "
            "<|pace:slow|> Please listen carefully."
        )
    },
)
```

### Multi-speaker Dialogue

Pass multiple entries in `references` to condition the model on several distinct voices within
one request. Each entry needs `audio_path` and `text`; the model learns to associate each
voice with the corresponding transcript segment.

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
| `references` | list | `null` | Reference audio for voice cloning; each item: `audio_path` (local path or HTTP URL) and `text` (transcript) |
| `reference_codes` | list[list[int]] | `null` | Pre-encoded discrete codes, shape `[T, 8]` — replaces `references[0].audio_path` |
| `reference_text` | string | `null` | Transcript paired with `reference_codes` |
| `max_new_tokens` | int | `2048` | Maximum number of generated multi-codebook steps |
| `temperature` | float | `1.0` | Sampling temperature |
| `top_p` | float | `null` | Top-p (nucleus) sampling |
| `top_k` | int | `null` | Top-k sampling |
| `seed` | int | `null` | Random seed for reproducibility |

## Advanced: Pre-encoded Reference Codes

For high-throughput pipelines where the same reference audio is reused across many requests
(e.g. RL rollout), encode the reference offline and pass the discrete codes via `reference_codes`.
This skips the server-side audio encode step. The shape must be `[T, num_codebooks=8]`
(pre-delay-pattern layout).

```python
import numpy as np
import requests

# Encode reference audio offline (pseudo-code — use your codec to get codes_TN)
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

Evaluated on seed-tts-eval-mini en (1 000 utterances), single A100 40 GB, bf16,
`top_k=50, temperature=0.8, max_new_tokens=1024`. WER scored with Whisper-large-v3 (fp32);
speaker similarity with WavLM-large ECAPA-TDNN cosine × 100.

| Metric | Value |
|---|---|
| avg WER | 0.0182 |
| avg speaker similarity | 64.81 |

Throughput (N=50/level, sequential thread pool):

| Concurrency | Mean Latency | RTF (per-req) | audio\_s/s |
|---|---|---|---|
| 1 | 4 637 ms | 0.526 | 1.90 |
| 16 | 7 138 ms | 0.747 | 12.88 |
| 32 | 10 188 ms | 0.865 | 16.94 |

## Known Limitations

- **Transcript required for best cloning quality.** The `text` field in each `references` entry
  is technically optional but omitting it degrades speaker similarity, especially for short
  reference clips.
- **`reference_codes` must be pre-delay-pattern.** Pass codes in the same layout the codec
  produces before the delay-pattern shift; the server does not undo the pattern on your behalf.
- **Long inputs may truncate.** The model has a finite context window. For very long texts,
  split into sentence-level chunks and concatenate the resulting WAV files.
- **Control tokens are order-sensitive.** A token applies from its position forward; place it
  immediately before the word or phrase it should affect.
