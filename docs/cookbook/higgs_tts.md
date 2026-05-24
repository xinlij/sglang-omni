# Higgs Audio v3 TTS

[Higgs Audio v3 TTS](https://huggingface.co/boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999)
is a chat-native text-to-speech model from Boson AI built on a Qwen3-4B backbone. It generates
24 kHz speech through 8 discrete codebooks and supports 100+ languages, voice cloning from a
reference clip, and fine-grained inline control over emotion, style, sound effects, and prosody.

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

To use a separate codec checkpoint:

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

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

### Voice Cloning

Supplying the reference transcript (`text`) materially improves cloning quality.

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

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "references": [{
            "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
            "text": "We asked over twenty different people, and they all said it was his.",
        }],
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

Embed control tokens directly in the `input` text.

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

#### Style

| Token | Description |
|---|---|
| `<\|style:singing\|>` | Singing |
| `<\|style:shouting\|>` | Shouting / projected voice |
| `<\|style:whispering\|>` | Whisper |

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
            "I can't believe it! <|emotion:surprise|> "
            "<|prosody:pause|> "
            "<|style:whispering|> That's absolutely incredible."
        )
    },
)
```

## Benchmark Results

Macro WER/CER (↓) and WavLM speaker similarity (↑, ×100):

| Benchmark | Higgs v3 WER ↓ | Higgs v3 SIM ↑ | Higgs v2 WER ↓ | Higgs v2 SIM ↑ | Fish S2 Pro WER ↓ | Fish S2 Pro SIM ↑ | Qwen3-TTS-1.7B WER ↓ | Qwen3-TTS-1.7B SIM ↑ | VibeVoice-7B WER ↓ | VibeVoice-7B SIM ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| Seed-TTS (en+zh) | **1.31** | 70.47 | 2.41 | 70.38 | 1.43 | 68.19 | 1.57 | 74.33 | 3.92 | 65.55 |
| CV3 (9 langs) | **4.67** | 69.77 | 21.28 | 65.39 | 4.63 | 67.28 | 7.80 | 72.45 | 11.74 | 65.96 |
| MiniMax-Multilingual (23 langs) | **1.88** | 78.56 | 43.72 | 70.34 | 4.17 | 74.74 | 27.85 | 77.34 | 7.20 | 73.95 |
| Higgs-Multilingual (100+ langs) | **5.20** | 75.49 | 55.62 | 63.03 | 13.33 | 71.88 | 97.80 | 73.13 | 20.81 | 71.85 |

## Known Limitations

- **Transcript improves cloning quality.** Omitting `text` in `references` degrades speaker similarity, especially for short clips.
- **Rare-word mispronunciation.** The model may mispronounce uncommon words or proper nouns.
- **Prosody drift on long generations.** Expressive control may weaken over long utterances.
- **Control token stacking.** Using many control tokens simultaneously can produce unexpected delivery.
- **Unsupported languages.** Performance outside the 95+ single-digit WER/CER languages is usable but less polished.
