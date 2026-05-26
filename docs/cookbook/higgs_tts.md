# Higgs Audio v3 TTS

[Higgs Audio v3 TTS](https://huggingface.co/boson-sglang/higgs-audio-v3-tts-4b-base)
is a chat-native text-to-speech model from Boson AI built on a Qwen3-4B backbone. It generates
24 kHz speech through 8 discrete codebooks and supports 100+ languages, voice cloning from a
reference clip, and fine-grained inline control over emotion, style, sound effects, and prosody.

## Highlights

- **Chat-native, low-latency** streaming multi-turn speech generation
- **Multilingual** — 100+ languages and dialects, 90+ with single-digit WER/CER
- **Voice clone accuracy** — high-fidelity zero-shot speaker cloning from reference clips
- **Inline control** via `<|emotion:…|>`, `<|style:…|>`, `<|sfx:…|>`, `<|prosody:…|>` tags

## Architecture

![Higgs Audio v3 Generation Architecture](./assets/higgs-architecture.png)

Higgs autoregressive decoder consumes interleaved text and audio tokens. Audio is encoded by the **Higgs Tokenizer** into 8 codebooks at 25 fps, staggered via a **delay pattern**, then mapped to backbone hidden states through a **multi-codebook fused embedding**. Output codes pass through a **multi-codebook fused head**, are de-delayed, and decoded back to waveform. Multi-turn generation interleaves `<|text|>…<|audio|>…` chunks so each new chunk is grounded on reference + prior chunks.

| Component | Spec |
|---|---|
| Backbone | ~4B autoregressive decoder (36 L, hidden=2560, GQA 32/8) |
| Audio tokens | 8 codebooks × 1026 vocab, delay pattern |
| Multi-codebook embedding / head | Fused single-tensor, tied with text embedding |
| Context length | 8,192 tokens (training sequence length) |

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
hf download boson-sglang/higgs-audio-v3-tts-4b-base
hf download bosonai/higgs-audio-v2-tokenizer
```

## Server Configuration

The pipeline is `preprocessing → audio_encoder → tts_engine → vocoder`.

```bash
sgl-omni serve \
  --model-path boson-sglang/higgs-audio-v3-tts-4b-base \
  --config examples/configs/higgs_tts.yaml \
  --port 8000
```

The audio codec defaults to the public [`bosonai/higgs-audio-v2-tokenizer`](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) repo. Override per-stage if you need a different codec checkpoint:

```bash
sgl-omni serve \
  --model-path boson-sglang/higgs-audio-v3-tts-4b-base \
  --config examples/configs/higgs_tts.yaml \
  --stage-arg preprocessing.audio_codec_path=<path-or-repo-id> \
  --stage-arg vocoder.audio_codec_path=<path-or-repo-id> \
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
<audio controls>
  <source src="../_static/audio/higgs-1.wav" type="audio/wav">
</audio>

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
# Python
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

Embed control tokens directly in the `input` field. Tokens from different
categories can be combined:

**Demo**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "I cant believe it! <|emotion:surprise|> <|prosody:pause|> <|style:whispering|> Thats absolutely incredible."
  }' \
  --output output.wav
```
<audio controls>
  <source src="../_static/audio/control-tokens-test1.wav" type="audio/wav">
</audio>

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:enthusiasm|> Welcome to the show! <|prosody:pause|> <|prosody:speed_slow|> Today we have something truly special for you."
  }' \
  --output output.wav
```
<audio controls>
  <source src="../_static/audio/control-tokens-test2.wav" type="audio/wav">
</audio>

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

### Request parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | (required) | Text to synthesize |
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


### Throughput

Throughput on seed-tts en (N=50 per concurrency, sequential thread pool, A100 40GB, bf16):

| Concurrency | Mean latency | RTF (per-req) | audio_s/s |
|---:|---:|---:|---:|
| 1 | 4637 ms | 0.526 | 1.90 |
| 16 | 7138 ms | 0.747 | 12.88 |
| 32 | 10188 ms | 0.865 | 16.94 |

## Evaluation Benchmarks

We report **WER / CER** (↓, %) and **WavLM speaker similarity** (↑, ×100) on three zero-shot voice-cloning benchmarks.

### Seed-TTS

<table>
<thead>
<tr><th>Lang</th><th>WER ↓</th><th>SIM ↑</th></tr>
</thead>
<tbody>
<tr><td>en</td><td>2.05</td><td>64.86</td></tr>
<tr><td>zh</td><td>2.00</td><td>70.96</td></tr>
<tr><td><b>macro</b></td><td><b>2.02</b></td><td><b>67.91</b></td></tr>
</tbody>
</table>

### CV3 (9 langs)

<table>
<thead>
<tr><th>Lang</th><th>WER ↓</th><th>SIM ↑</th></tr>
</thead>
<tbody>
<tr><td>de</td><td>8.62</td><td>65.43</td></tr>
<tr><td>en</td><td>6.73</td><td>60.37</td></tr>
<tr><td>es</td><td>5.03</td><td>68.18</td></tr>
<tr><td>fr</td><td>14.50</td><td>62.34</td></tr>
<tr><td>it</td><td>8.55</td><td>67.34</td></tr>
<tr><td>ja</td><td>7.96</td><td>67.91</td></tr>
<tr><td>ko</td><td>4.38</td><td>68.40</td></tr>
<tr><td>ru</td><td>9.38</td><td>66.77</td></tr>
<tr><td>zh</td><td>5.19</td><td>69.71</td></tr>
<tr><td><b>macro</b></td><td><b>7.82</b></td><td><b>66.27</b></td></tr>
</tbody>
</table>

### MiniMax-Multilingual (23 langs)

<table>
<thead>
<tr><th>Lang</th><th>WER ↓</th><th>SIM ↑</th></tr>
</thead>
<tbody>
<tr><td>ar</td><td>2.59</td><td>74.77</td></tr>
<tr><td>cs</td><td>4.62</td><td>78.80</td></tr>
<tr><td>de</td><td>0.74</td><td>70.65</td></tr>
<tr><td>el</td><td>1.81</td><td>78.02</td></tr>
<tr><td>en</td><td>1.87</td><td>81.32</td></tr>
<tr><td>es</td><td>3.06</td><td>72.78</td></tr>
<tr><td>fi</td><td>4.62</td><td>82.69</td></tr>
<tr><td>fr</td><td>4.70</td><td>70.27</td></tr>
<tr><td>hi</td><td>6.81</td><td>80.94</td></tr>
<tr><td>id</td><td>2.38</td><td>72.42</td></tr>
<tr><td>it</td><td>2.07</td><td>74.56</td></tr>
<tr><td>ja</td><td>3.74</td><td>74.23</td></tr>
<tr><td>ko</td><td>3.57</td><td>74.86</td></tr>
<tr><td>nl</td><td>2.10</td><td>73.02</td></tr>
<tr><td>pl</td><td>2.08</td><td>83.16</td></tr>
<tr><td>pt</td><td>2.59</td><td>76.52</td></tr>
<tr><td>ro</td><td>3.64</td><td>77.10</td></tr>
<tr><td>ru</td><td>4.66</td><td>74.48</td></tr>
<tr><td>th</td><td>7.59</td><td>77.64</td></tr>
<tr><td>tr</td><td>2.09</td><td>77.72</td></tr>
<tr><td>uk</td><td>2.69</td><td>71.79</td></tr>
<tr><td>vi</td><td>1.18</td><td>73.46</td></tr>
<tr><td>zh</td><td>1.65</td><td>74.85</td></tr>
<tr><td><b>macro</b></td><td><b>3.17</b></td><td><b>75.92</b></td></tr>
</tbody>
</table>
