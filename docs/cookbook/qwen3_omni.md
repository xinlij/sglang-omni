# Qwen3-Omni

[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) is a multi-modal model
that accepts text, image, audio, and video input and can produce text-only or text + audio output.
This page covers every supported server configuration — use the generator to get the exact launch
command for your hardware, then check the tables to confirm your combination is supported.

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

## Server Configuration

Use the selector below to generate the exact launch command for your configuration.

```{raw} html
<div id="sgl-server-gen-mount"></div>
```

## Compatibility Matrix

- ✅ = CI-tested on every PR
- ⚠️ = code path exists, not yet in CI
- ❌ = explicitly rejected by the CLI/runtime

Colocated topology requires `--config examples/configs/qwen3_omni_colocated_h20.yaml`
(or `qwen3_omni_colocated_h200.yaml` on H200) to set per-stage GPU memory budgets.

| Mode | Topology | Thinker TP | Precision | Status | Notes |
|---|---|---|---|---|---|
| Thinker-only | — | — | BF16 | ✅ | Tested in CI (MMMU, MMSU, Video-MME, Video-AMME) |
| Thinker-only | — | — | FP8 | ⚠️ | Code path exists; not in CI |
| Thinker-Talker | Disaggregated | TP=1 | BF16 | ✅ | Tested in CI (TTS, MMMU/MMSU/Video talker stages) |
| Thinker-Talker | Disaggregated | TP=1 | FP8 | ⚠️ | Code path exists; not in CI |
| Thinker-Talker | Disaggregated | TP=2 | BF16 | ⚠️ | Placement unit-tested; no end-to-end CI |
| Thinker-Talker | Disaggregated | TP=2 | FP8 | ⚠️ | Code path exists; not in CI |
| Thinker-Talker | Colocated | TP=1 | BF16 | ✅ | Tested in CI (colocated router stage) |
| Thinker-Talker | Colocated | TP=1 | FP8 | ⚠️ | Code path exists; not in CI |
| Thinker-Talker | Colocated | TP=2 | — | ❌ | Rejected by runtime — use Disaggregated for TP=2 |

## Input / Output Modalities

All input modality combinations work with both text-only and speech servers.
`modalities: ["text", "audio"]` requires a **speech-mode server** (omit `--text-only`).

| Input | Output | Speech server | Minimal request body | Notes |
|---|---|---|---|---|
| Text | Text | No | `"messages": [{"role": "user", "content": "..."}], "modalities": ["text"]` | — |
| Image + text | Text | No | `"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text"]` | — |
| Audio | Text | No | `"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text"]` | content must be "" when the query is spoken |
| Image + audio | Text | No | `"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]` | content must be "" when the query is spoken |
| Image | Text | No | `"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text"]` | content must be "" when query comes from image |
| Video + text | Text | No | `"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text"]` | — |
| Video + audio | Text | No | `"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]` | content must be "" when the query is spoken |
| Video | Text | No | `"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text"]` | content must be "" when query comes from video |
| Text | Text + Audio | **Yes** | `"messages": [{"role": "user", "content": "..."}], "modalities": ["text", "audio"]` | — |
| Image + text | Text + Audio | **Yes** | `"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text", "audio"]` | — |
| Audio | Text + Audio | **Yes** | `"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text", "audio"]` | content must be "" when the query is spoken |
| Image + audio | Text + Audio | **Yes** | `"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]` | content must be "" when the query is spoken |
| Image | Text + Audio | **Yes** | `"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text", "audio"]` | content must be "" when query comes from image |
| Video + text | Text + Audio | **Yes** | `"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text", "audio"]` | — |
| Video + audio | Text + Audio | **Yes** | `"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]` | content must be "" when the query is spoken |
| Video | Text + Audio | **Yes** | `"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text", "audio"]` | content must be "" when query comes from video |

### Sampling Parameters

Standard sampling parameters apply to the thinker stage. When `modalities` includes `"audio"`, the additional talker-specific parameters below control the speech generation independently.

| Parameter | Type | Default | Applies to |
|---|---|---|---|
| `temperature` | float | `null` | Thinker |
| `top_p` | float | `null` | Thinker |
| `top_k` | int | `null` | Thinker |
| `min_p` | float | `null` | Thinker |
| `repetition_penalty` | float | `null` | Thinker |
| `max_tokens` | int | `null` | Thinker |
| `stop` | str \| list | `null` | Thinker |
| `seed` | int | `null` | Thinker |
| `stream` | bool | `false` | Both |
| `audio` | dict | `null` | Talker (speech output only) — format config, e.g. `{"voice": "default", "format": "wav"}` |
| `talker_temperature` | float | `null` | Talker (audio output only) |
| `talker_top_p` | float | `null` | Talker (audio output only) |
| `talker_top_k` | int | `null` | Talker (audio output only) |
| `talker_repetition_penalty` | float | `null` | Talker (audio output only) |
| `talker_max_new_tokens` | int | `null` | Talker (audio output only) |
| `stage_sampling` | dict | `null` | Per-stage sampling override, e.g. `{"thinker": {"temperature": 0.8}}` |
| `stage_params` | dict | `null` | Per-stage non-sampling params, e.g. `{"preprocessor": {"video_fps": 1.0}}` |
| `video_fps` | float | `null` | Frame sampling rate for video input (uses server default if unset) |
| `video_max_frames` | int | `null` | Maximum number of frames sampled from a video |
| `video_min_pixels` | int | `null` | Minimum pixels per video frame |
| `video_max_pixels` | int | `null` | Maximum pixels per video frame |
| `video_total_pixels` | int | `null` | Total pixel budget across all video frames |

### Known Limitations

- **`modalities: ["text", "audio"]` has no effect on a text-only server.** No error is raised — the response simply contains no audio. Use a speech-mode server (without `--text-only`) to get audio output.
- **`content` must be `""` when the query is entirely in `audios`, `videos`, or `images`.** Leaving a text query in `content` alongside audio causes the model to process both, which is usually not what you want.
- **Colocated topology does not support `--thinker-tp-size 2`.** The server raises a `ValueError` at startup ("Qwen Phase 1 colocation does not support thinker TP"). Use disaggregated topology for TP=2.
- **Requests that exceed the model's context length are rejected with an error.** The preprocessor raises a `ValueError` when the prompt token count alone meets or exceeds `max_seq_len`, or when `prompt tokens + max_new_tokens ≥ max_seq_len`. Reduce input length or lower `max_tokens` to stay within the limit.
