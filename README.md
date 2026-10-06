# Talk Like You

Official implementation of **Talk Like You: Imitating How You Speak in Real-Time Talking Head Generation**.

[Baiqin Wang](https://scholar.google.cz/citations?user=pBF9Mn8AAAAJ&hl=zh-CN&oi=ao), Zhixing Ding, Jijie Li, Jiankuo Zhao, [Zhen Lei](https://scholar.google.cz/citations?hl=zh-CN&user=cuJ3QG8AAAAJ), [Xiangyu Zhu](https://xiangyuzhu-open.github.io/homepage/)

MAIS, Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences; CAIR, HKISI, Chinese Academy of Sciences; Macau University of Science and Technology

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f?logo=googlechrome&logoColor=white)](https://bq-wang0511.github.io/TalkLikeYou/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.06658)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Checkpoints-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/doubi-killer/TalkLikeYou)
[![Streaming RealTime Demo](https://img.shields.io/badge/Streaming-RealTime%20Demo-8b5cf6)](https://github.com/BQ-Wang0511/TalkLikeYou-Streaming-RealTime-Demo)

## Introduction

TalkLikeYou imitates a target person's speaking habits for real-time audio-driven talking-head generation. It uses a one-step Flow Matching Motion Generator in an 18-dimensional lip-motion space and supports either a preset habit ID or a reference video.

The separate [Streaming RealTime Demo](https://github.com/BQ-Wang0511/TalkLikeYou-Streaming-RealTime-Demo) provides a browser interface for live microphone animation, video dubbing, and interactive chat-driven avatars.

<p align="center">
  <img src="assets/teaser.png" alt="TalkLikeYou overview" width="100%">
</p>

## To do List

- [x] Release Inference Code
- [x] Release Checkpoint
- [x] Release [Streaming RealTime Demo](https://github.com/BQ-Wang0511/TalkLikeYou-Streaming-RealTime-Demo)
- [ ] PLAD Evaluation

## Installation

The recommended environment is Linux, Python 3.10, CUDA 12.1, cuDNN 9, and an NVIDIA GPU with at least 12 GB of memory. FFmpeg must be available on `PATH`.

```bash
conda env create -f environment.yaml
conda activate talklikeyou
```

Alternatively, install a CUDA-compatible PyTorch build and then run:

```bash
pip install -r requirements.txt
```

## Checkpoints

Download the inference checkpoints from [Hugging Face](https://huggingface.co/doubi-killer/TalkLikeYou) into the repository root:

```bash
hf download doubi-killer/TalkLikeYou --include "checkpoints/**" --local-dir .
```

Keep the downloaded `checkpoints/` directory structure unchanged. Model weights are not included in this code repository; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for their usage terms.

## Inference

Run commands from the repository root. Recommended preset IDs are `192`, `166`, and `202`.

### Recommended configuration: image source with a preset habit

```bash
CUDA_VISIBLE_DEVICES=0 python -m talklikeyou \
  --source examples/source.jpg \
  --audio examples/audio.wav \
  --person-id 192 \
  --stage0 \
  --neutral \
  --motion-scale 1.0 \
  --guidance-scale 1.3 \
  --sampling-steps 1 \
  --pose-mode audio \
  --pose-sampling-steps 1 \
  --output outputs/preset_192.mp4
```

### Image source with a reference habit

```bash
CUDA_VISIBLE_DEVICES=0 python -m talklikeyou \
  --source examples/source.jpg \
  --audio examples/audio.wav \
  --habit-reference examples/habit_reference.mp4 \
  --stage0 \
  --neutral \
  --motion-scale 1.0 \
  --guidance-scale 1.3 \
  --sampling-steps 1 \
  --output outputs/reference_habit.mp4
```

The reference video must contain at least 100 frames. It overrides `--person-id` when provided.

### Video source

A source video does not require a habit-reference video.

```bash
CUDA_VISIBLE_DEVICES=0 python -m talklikeyou \
  --source path/to/source.mp4 \
  --audio path/to/audio.wav \
  --person-id 166 \
  --stage0 \
  --neutral \
  --motion-scale 1.0 \
  --guidance-scale 1.3 \
  --sampling-steps 1 \
  --pose-mode source \
  --output outputs/dubbed.mp4
```

Each command produces only one final MP4. If `--output` is omitted, a randomly named video is written to `outputs/`. Input and reference videos should use 25 FPS.

### Common options

| Option | Default | Description |
| --- | --- | --- |
| `--source` | Required | Source portrait image or video. |
| `--audio` | Required | Driving audio file. |
| `--person-id` | `192` | Preset speaking habit. Recommended IDs: `192`, `166`, and `202`. |
| `--habit-reference` | None | Imitates the speaking habit in a reference video or motion pickle and overrides `--person-id`. |
| `--stage0` | Off | Uses the built-in neutral mouth expression as the relative starting state. Recommended together with `--neutral`. |
| `--neutral` | Off | Neutralizes the source mouth to a stable, slightly open state before animation. Recommended together with `--stage0`. |
| `--motion-scale` | `1.0` | Scales the generated lip-motion amplitude. |
| `--guidance-scale` | `1.3` | Controls habit-conditioning strength. |
| `--sampling-steps` | `1` | Flow Matching sampling steps. The recommended paper setting is one step. |
| `--pose-mode` | Automatic | Uses Ditto audio pose for an image and native source pose for a video. Accepts `audio` or `source`. |
| `--pose-sampling-steps` | `1` | Ditto pose sampling steps when `--pose-mode audio` is active. |
| `--seed` | `0` | Random seed for motion generation. |
| `--no-smooth` | Off | Disables temporal smoothing. |
| `--crop-scale` | `2.3` | Face crop scale. |
| `--face-index` | `0` | Face index when multiple faces are detected. |
| `--output` | Random path | Final MP4 path; otherwise a random filename is created under `outputs/`. |

### Stage0, Neutral, and pose modes

`--stage0` generates lip motion relative to the built-in neutral mouth expression. `--neutral` first converts the source mouth to the same stable, slightly open configuration so that teeth and lip structure remain consistent. Using both options is recommended for an image source. For a video source, `--neutral` automatically uses relative lip motion while preserving the source video's native head pose. No separate neutral image is required.

Pose mode is selected automatically from the source type:

- An image source defaults to `--pose-mode audio`: TalkLikeYou generates habit-aware lip motion with its one-step Flow Matching Motion Generator, while Ditto generates the accompanying audio-driven head pose. The default Ditto sampling setting is `--pose-sampling-steps 1`.
- A video source defaults to `--pose-mode source`: it preserves the video's native head pose and does not load Ditto's pose-generation model.

To preserve the pose of an image and skip Ditto's audio-driven pose generation, select source-pose mode explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m talklikeyou \
  --source examples/source.jpg \
  --audio examples/audio.wav \
  --person-id 192 \
  --pose-mode source \
  --output outputs/source_pose.mp4
```

Either automatic choice can be overridden explicitly with `--pose-mode audio` or `--pose-mode source`. Model initialization, face registration, optional neutralization, and audio muxing are one-time preprocessing/postprocessing costs and are not included in steady-state rendering FPS.

## Citation

```bibtex
@misc{wang2026talklikeyouimitating,
  title={Talk Like You: Imitating How You Speak in Real-Time Talking Head Generation},
  author={Baiqin Wang and Zhixing Ding and Jijie Li and Jiankuo Zhao and Zhen Lei and Xiangyu Zhu},
  year={2026},
  eprint={2610.06658},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.06658}
}
```

## License

The TalkLikeYou code is released under the [MIT License](LICENSE). Third-party code and model files remain subject to their original licenses and usage terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contact

For questions, contact [wangbaiqin0511@gmail.com](mailto:wangbaiqin0511@gmail.com).
