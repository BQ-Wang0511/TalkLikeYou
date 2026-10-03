# Third-party notices

TalkLikeYou contains or adapts components from the projects below. Their original licenses continue to apply to those components.

## Ditto

- Project: https://github.com/antgroup/ditto-talkinghead
- Location: `talklikeyou/ditto/`, `checkpoints/ditto/`
- License: Apache License 2.0; see `licenses/DITTO-APACHE-2.0.txt`.

## LivePortrait

- Project: https://github.com/KwaiVGI/LivePortrait
- Location: `src/` and the corresponding inference checkpoints under `checkpoints/`
- License: MIT; see `licenses/LIVEPORTRAIT-MIT.txt`.
- The bundled InsightFace model files are limited to non-commercial research use by their model terms. They must be removed and replaced for commercial use.

## FaceFormer

- Project: https://github.com/EvelynFan/FaceFormer
- Location: portions adapted for the temporal habit encoder in `talklikeyou/models/habit_encoder.py`
- License: MIT; see `licenses/FACEFORMER-MIT.txt`.

## Wav2Lip

- Project: https://github.com/Rudrabha/Wav2Lip
- Location: the audio-encoder architecture and `checkpoints/audio_encoder.pth`
- The upstream repository states that its released results and models are for research, academic, or personal use. Consult the upstream terms before commercial use.

The TalkLikeYou MIT license does not override third-party code, model, or checkpoint terms.
