from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
from scipy import signal

from .models.audio_encoder import AudioFeatureEncoder


SAMPLE_RATE = 16000
FPS = 25
N_FFT = 800
HOP_SIZE = 200
WIN_SIZE = 800
N_MELS = 80
MEL_FRAMES = 16


def _mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    emphasized = signal.lfilter([1, -0.97], [1], audio)
    spectrum = librosa.stft(
        y=emphasized,
        n_fft=N_FFT,
        hop_length=HOP_SIZE,
        win_length=WIN_SIZE,
    )
    mel_basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=55,
        fmax=7600,
    )
    magnitude = np.dot(mel_basis, np.abs(spectrum))
    min_level = np.exp(-100 / 20 * np.log(10))
    decibels = 20 * np.log10(np.maximum(min_level, magnitude)) - 20
    normalized = np.clip(8 * ((decibels + 100) / 100) - 4, -4, 4)
    return normalized.T.astype(np.float32)


class AudioFeatureExtractor:
    def __init__(self, checkpoint: str | Path, device: torch.device, batch_size: int = 64):
        self.device = device
        self.batch_size = batch_size
        self.model = AudioFeatureEncoder().to(device)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        self.model.audio_encoder.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, audio_path: str | Path) -> np.ndarray:
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        mel = _mel_spectrogram(audio)
        frame_count = int((mel.shape[0] - MEL_FRAMES) / 80.0 * FPS)
        if frame_count <= 0:
            raise ValueError("Audio is too short to extract motion features")

        windows = []
        for frame_index in range(frame_count):
            start = int(80.0 * frame_index / FPS)
            window = mel[start : start + MEL_FRAMES]
            if window.shape[0] == MEL_FRAMES:
                windows.append(window.T)
        if not windows:
            raise ValueError("No complete audio feature windows were found")

        values = torch.from_numpy(np.stack(windows)).unsqueeze(1)
        outputs = []
        for start in range(0, len(values), self.batch_size):
            outputs.append(self.model(values[start : start + self.batch_size].to(self.device)).cpu())
        return torch.cat(outputs, dim=0).numpy().astype(np.float32)
