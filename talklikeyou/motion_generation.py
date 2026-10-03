from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .audio_features import AudioFeatureExtractor
from .models import FlowMatchingMotionGenerator, HabitEncoder
from .smoothing import kalman_smooth


LIP_KEYPOINTS = [6, 12, 14, 17, 19, 20]


def _expression(frame: dict) -> np.ndarray:
    return np.asarray(frame["exp"], dtype=np.float32).reshape(21, 3)


def _lip_motion(frame: dict) -> np.ndarray:
    return _expression(frame)[LIP_KEYPOINTS].reshape(-1)


class MotionGenerationEngine:
    """One-step habit-aware Flow Matching motion inference."""

    def __init__(
        self,
        motion_checkpoint: str | Path,
        audio_checkpoint: str | Path,
        habit_checkpoint: str | Path,
        device: torch.device,
        sampling_steps: int = 1,
        guidance_scale: float = 1.3,
    ):
        self.device = device
        state = torch.load(str(motion_checkpoint), map_location="cpu", weights_only=True)
        config = dict(state["config"])
        config["sampling_steps"] = sampling_steps
        config["guidance_scale"] = guidance_scale
        self.config = config

        self.motion_generator = FlowMatchingMotionGenerator(
            motion_feat_dim=config["motion_feat_dim"],
            person_num=config["person_num"],
            audio_feat_dim=config["audio_feat_dim"],
            seq_frames=config["sequence_length"],
            latent_dim=config["latent_dim"],
            ff_size=config["feedforward_dim"],
            num_layers=config["layer_count"],
            num_heads=config["head_count"],
            dropout=0.0,
            time_steps=config["time_steps"],
            sampling_steps=config["sampling_steps"],
            guidance_scale=config["guidance_scale"],
            checkpoint=str(motion_checkpoint),
            device=str(device),
            predict_clean_motion=config["predict_clean_motion"],
        )
        self.motion_generator.eval()
        self.audio_extractor = AudioFeatureExtractor(audio_checkpoint, device)
        self.habit_encoder = self._load_habit_encoder(habit_checkpoint)

    def _load_habit_encoder(self, checkpoint: str | Path) -> HabitEncoder:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        model_state = state["model_state_dict"]
        encoder = HabitEncoder(
            motion_dim=model_state["vertice_map.weight"].shape[1],
            feature_dim=model_state["vertice_map.weight"].shape[0],
            habit_dim=model_state["vertice_map_r.weight"].shape[0],
            reference_frames=100,
        ).to(self.device)
        encoder.load_state_dict(model_state, strict=True)
        encoder.eval()
        return encoder

    def _habit_embedding(self, reference_data: dict) -> torch.Tensor:
        motion = reference_data.get("motion", [])
        reference_frames = 100
        if len(motion) < reference_frames:
            raise ValueError(
                f"Habit reference needs at least {reference_frames} frames; got {len(motion)}"
            )
        start = (len(motion) - reference_frames) // 2
        values = np.stack(
            [_expression(motion[index])[LIP_KEYPOINTS] for index in range(start, start + reference_frames)]
        )
        tensor = torch.from_numpy(values).permute(0, 2, 1).unsqueeze(0).to(self.device)
        return self.habit_encoder(tensor)

    def extract_audio_features(self, audio_path: str | Path) -> np.ndarray:
        return self.audio_extractor(audio_path)

    @torch.inference_mode()
    def generate(
        self,
        template_data: dict,
        audio_path: str | Path | None = None,
        audio_features: np.ndarray | None = None,
        person_id: int = 192,
        reference_data: dict | None = None,
        motion_scale: float = 1.0,
        smooth: bool = True,
        overlap: int | None = None,
        seed: int = 0,
    ) -> dict:
        person_num = self.config["person_num"]
        if not 0 <= person_id < person_num:
            raise ValueError(f"person_id must be in [0, {person_num - 1}]")

        if audio_features is None:
            if audio_path is None:
                raise ValueError("audio_path is required when audio_features is not provided")
            audio_features = self.extract_audio_features(audio_path)
        total_frames = min(len(template_data["motion"]), len(audio_features))
        if total_frames == 0:
            raise ValueError("No frames are available for motion generation")

        sequence_length = self.config["sequence_length"]
        overlap = self.config.get("overlap", 5) if overlap is None else overlap
        overlap = min(max(overlap, 0), sequence_length - 1)
        stride = sequence_length - overlap
        one_hot = torch.zeros(1, person_num, device=self.device)
        habit_embedding = None
        if reference_data is None:
            one_hot[:, person_id] = 1.0
        else:
            habit_embedding = self._habit_embedding(reference_data)

        random_generator = torch.Generator(device=self.device)
        random_generator.manual_seed(seed)
        chunks = []
        previous_last = None
        for start in tqdm(range(0, total_frames, stride), desc="habit motion"):
            end = min(start + sequence_length, total_frames)
            audio = torch.from_numpy(audio_features[start:end]).unsqueeze(0).to(self.device)
            initial = _lip_motion(template_data["motion"][start]) if previous_last is None else previous_last
            condition = torch.from_numpy(initial).unsqueeze(0).to(self.device)
            noise = torch.randn(
                (1, end - start, self.config["motion_feat_dim"]),
                generator=random_generator,
                device=self.device,
            )
            prediction = self.motion_generator.sample(
                condition,
                audio,
                one_hot,
                noise=noise,
                habit_emb=habit_embedding,
            ).squeeze(0).cpu().numpy()
            previous_last = prediction[-1].copy()
            chunks.append(prediction if start == 0 else prediction[min(overlap, len(prediction) - 1) :])
            if end >= total_frames:
                break

        prediction = np.concatenate(chunks, axis=0)[:total_frames]
        if smooth and len(prediction) > 1:
            prediction = kalman_smooth(prediction, self.device)

        output = copy.deepcopy(template_data)
        output["n_frames"] = total_frames
        output["motion"] = output["motion"][:total_frames]
        for index, values in enumerate(prediction):
            expression = _expression(output["motion"][index])[None]
            expression[0, LIP_KEYPOINTS] = values.reshape(6, 3) * motion_scale
            output["motion"][index]["exp"] = expression.astype(np.float32)
        return output
