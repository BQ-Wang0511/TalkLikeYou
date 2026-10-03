"""Neutral-source preprocessing shared by image animation and video dubbing."""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import torch

from src.utils.camera import get_rotation_matrix
from src.utils.crop import paste_back, prepare_paste_back
from src.utils.io import load_image_rgb, load_video, resize_to_limit


LIP_KEYPOINTS = [6, 12, 14, 17, 19, 20]
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
TARGET_LIP_OPEN_RATIO = 0.2
LIP_CLOSE_OPEN = 40.0
GRIN = 4.86
LIP_NORMALIZE_THRESHOLD = 0.03
NEUTRAL_LIP_EXPRESSION = np.asarray(
    [
        [0.0000063777, -0.0000252724, -0.0000009537],
        [0.0000043511, 0.0000156760, 0.0000007153],
        [0.0047416687, 0.0022163391, -0.0000014305],
        [-0.0056610107, -0.0011396408, 0.0005669594],
        [0.0002758503, -0.0224151611, -0.0088729858],
        [0.0013027191, 0.0039100647, 0.0001378059],
    ],
    dtype=np.float32,
)


def _crop(image_rgb: np.ndarray, pipeline, face_index: int, description: str):
    wrapper = pipeline.live_portrait_wrapper
    image_rgb = resize_to_limit(
        image_rgb,
        wrapper.inference_cfg.source_max_dim,
        wrapper.inference_cfg.source_division,
    )
    crop = pipeline.cropper.crop_source_image(
        image_rgb,
        pipeline.cropper.crop_cfg,
        face_idx=face_index,
    )
    if crop is None:
        raise RuntimeError(f"No face detected in {description}")
    return image_rgb, crop


def _apply_stage0_controls(
    image_rgb: np.ndarray,
    pipeline,
    face_index: int,
) -> np.ndarray:
    image_rgb, crop = _crop(
        image_rgb,
        pipeline,
        face_index,
        "neutralized source",
    )
    wrapper = pipeline.live_portrait_wrapper
    source = wrapper.prepare_source(crop["img_crop_256x256"])
    features = wrapper.extract_feature_3d(source)
    motion = wrapper.get_kp_info(source)
    source_points = wrapper.transform_keypoint(motion)
    expression = motion["exp"].clone()
    expression[0, 20, 2] -= GRIN * 0.001
    expression[0, 20, 1] -= GRIN * 0.001
    expression[0, 14, 1] -= GRIN * 0.001
    expression[0, 19, 1] += LIP_CLOSE_OPEN * 0.001
    expression[0, 19, 2] += LIP_CLOSE_OPEN * 0.0001
    expression[0, 17, 1] -= LIP_CLOSE_OPEN * 0.0001

    rotation = get_rotation_matrix(
        motion["pitch"], motion["yaw"], motion["roll"]
    )
    target_points = motion["scale"] * (
        motion["kp"] @ rotation + expression
    ) + motion["t"]
    lip_ratio = wrapper.calc_combined_lip_ratio(
        [[TARGET_LIP_OPEN_RATIO]],
        crop["lmk_crop"],
    )
    target_points += wrapper.retarget_lip(source_points, lip_ratio)
    target_points = wrapper.stitching(source_points, target_points)
    output = wrapper.warp_decode(features, source_points, target_points)
    output = wrapper.parse_output(output["out"])[0]
    mask = prepare_paste_back(
        wrapper.inference_cfg.mask_crop,
        crop["M_c2o"],
        dsize=(image_rgb.shape[1], image_rgb.shape[0]),
    )
    return paste_back(output, crop["M_c2o"], image_rgb, mask)


def _neutralize_frame(
    source_rgb: np.ndarray,
    pipeline,
    face_index: int,
) -> np.ndarray:
    source_rgb, crop = _crop(
        source_rgb,
        pipeline,
        face_index,
        "source frame",
    )
    wrapper = pipeline.live_portrait_wrapper
    source = wrapper.prepare_source(crop["img_crop_256x256"])
    features = wrapper.extract_feature_3d(source)
    motion = wrapper.get_kp_info(source)
    source_points = wrapper.transform_keypoint(motion)
    expression = motion["exp"].clone()
    expression[:, LIP_KEYPOINTS, :] = torch.as_tensor(
        NEUTRAL_LIP_EXPRESSION,
        device=expression.device,
        dtype=expression.dtype,
    )
    rotation = get_rotation_matrix(
        motion["pitch"], motion["yaw"], motion["roll"]
    )
    target_points = motion["scale"] * (
        motion["kp"] @ rotation + expression
    ) + motion["t"]
    target_points = wrapper.stitching(source_points, target_points)
    output = wrapper.warp_decode(features, source_points, target_points)
    output = wrapper.parse_output(output["out"])[0]
    mask = prepare_paste_back(
        wrapper.inference_cfg.mask_crop,
        crop["M_c2o"],
        dsize=(source_rgb.shape[1], source_rgb.shape[0]),
    )
    rendered = paste_back(output, crop["M_c2o"], source_rgb, mask)
    return _apply_stage0_controls(rendered, pipeline, face_index)


def neutralize_source(
    source_path: str,
    pipeline,
    face_index: int,
) -> tuple[str, str]:
    """Create a temporary source with a consistent, slightly open mouth."""
    temporary_dir = tempfile.mkdtemp(prefix="talklikeyou-neutral-")

    if source_path.lower().endswith(VIDEO_EXTENSIONS):
        frames = load_video(source_path)
        if not frames:
            raise ValueError(f"Source video contains no frames: {source_path}")
        capture = cv2.VideoCapture(source_path)
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
        fps = float(fps) if fps and fps > 0 else 25.0
        output_path = os.path.join(temporary_dir, "neutralized.mp4")
        writer = None
        try:
            for frame in frames:
                output = _neutralize_frame(
                    frame,
                    pipeline,
                    face_index,
                )
                if writer is None:
                    height, width = output.shape[:2]
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Failed to create neutralized source video")
                writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        finally:
            if writer is not None:
                writer.release()
        return output_path, temporary_dir

    output = _neutralize_frame(
        load_image_rgb(source_path),
        pipeline,
        face_index,
    )
    output_path = os.path.join(temporary_dir, "neutralized.png")
    if not cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Failed to write neutralized source image")
    return output_path, temporary_dir


def compute_lip_normalization(
    source_rgb: np.ndarray,
    pipeline,
    face_index: int,
) -> np.ndarray | None:
    source_rgb, crop = _crop(
        source_rgb,
        pipeline,
        face_index,
        "source used for lip normalization",
    )
    wrapper = pipeline.live_portrait_wrapper
    tensor = wrapper.prepare_source(crop["img_crop_256x256"])
    motion = wrapper.get_kp_info(tensor)
    source_points = wrapper.transform_keypoint(motion)
    lip_ratio = wrapper.calc_combined_lip_ratio([[0.0]], crop["lmk_crop"])
    if float(lip_ratio[0][0].item()) < LIP_NORMALIZE_THRESHOLD:
        return None
    return (
        wrapper.retarget_lip(source_points, lip_ratio)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
