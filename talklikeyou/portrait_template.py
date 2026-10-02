from __future__ import annotations

import cv2
import numpy as np
from tqdm import tqdm

from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.camera import get_rotation_matrix
from src.utils.cropper import Cropper


class MotionTemplatePipeline:
    def __init__(self, inference_config: InferenceConfig, crop_config: CropConfig):
        self.live_portrait_wrapper = LivePortraitWrapper(inference_cfg=inference_config)
        self.cropper = Cropper(crop_cfg=crop_config)

    def make_motion_template(
        self,
        input_frames,
        eye_ratios,
        lip_ratios,
        output_fps: float = 25.0,
    ) -> dict:
        template = {
            "n_frames": input_frames.shape[0],
            "output_fps": output_fps,
            "motion": [],
            "c_eyes_lst": [],
            "c_lip_lst": [],
        }
        for index in tqdm(range(input_frames.shape[0]), desc="motion template"):
            motion_info = self.live_portrait_wrapper.get_kp_info(input_frames[index])
            rotation = get_rotation_matrix(
                motion_info["pitch"], motion_info["yaw"], motion_info["roll"]
            )
            source_keypoints = self.live_portrait_wrapper.transform_keypoint(motion_info)
            template["motion"].append(
                {
                    "scale": motion_info["scale"].cpu().numpy().astype(np.float32),
                    "R": rotation.cpu().numpy().astype(np.float32),
                    "exp": motion_info["exp"].cpu().numpy().astype(np.float32),
                    "t": motion_info["t"].cpu().numpy().astype(np.float32),
                    "kp": motion_info["kp"].cpu().numpy().astype(np.float32),
                    "x_s": source_keypoints.cpu().numpy().astype(np.float32),
                }
            )
            template["c_eyes_lst"].append(eye_ratios[index].astype(np.float32))
            template["c_lip_lst"].append(lip_ratios[index].astype(np.float32))
        return template


def _matching_fields(target_class, values: dict):
    return target_class(
        **{key: value for key, value in values.items() if hasattr(target_class, key)}
    )


def create_pipeline(arguments) -> MotionTemplatePipeline:
    values = vars(arguments)
    return MotionTemplatePipeline(
        inference_config=_matching_fields(InferenceConfig, values),
        crop_config=_matching_fields(CropConfig, values),
    )


def video_fps(path: str, default: float = 25.0) -> float:
    capture = cv2.VideoCapture(path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    return float(fps) if fps and fps > 0 else default
