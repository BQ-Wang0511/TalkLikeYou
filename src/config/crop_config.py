# coding: utf-8

"""
parameters used for crop faces
"""

from dataclasses import dataclass

from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)  # use repr from PrintableConfig
class CropConfig(PrintableConfig):
    insightface_root: str = make_abs_path("../../pretrained_weights/insightface")
    landmark_ckpt_path: str = make_abs_path("../../pretrained_weights/liveportrait/landmark.onnx")
    device_id: int = 0
    flag_force_cpu: bool = False
    det_thresh: float = 0.1
    dsize: int = 512
    scale: float = 2.3
    vx_ratio: float = 0
    vy_ratio: float = -0.125
    max_face_num: int = 0
    flag_do_rot: bool = True
    scale_crop_driving_video: float = 2.2
    vx_ratio_crop_driving_video: float = 0.0
    vy_ratio_crop_driving_video: float = -0.1
    direction: str = "left-right"
