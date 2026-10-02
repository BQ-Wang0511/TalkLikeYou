#!/usr/bin/env python3
"""TalkLikeYou inference: motion generation, habit control, and portrait rendering."""
from __future__ import annotations

import argparse
import copy
import gc
import os
import pickle
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import librosa
import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CURRENT_DIR = str(REPO_ROOT)

from .ditto.stream_pipeline import StreamSDK
from .ditto.core.atomic_components.condition_handler import _mirror_index
from .ditto.core.atomic_components.motion_stitch import (
    _fix_exp_for_x_d_info_v2,
    _fix_gaze,
    _mix_s_d_info,
    bin66_to_degree,
    ctrl_motion,
    ctrl_vad,
    fade,
    transform_keypoint,
)
from .motion_generation import MotionGenerationEngine
from .portrait_template import create_pipeline, video_fps
from src.utils.camera import get_rotation_matrix, headpose_pred_to_degree
from src.utils.io import load_image_rgb, load_video, resize_to_limit


LIP_IDX = [6, 12, 14, 17, 19, 20]
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def read_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_input_path(path: str, fallback_base: str | None = None) -> str:
    if os.path.isabs(path):
        return path
    direct = os.path.abspath(path)
    if os.path.exists(direct):
        return direct
    if fallback_base is not None:
        fallback = os.path.abspath(os.path.join(fallback_base, path))
        if os.path.exists(fallback):
            return fallback
    return direct


def default_video_path() -> str:
    return str(REPO_ROOT / "outputs" / f"{uuid.uuid4().hex[:12]}.mp4")


def create_motion_template_pipeline(crop_scale: float, face_idx: int):
    pipeline_args = SimpleNamespace(scale=crop_scale, face_idx=face_idx)
    return create_pipeline(pipeline_args)


def release_motion_template_pipeline(pipeline) -> None:
    if pipeline is None:
        return
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extend_list_mirror(items: list, length: int) -> list:
    items = [copy.deepcopy(item) for item in items]
    if length == 0:
        return items
    mirrored = items + [copy.deepcopy(item) for item in items[::-1]]
    if length <= len(mirrored):
        return mirrored[:length]
    out = mirrored[:]
    base_len = len(mirrored)
    for idx in range(length - base_len):
        out.append(copy.deepcopy(mirrored[idx % base_len]))
    return out


def reshape_motion_points(value, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (1, 21, 3):
        return arr
    if arr.size != 63:
        raise ValueError(f"Unexpected shape for {field_name}: {arr.shape}")
    return arr.reshape(1, 21, 3)


def flatten_motion_points(value, field_name: str) -> np.ndarray:
    arr = reshape_motion_points(value, field_name)
    return arr.reshape(1, 63).astype(np.float32)


def build_motion_template_base(
    source_path: str,
    crop_scale: float,
    face_idx: int,
    output_fps: float,
    pipeline=None,
) -> dict:
    if pipeline is None:
        args = SimpleNamespace(scale=crop_scale, face_idx=face_idx)
        pipeline = create_pipeline(args)
    pipeline.cropper.crop_cfg.scale = crop_scale

    if source_path.lower().endswith(VIDEO_EXTS):
        source_rgb_lst = load_video(source_path)
    else:
        source_rgb_lst = [load_image_rgb(source_path)]

    source_rgb_lst = [
        resize_to_limit(
            img,
            pipeline.live_portrait_wrapper.inference_cfg.source_max_dim,
            pipeline.live_portrait_wrapper.inference_cfg.source_division,
        )
        for img in source_rgb_lst
    ]

    ret = pipeline.cropper.crop_source_video(
        source_rgb_lst, pipeline.cropper.crop_cfg, face_idx=face_idx
    )
    crop_frames = [cv2.resize(frame, (256, 256)) for frame in ret["frame_crop_lst"]]
    I_d_lst = pipeline.live_portrait_wrapper.prepare_videos(crop_frames)
    c_eyes_lst, c_lip_lst = pipeline.live_portrait_wrapper.calc_ratio(ret["lmk_crop_lst"])
    template = pipeline.make_motion_template(I_d_lst, c_eyes_lst, c_lip_lst, output_fps=output_fps)
    return template


def finalize_motion_template(template_base: dict, num_frames: int, output_fps: float) -> dict:
    template = copy.deepcopy(template_base)
    template["output_fps"] = float(output_fps)
    template["n_frames"] = num_frames
    template["motion"] = extend_list_mirror(template["motion"], num_frames)
    template["c_eyes_lst"] = extend_list_mirror(template["c_eyes_lst"], num_frames)
    template["c_lip_lst"] = extend_list_mirror(template["c_lip_lst"], num_frames)
    return template


def to_template_motion_item(motion_info: dict, template_motion: dict) -> dict:
    pitch = torch.from_numpy(np.asarray(motion_info["pitch"], dtype=np.float32))
    yaw = torch.from_numpy(np.asarray(motion_info["yaw"], dtype=np.float32))
    roll = torch.from_numpy(np.asarray(motion_info["roll"], dtype=np.float32))
    R = get_rotation_matrix(
        headpose_pred_to_degree(pitch),
        headpose_pred_to_degree(yaw),
        headpose_pred_to_degree(roll),
    ).cpu().numpy().astype(np.float32)

    return {
        "scale": np.asarray(motion_info["scale"], dtype=np.float32),
        "R": R,
        "exp": reshape_motion_points(motion_info["exp"], "exp"),
        "t": np.asarray(motion_info["t"], dtype=np.float32),
        "kp": reshape_motion_points(motion_info.get("kp", template_motion["kp"]), "kp"),
        "x_s": np.asarray(template_motion["x_s"], dtype=np.float32),
    }


def combine_base_motion(template: dict, base_motion: list) -> dict:
    n_frames = min(len(template["motion"]), len(base_motion))
    out = copy.deepcopy(template)
    out["n_frames"] = n_frames
    out["motion"] = out["motion"][:n_frames]
    out["c_eyes_lst"] = out["c_eyes_lst"][:n_frames]
    out["c_lip_lst"] = out["c_lip_lst"][:n_frames]
    for idx in range(n_frames):
        out["motion"][idx] = to_template_motion_item(
            base_motion[idx], template["motion"][idx]
        )
    if n_frames > 0:
        for idx in range(n_frames):
            out["motion"][idx]["exp"][:, LIP_IDX] = np.asarray(
                template["motion"][0]["exp"][:, LIP_IDX], dtype=np.float32
            ).copy()
    return out


def compute_base_motion(sdk: StreamSDK, audio_path: str) -> tuple[list, int]:
    audio, _ = librosa.core.load(audio_path, sr=16000)
    aud_feat = sdk.wav2feat.wav2feat(audio)
    aud_cond_all = sdk.condition_handler(aud_feat, 0)
    num_frames = len(aud_cond_all)

    seq_frames = sdk.audio2motion.seq_frames
    valid_clip_len = sdk.audio2motion.valid_clip_len
    idx = 0
    res_kp_seq = None
    while idx < num_frames:
        aud_cond = aud_cond_all[idx : idx + seq_frames][None]
        if aud_cond.shape[1] < seq_frames:
            pad = np.stack([aud_cond[:, -1]] * (seq_frames - aud_cond.shape[1]), axis=1)
            aud_cond = np.concatenate([aud_cond, pad], axis=1)
        res_kp_seq = sdk.audio2motion(aud_cond, res_kp_seq)
        idx += valid_clip_len

    res_kp_seq = res_kp_seq[:, :num_frames]
    res_kp_seq = sdk.audio2motion._smo(res_kp_seq, 0, res_kp_seq.shape[1])
    return sdk.audio2motion.cvt_fmt(res_kp_seq), num_frames


def _load_habit_reference(args) -> dict | None:
    if args.habit_reference is None:
        return None
    if args.habit_reference.lower().endswith(".pkl"):
        return read_pickle(args.habit_reference)

    pipeline = create_motion_template_pipeline(args.crop_scale, args.face_idx)
    try:
        return build_motion_template_base(
            source_path=args.habit_reference,
            crop_scale=args.crop_scale,
            face_idx=args.face_idx,
            output_fps=video_fps(args.habit_reference),
            pipeline=pipeline,
        )
    finally:
        release_motion_template_pipeline(pipeline)


def generate_habit_motion(args, template_data: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = MotionGenerationEngine(
        motion_checkpoint=args.motion_checkpoint,
        audio_checkpoint=args.audio_encoder_checkpoint,
        habit_checkpoint=args.habit_encoder_checkpoint,
        device=device,
        sampling_steps=args.sampling_steps,
        guidance_scale=args.guidance_scale,
    )
    try:
        return engine.generate(
            template_data=template_data,
            audio_path=args.audio_path,
            person_id=args.person_id,
            reference_data=_load_habit_reference(args),
            motion_scale=args.motion_scale,
            smooth=not args.no_smooth,
            overlap=args.overlap,
            seed=args.seed,
        )
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def replace_lip_expression(base_motion: list, generated_motion: dict) -> list:
    n_frames = min(len(base_motion), len(generated_motion["motion"]))
    out = [copy.deepcopy(base_motion[idx]) for idx in range(n_frames)]
    for idx in range(n_frames):
        generated_expression = reshape_motion_points(
            generated_motion["motion"][idx]["exp"], "exp"
        )
        exp_flat = flatten_motion_points(out[idx]["exp"], "exp")
        exp_frame = exp_flat.reshape(1, 21, 3)
        exp_frame[0, LIP_IDX] = generated_expression[0, LIP_IDX]
        out[idx]["exp"] = exp_frame.reshape(1, 63).astype(np.float32)
    return out


def configure_generated_lip_motion(sdk: StreamSDK) -> None:
    base_motion_stitch = sdk.motion_stitch

    class GeneratedLipMotion:
        def __init__(self, base):
            self._base = base

        def __getattr__(self, name):
            return getattr(self._base, name)

        def __call__(self, x_s_info, x_d_info, **kwargs):
            base = self._base
            kwargs = base._merge_kwargs(base.overall_ctrl_info, kwargs)

            if base.scale_ratio is None:
                base.scale_b = x_s_info["scale"].item()
                base.scale_ratio = base.scale_a / base.scale_b
                base._set_scale_ratio(base.scale_ratio)

            generated_lip_expression = reshape_motion_points(
                x_d_info["exp"], "exp"
            ).copy()
            if base.relative_d and base.d0 is None:
                base.d0 = copy.deepcopy(x_d_info)

            x_d_info = _mix_s_d_info(
                x_s_info,
                x_d_info,
                base.use_d_keys,
                base.d0,
            )

            delta_eye = 0
            if base.drive_eye and base.delta_eye_arr is not None:
                delta_eye = base.delta_eye_arr[
                    base.delta_eye_idx_list[base.idx % len(base.delta_eye_idx_list)]
                ][None]
            x_d_info = _fix_exp_for_x_d_info_v2(
                x_d_info,
                x_s_info,
                delta_eye,
                base.fix_exp_a1,
                base.fix_exp_a2,
                base.fix_exp_a3,
            )

            if kwargs.get("vad_alpha", 1) < 1:
                x_d_info = ctrl_vad(x_d_info, x_s_info, kwargs["vad_alpha"])
            x_d_info = ctrl_motion(x_d_info, **kwargs)

            if base.fade_type == "d0" and base.fade_dst is None:
                base.fade_dst = copy.deepcopy(x_d_info)
            if "fade_alpha" in kwargs and base.fade_type in ["d0", "s"]:
                fade_dst = base.fade_dst
                if base.fade_type == "s" and fade_dst is None:
                    fade_dst = copy.deepcopy(x_s_info)
                    if base.is_image_flag:
                        base.fade_dst = fade_dst
                x_d_info = fade(
                    x_d_info,
                    fade_dst,
                    kwargs["fade_alpha"],
                    kwargs.get("fade_out_keys", base.fade_out_keys),
                )

            if base.drive_eye:
                if base.pose_s is None:
                    yaw_s = bin66_to_degree(x_s_info["yaw"]).item()
                    pitch_s = bin66_to_degree(x_s_info["pitch"]).item()
                    base.pose_s = [yaw_s, pitch_s]
                x_d_info = _fix_gaze(base.pose_s, x_d_info)

            exp_frame = reshape_motion_points(x_d_info["exp"], "exp")
            exp_frame[0, LIP_IDX] = generated_lip_expression[0, LIP_IDX]
            x_d_info["exp"] = exp_frame.reshape(1, 63).astype(np.float32)

            if base.x_s is not None:
                x_s = base.x_s
            else:
                x_s = transform_keypoint(x_s_info)
                if base.is_image_flag:
                    base.x_s = x_s

            x_d = transform_keypoint(x_d_info)
            if base.flag_stitching:
                x_d = base.stitch_net(x_s, x_d)

            base.idx += 1
            return x_s, x_d

    sdk.motion_stitch = GeneratedLipMotion(base_motion_stitch)


def render_frames(sdk: StreamSDK, motion_frames: list) -> None:
    for gen_frame_idx, x_d_info in enumerate(motion_frames):
        frame_idx = _mirror_index(gen_frame_idx, sdk.source_info_frames)
        ctrl_kwargs = sdk._get_ctrl_info(gen_frame_idx)
        sdk.motion_stitch_queue.put([frame_idx, x_d_info, ctrl_kwargs])
    sdk.motion_stitch_queue.put(None)


def mux_audio(tmp_video_path: str, audio_path: str, output_video_path: str) -> None:
    ensure_dir(os.path.dirname(output_video_path))
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        tmp_video_path,
        "-i",
        audio_path,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        output_video_path,
    ]
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a habit-aware talking portrait with TalkLikeYou."
    )
    parser.add_argument("--source", dest="source_path", required=True, help="Source image or video")
    parser.add_argument("--audio", dest="audio_path", required=True, help="Driving audio")
    parser.add_argument("--output", "--output-video", dest="output_video", default=None)
    parser.add_argument(
        "--person-id",
        type=int,
        default=192,
        help="Preset speaking habit ID; recommended examples: 192, 166, 202",
    )
    parser.add_argument(
        "--habit-reference",
        "--motion-reference",
        dest="habit_reference",
        default=None,
        help="Optional reference video or motion pickle; overrides --person-id",
    )
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.3)
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=1,
        help="Flow Matching steps; the paper setting is 1",
    )
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-smooth", dest="no_smooth", action="store_true")
    parser.add_argument("--crop-scale", type=float, default=2.3)
    parser.add_argument("--face-index", dest="face_idx", type=int, default=0)
    parser.add_argument(
        "--motion-checkpoint",
        default=str(REPO_ROOT / "checkpoints" / "motion_generator.pt"),
    )
    parser.add_argument(
        "--habit-encoder-checkpoint",
        default=str(REPO_ROOT / "checkpoints" / "habit_encoder.pt"),
    )
    parser.add_argument(
        "--audio-encoder-checkpoint",
        default=str(REPO_ROOT / "checkpoints" / "audio_encoder.pth"),
    )
    parser.add_argument(
        "--renderer-config",
        dest="cfg_pkl",
        default=str(REPO_ROOT / "checkpoints" / "ditto" / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"),
    )
    parser.add_argument(
        "--renderer-checkpoint-dir",
        dest="data_root",
        default=str(REPO_ROOT / "checkpoints" / "ditto" / "ditto_pytorch"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    args.source_path = resolve_input_path(args.source_path, CURRENT_DIR)
    args.audio_path = resolve_input_path(args.audio_path, CURRENT_DIR)
    args.cfg_pkl = resolve_input_path(args.cfg_pkl, CURRENT_DIR)
    args.data_root = resolve_input_path(args.data_root, CURRENT_DIR)
    args.motion_checkpoint = resolve_input_path(args.motion_checkpoint, CURRENT_DIR)
    args.habit_encoder_checkpoint = resolve_input_path(args.habit_encoder_checkpoint, CURRENT_DIR)
    args.audio_encoder_checkpoint = resolve_input_path(args.audio_encoder_checkpoint, CURRENT_DIR)
    args.habit_reference = (
        None if args.habit_reference is None else resolve_input_path(args.habit_reference, CURRENT_DIR)
    )
    args.output_video = (
        os.path.abspath(args.output_video)
        if args.output_video is not None
        else default_video_path()
    )

    for checkpoint in (
        args.motion_checkpoint,
        args.habit_encoder_checkpoint,
        args.audio_encoder_checkpoint,
        args.cfg_pkl,
    ):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)

    sdk = StreamSDK(args.cfg_pkl, args.data_root)
    sdk.setup(args.source_path, args.output_video, crop_scale=args.crop_scale)
    render_started = False
    try:
        print("=" * 60)
        print("Stage 1: Generate base facial motion")
        print("=" * 60)
        base_motion, num_frames = compute_base_motion(sdk, args.audio_path)

        sdk.setup_Nd(N_d=num_frames, fade_in=-1, fade_out=-1, ctrl_info={})

        print("=" * 60)
        print("Stage 2: Build the motion template")
        print("=" * 60)
        output_fps = float(video_fps(args.source_path)) if args.source_path.lower().endswith(VIDEO_EXTS) else 25.0
        shell_template_base = build_motion_template_base(
            source_path=args.source_path,
            crop_scale=args.crop_scale,
            face_idx=args.face_idx,
            output_fps=output_fps,
        )
        motion_template = finalize_motion_template(shell_template_base, num_frames, output_fps)
        template_data = combine_base_motion(motion_template, base_motion)
        template_data["output_fps"] = output_fps

        print("=" * 60)
        print("Stage 3: Generate one-step habit-aware lip motion")
        print("=" * 60)
        generated_motion = generate_habit_motion(args, template_data)

        print("=" * 60)
        print("Stage 4: Render the generated motion")
        print("=" * 60)
        final_motion = replace_lip_expression(base_motion, generated_motion)

        configure_generated_lip_motion(sdk)
        render_started = True
        render_frames(sdk, final_motion)
        sdk.close()

        mux_audio(sdk.tmp_output_path, args.audio_path, args.output_video)
        print(f"Final video saved to: {args.output_video}")
    finally:
        if not render_started:
            try:
                sdk.motion_stitch_queue.put(None)
                sdk.close()
            except Exception:
                pass
        if os.path.exists(sdk.tmp_output_path):
            os.remove(sdk.tmp_output_path)


if __name__ == "__main__":
    main()
