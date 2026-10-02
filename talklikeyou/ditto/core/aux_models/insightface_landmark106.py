from __future__ import division
import numpy as np
import torch
import cv2
from skimage import transform as trans

from ..utils.load_model import load_model


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M
   

def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


class Landmark106:
    def __init__(self, model_path, device="cuda", **kwargs):
        kwargs["model_file"] = model_path
        kwargs["module_name"] = "Landmark106"
        kwargs["package_name"] = "..aux_models.modules"

        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

        if self.model_type != "ori":
            self._init_vars()

    def _init_vars(self):
        self.input_mean = 0.0
        self.input_std = 1.0
        self.input_size = (192, 192)
        self.lmk_num = 106

        self.output_names = ["fc1"]

    def _run_model(self, blob):
        if self.model_type == "onnx":
            pred = self.model.run(None, {"data": blob})[0]
        elif self.model_type == "tensorrt":
            self.model.setup({"data": blob})
            self.model.infer()
            pred = self.model.buffer[self.output_names[0]][0]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return pred
    
    def get(self, img, bbox):
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0]  / (max(w, h)*1.5)
        
        aimg, M = transform(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        
        blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        pred = self._run_model(blob)

        pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num*-1:,:]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = trans_points2d(pred, IM)
        return pred

    def __call__(self, img, bbox):
        if self.model_type == "ori":
            pred = self.model.get(img, bbox)
        else:
            pred = self.get(img, bbox)
        
        return pred
