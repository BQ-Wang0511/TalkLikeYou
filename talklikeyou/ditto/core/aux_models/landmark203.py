import numpy as np
from ..utils.load_model import load_model


def _transform_pts(pts, M):
    """ conduct similarity or affine transformation to the pts
    pts: Nx2 ndarray
    M: 2x3 matrix or 3x3 matrix
    return: Nx2
    """
    return pts @ M[:2, :2].T + M[:2, 2]


class Landmark203:
    def __init__(self, model_path, device="cuda", **kwargs):
        kwargs["model_file"] = model_path
        kwargs["module_name"] = "Landmark203"
        kwargs["package_name"] = "..aux_models.modules"

        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

        self.output_names = ["landmarks"]
        self.dsize = 224

    def _run_model(self, inp):
        if self.model_type == "onnx":
            out_pts = self.model.run(None, {"input": inp})[0]
        elif self.model_type == "tensorrt":
            self.model.setup({"input": inp})
            self.model.infer()
            out_pts = self.model.buffer[self.output_names[0]][0]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return out_pts
    
    def run(self, img_crop_rgb, M_c2o=None):
        # img_crop_rgb: 224x224

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)

        out_pts = self._run_model(inp)

        # 2d landmarks 203 points
        lmk = out_pts[0].reshape(-1, 2) * self.dsize  # scale to 0-224
        if M_c2o is not None:
            lmk = _transform_pts(lmk, M=M_c2o)

        return lmk
    
    def __call__(self, img_crop_rgb, M_c2o=None):
        if self.model_type == "ori":
            lmk = self.model.run(img_crop_rgb, M_c2o)
        else:
            lmk = self.run(img_crop_rgb, M_c2o)
        
        return lmk
    