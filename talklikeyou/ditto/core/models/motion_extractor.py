import numpy as np
import torch
from ..utils.load_model import load_model


class MotionExtractor:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "MotionExtractor",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

        self.output_names = [
            "pitch",
            "yaw",
            "roll",
            "t",
            "exp",
            "scale",
            "kp",
        ]

    def __call__(self, image):
        """
        image: np.ndarray, shape (1, 3, 256, 256), RGB, 0-1
        """
        outputs = {}
        if self.model_type == "onnx":
            out_list = self.model.run(None, {"image": image})
            for i, name in enumerate(self.output_names):
                outputs[name] = out_list[i]
        elif self.model_type == "tensorrt":
            self.model.setup({"image": image})
            self.model.infer()
            for name in self.output_names:
                outputs[name] = self.model.buffer[name][0].copy()
        elif self.model_type == "pytorch":
            with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
                pred = self.model(torch.from_numpy(image).to(self.device))
                for i, name in enumerate(self.output_names):
                    outputs[name] = pred[i].float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        outputs["exp"] = outputs["exp"].reshape(1, -1)
        outputs["kp"] = outputs["kp"].reshape(1, -1)
        return outputs


