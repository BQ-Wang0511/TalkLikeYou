import cv2
import numpy as np

from ..utils.load_model import load_model


class FaceMesh:
    def __init__(self, model_path, device="cuda"):
        self.model, self.model_type = load_model(model_path, device=device)
        self.input_size = (256, 256)  # (w, h)
        self.output_names = [
            "Identity",
            "Identity_1",
            "Identity_2",
        ]  # Identity is the mesh

    def project_landmarks(self, points, roi):
        width, height = self.input_size
        points /= (width, height, width)
        sin, cos = np.sin(roi[4]), np.cos(roi[4])
        matrix = np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [1.0, 1.0, 1.0]])
        points -= (0.5, 0.5, 0.0)
        rotated = np.matmul(points * (1, 1, 0), matrix)
        points *= (0, 0, 1)
        points += rotated
        points *= (roi[2], roi[3], roi[2])
        points += (roi[0], roi[1], 0.0)
        return points

    def __call__(self, image, roi):
        """
        image: np.ndarray, RGB, (H, W, C), [0, 255]
        roi: np.ndarray, (cx, cy, w, h, rotation), rotation in radian
        """
        cx, cy, w, h = roi[:4]
        w_half, h_half = w / 2, h / 2
        pts = [
            (cx - w_half, cy - h_half),
            (cx + w_half, cy - h_half),
            (cx + w_half, cy + h_half),
            (cx - w_half, cy + h_half),
        ]
        rotation = roi[4]
        s, c = np.sin(rotation), np.cos(rotation)
        t = np.array(pts) - (cx, cy)
        r = np.array([[c, s], [-s, c]])
        src_pts = np.matmul(t, r) + (cx, cy)
        src_pts = src_pts.astype(np.float32)

        dst_pts = np.array(
            [
                [0.0, 0.0],
                [self.input_size[0], 0.0],
                [self.input_size[0], self.input_size[1]],
                [0.0, self.input_size[1]],
            ]
        ).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        roi_image = cv2.warpPerspective(
            image, M, self.input_size, flags=cv2.INTER_LINEAR
        )
        roi_image = roi_image / 255.0
        roi_image = roi_image.astype(np.float32)
        roi_image = roi_image[np.newaxis, :, :, :]

        outputs = {}
        if self.model_type == "onnx":
            out_list = self.model.run(None, {"input": roi_image})
            for i, name in enumerate(self.output_names):
                outputs[name] = out_list[i]
        elif self.model_type == "tensorrt":
            self.model.setup({"input": roi_image})
            self.model.infer()
            for name in self.output_names:
                outputs[name] = self.model.buffer[name][0]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        points = outputs["Identity"].reshape(1434 // 3, 3)
        points = self.project_landmarks(points, roi)
        return points


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--image", type=str, help="image path")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    args = parser.parse_args()

    face_mesh = FaceMesh(args.model, args.device)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_mesh = FaceMesh(args.model, args.device)
    roi = np.array([128, 128, 256, 256, np.pi / 2])
    mesh = face_mesh(image, roi)
    print(mesh.shape)
