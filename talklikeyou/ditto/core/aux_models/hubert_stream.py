from ..utils.load_model import load_model


class HubertStreaming:
    def __init__(self, model_path, device="cuda", **kwargs):
        kwargs["model_file"] = model_path
        kwargs["module_name"] = "HubertStreamingONNX"
        kwargs["package_name"] = "..aux_models.modules"

        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def forward_chunk(self, audio_chunk):
        if self.model_type == "onnx":
            output = self.model.run(None, {"input_values": audio_chunk.reshape(1, -1)})[0]
        elif self.model_type == "tensorrt":
            self.model.setup({"input_values": audio_chunk.reshape(1, -1)})
            self.model.infer()
            output = self.model.buffer["encoding_out"][0]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return output
    
    def __call__(self, audio_chunk):
        if self.model_type == "ori":
            output = self.model.forward_chunk(audio_chunk)
        else:
            output = self.forward_chunk(audio_chunk)
        return output
