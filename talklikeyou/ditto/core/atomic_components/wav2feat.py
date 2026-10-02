import librosa
import numpy as np
import math

from ..aux_models.hubert_stream import HubertStreaming

"""
wavlm_cfg = {
    "model_path": "",
    "device": "cuda",
    "force_ori_type": False,
}
hubert_cfg = {
    "model_path": "",
    "device": "cuda",
    "force_ori_type": False,
}
"""


class Wav2Feat:
    def __init__(self, w2f_cfg, w2f_type="hubert"):
        self.w2f_type = w2f_type.lower()
        if self.w2f_type == "hubert":
            self.w2f = Wav2FeatHubert(hubert_cfg=w2f_cfg)
            self.feat_dim = 1024
            self.support_streaming = True
        else:
            raise ValueError(f"Unsupported w2f_type: {w2f_type}")
        
    def __call__(
        self, 
        audio, 
        sr=16000, 
        norm_mean_std=None,   # for s2g
        chunksize=(3, 5, 2),   # for hubert
    ):
        if self.w2f_type == "hubert":
            feat = self.w2f(audio, chunksize=chunksize)
        elif self.w2f_type == "s2g":
            feat = self.w2f(audio, sr=sr, norm_mean_std=norm_mean_std)
        else:
            raise ValueError(f"Unsupported w2f_type: {self.w2f_type}")
        return feat
    
    def wav2feat(
        self,
        audio, 
        sr=16000, 
        norm_mean_std=None,   # for s2g
        chunksize=(3, 5, 2),
    ):
        # for offline
        if self.w2f_type == "hubert":
            feat = self.w2f.wav2feat(audio, sr=sr, chunksize=chunksize)
        elif self.w2f_type == "s2g":
            feat = self.w2f(audio, sr=sr, norm_mean_std=norm_mean_std)
        else:
            raise ValueError(f"Unsupported w2f_type: {self.w2f_type}")
        return feat
    

class Wav2FeatHubert:
    def __init__(
        self,
        hubert_cfg,
    ):
        self.hubert = HubertStreaming(**hubert_cfg)

    def __call__(self, audio_chunk, chunksize=(3, 5, 2)):
        """
        audio_chunk: int(sum(chunksize) * 0.04 * 16000) + 80    # 6480
        """
        valid_feat_s = - sum(chunksize[1:]) * 2   # -7
        valid_feat_e = - chunksize[2] * 2   # -2

        encoding_chunk = self.hubert(audio_chunk)
        valid_encoding = encoding_chunk[valid_feat_s:valid_feat_e]
        valid_feat = valid_encoding.reshape(chunksize[1], 2, 1024).mean(1)    # [5, 1024]
        return valid_feat

    def wav2feat(self, audio, sr, chunksize=(3, 5, 2)):
        # for offline
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        num_f = math.ceil(len(audio_16k) / 16000 * 25)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80    # 6480

        speech_pad = np.concatenate([
            np.zeros((split_len - int(sum(chunksize[1:]) * 0.04 * 16000),), dtype=audio_16k.dtype),
            audio_16k,
            np.zeros((split_len,), dtype=audio_16k.dtype),
        ], 0)
        
        i = 0
        res_lst = []
        while i < num_f:
            sss = int(i * 0.04 * 16000)
            eee = sss + split_len
            audio_chunk = speech_pad[sss:eee]
            valid_feat = self.__call__(audio_chunk, chunksize)
            res_lst.append(valid_feat)
            i += chunksize[1]
        
        ret = np.concatenate(res_lst, 0)
        ret = ret[:num_f]
        return ret
