import torch

from .flow_matching import FlowMatchingProcess
from .flow_matching_motion_transformer import FlowMatchingMotionTransformer


class FlowMatchingMotionGenerator:
    def __init__(
        self,
        motion_feat_dim,
        person_num,
        audio_feat_dim=512,
        seq_frames=25,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        time_steps=1000,
        sampling_steps=1,
        guidance_scale=1.3,
        checkpoint="",
        device="cuda",
        predict_clean_motion=True,
    ):
        self.motion_feat_dim = motion_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.seq_frames = seq_frames
        self.person_num = person_num
        self.device = device

        model = FlowMatchingMotionTransformer(
            nfeats=motion_feat_dim,
            person_num=person_num,
            seq_len=seq_frames,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_feature_dim=audio_feat_dim,
        )
        flow_process = FlowMatchingProcess(
            model=model,
            time_steps=time_steps,
            sampling_steps=sampling_steps,
            guidance_scale=guidance_scale,
            predict_clean_motion=predict_clean_motion,
        )

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state_dict"], strict=True)

        self.model = model
        self.flow_process = flow_process.to(device)

    def eval(self):
        self.flow_process.eval()

    @torch.no_grad()
    def sample(self, kp_cond, aud_cond, habit_one_hot, noise=None, ref_habit=None, habit_emb=None):
        batch_size, seq_len, _ = aud_cond.shape
        shape = (batch_size, seq_len, self.motion_feat_dim)
        return self.flow_process.sample(
            shape,
            kp_cond,
            aud_cond,
            habit_one_hot,
            noise=noise,
            reference_habit=ref_habit,
            habit_embedding=habit_emb,
        )
