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
        diffusion_steps=1000,
        sampling_timesteps=50,
        guidance_weight=2.0,
        cond_drop_prob=0.2,
        checkpoint="",
        device="cuda",
        use_last_frame_loss=False,
        part_w_dict=None,
        flow_matching=False,
        predict_epsilon=False,
        use_pva_loss=True,
        pva_xstart_only=False,
        flow_loss_weight=1.0,
        flow_predict_xstart=False,
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
            horizon=seq_frames,
            repr_dim=motion_feat_dim,
            n_timestep=diffusion_steps,
            sampling_timesteps=sampling_timesteps,
            schedule="cosine",
            loss_type="l2",
            clip_denoised=False,
            predict_epsilon=predict_epsilon,
            guidance_weight=guidance_weight,
            cond_drop_prob=cond_drop_prob,
            part_w_dict=part_w_dict,
            use_last_frame_loss=use_last_frame_loss,
            flow_matching=flow_matching,
            use_pva_loss=use_pva_loss,
            pva_xstart_only=pva_xstart_only,
            flow_loss_weight=flow_loss_weight,
            flow_predict_xstart=flow_predict_xstart,
        )

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state_dict"], strict=True)

        self.model = model
        self.flow_process = flow_process.to(device)

    def train(self):
        self.flow_process.train()

    def eval(self):
        self.flow_process.eval()

    @torch.no_grad()
    def sample(self, kp_cond, aud_cond, habit_one_hot, noise=None, ref_habit=None, habit_emb=None):
        batch_size, seq_len, _ = aud_cond.shape
        shape = (batch_size, seq_len, self.motion_feat_dim)
        return self.flow_process.ddim_sample(
            shape,
            kp_cond,
            aud_cond,
            habit_one_hot,
            noise=noise,
            ref_habit=ref_habit,
            habit_emb=habit_emb,
        )
