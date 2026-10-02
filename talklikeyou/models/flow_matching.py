from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def identity(tensor, *args, **kwargs):
    return tensor


def extract(values, timesteps, x_shape):
    batch_size, *_ = timesteps.shape
    out = values.gather(-1, timesteps)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
    cosine_s=8e-3,
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start ** 0.5,
                linear_end ** 0.5,
                n_timestep,
                dtype=torch.float64,
            )
            ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return betas.numpy()


class FlowMatchingProcess(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        n_timestep=1000,
        sampling_timesteps=50,
        schedule="cosine",
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=False,
        guidance_weight=2.0,
        cond_drop_prob=0.2,
        part_w_dict=None,
        use_last_frame_loss=False,
        flow_matching=False,
        use_pva_loss=True,
        pva_xstart_only=False,
        flow_loss_weight=1.0,
        flow_predict_xstart=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.cond_drop_prob = cond_drop_prob
        self.n_timestep = int(n_timestep)
        self.sampling_timesteps = int(sampling_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.guidance_weight = guidance_weight
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss
        self.use_last_frame_loss = use_last_frame_loss
        self.part_w_dict = part_w_dict or {"motion": (0, -1, 1.0)}
        self.flow_matching = flow_matching
        self.use_pva_loss = use_pva_loss
        self.pva_xstart_only = pva_xstart_only
        self.flow_loss_weight = flow_loss_weight
        self.flow_predict_xstart = flow_predict_xstart
        if self.flow_predict_xstart and not self.flow_matching:
            raise ValueError("flow_predict_xstart requires flow_matching=True")
        if self.flow_predict_xstart and self.predict_epsilon:
            raise ValueError("flow_predict_xstart is incompatible with predict_epsilon")
        if self.flow_predict_xstart and not self.use_pva_loss:
            raise ValueError("flow_predict_xstart requires pva_loss supervision")

        betas = torch.Tensor(make_beta_schedule(schedule=schedule, n_timestep=n_timestep))
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        return noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - x0
            )
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def model_predictions(
        self,
        x,
        cond_frame,
        cond,
        habit_one_hot,
        t,
        weight=None,
        clip_x_start=False,
        ref_habit=None,
        habit_emb=None,
    ):
        weight = self.guidance_weight if weight is None else weight
        model_output = self.model.guided_forward(
            x,
            cond_frame,
            cond,
            habit_one_hot,
            t,
            weight,
            ref_habit=ref_habit,
            habit_emb=habit_emb,
        )
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        if self.predict_epsilon:
            pred_noise = model_output
            x_start = maybe_clip(self.predict_start_from_noise(x, t, pred_noise))
        else:
            x_start = maybe_clip(model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def _scale_time(self, t):
        max_t = max(self.n_timestep - 1, 1)
        return t.float() / max_t

    def _expand_time(self, t, x_shape):
        return t.view(-1, *((1,) * (len(x_shape) - 1)))

    def _flow_match_xt(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t_unit = self._scale_time(t)
        t_view = self._expand_time(t_unit, x_start.shape)
        x_t = (1.0 - t_view) * x_start + t_view * noise
        target_velocity = noise - x_start
        return x_t, target_velocity, t_unit

    def _flow_model_predicts_velocity(self):
        return self.flow_matching and not self.flow_predict_xstart

    def _flow_output_to_xstart(self, x_t, model_output, t_unit):
        if self._flow_model_predicts_velocity():
            return x_t - self._expand_time(t_unit, x_t.shape) * model_output
        return model_output

    def _flow_output_to_velocity(self, x_t, model_output, t_unit):
        if self._flow_model_predicts_velocity():
            return model_output
        pred_x_start = self._flow_output_to_xstart(x_t, model_output, t_unit)
        t_view = self._expand_time(t_unit, x_t.shape).clamp_min(1e-3)
        return (x_t - pred_x_start) / t_view

    @torch.no_grad()
    def ddim_sample(self, shape, cond_frame, cond, habit_one_hot, noise=None, ref_habit=None, habit_emb=None):
        if self.flow_matching:
            return self.flow_sample(
                shape,
                cond_frame,
                cond,
                habit_one_hot,
                noise=noise,
                ref_habit=ref_habit,
                habit_emb=habit_emb,
            )
        batch = shape[0]
        device = self.betas.device
        total_timesteps = self.n_timestep
        sampling_timesteps = min(self.sampling_timesteps, total_timesteps)
        eta = 1.0

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = noise.to(device) if noise is not None else torch.randn(shape, device=device)
        cond_frame = cond_frame.to(device)
        cond = cond.to(device)
        habit_one_hot = habit_one_hot.to(device)
        if ref_habit is not None:
            ref_habit = ref_habit.to(device)
        if habit_emb is not None:
            habit_emb = habit_emb.to(device)

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x,
                cond_frame,
                cond,
                habit_one_hot,
                time_cond,
                clip_x_start=self.clip_denoised,
                ref_habit=ref_habit,
                habit_emb=habit_emb,
            )

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            coeff = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + coeff * pred_noise + sigma * noise

        return x

    @torch.no_grad()
    def flow_sample(self, shape, cond_frame, cond, habit_one_hot, noise=None, ref_habit=None, habit_emb=None):
        batch = shape[0]
        device = self.betas.device
        sampling_timesteps = max(1, min(self.sampling_timesteps, self.n_timestep))
        times = torch.linspace(1.0, 0.0, steps=sampling_timesteps + 1, device=device)

        x = noise.to(device) if noise is not None else torch.randn(shape, device=device)
        cond_frame = cond_frame.to(device)
        cond = cond.to(device)
        habit_one_hot = habit_one_hot.to(device)
        if ref_habit is not None:
            ref_habit = ref_habit.to(device)
        if habit_emb is not None:
            habit_emb = habit_emb.to(device)

        for time, time_next in zip(times[:-1], times[1:]):
            time_cond = torch.full(
                (batch,),
                float(time * max(self.n_timestep - 1, 1)),
                device=device,
                dtype=x.dtype,
            )
            t_unit = torch.full((batch,), float(time), device=device, dtype=x.dtype)
            model_output = self.model.guided_forward(
                x,
                cond_frame,
                cond,
                habit_one_hot,
                time_cond,
                self.guidance_weight,
                ref_habit=ref_habit,
                habit_emb=habit_emb,
            )
            pred_velocity = self._flow_output_to_velocity(x, model_output, t_unit)
            dt = float(time_next - time)
            x = x + dt * pred_velocity

        return x

    def _get_pva_loss(self, pred, gt, cond_frame):
        loss_dict = {}

        for name, (start, end, weight) in self.part_w_dict.items():
            if start <= 0:
                start = 0
            if end <= 0:
                end = gt.shape[-1]

            pred_part = pred[..., start:end]
            gt_part = gt[..., start:end]

            vel_pred = pred_part[:, 1:] - pred_part[:, :-1]
            vel_gt = gt_part[:, 1:] - gt_part[:, :-1]
            acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
            acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]

            p_loss = self.loss_fn(pred_part, gt_part, reduction="none").mean() * weight

            loss_dict[f"{name}_P"] = p_loss
            if not self.pva_xstart_only:
                v_loss = self.loss_fn(vel_pred, vel_gt, reduction="none").mean() * weight
                a_loss = self.loss_fn(acc_pred, acc_gt, reduction="none").mean() * weight
                loss_dict[f"{name}_V"] = v_loss
                loss_dict[f"{name}_A"] = a_loss

            if self.use_last_frame_loss:
                cond_target = cond_frame[..., start:end][:, None]
                l_loss = self.loss_fn(pred_part[:, 0:1], cond_target, reduction="none").mean() * weight
                loss_dict[f"{name}_L"] = l_loss

        return loss_dict

    def p_losses(self, x_start, cond_frame, cond, habit_one_hot, t, ref_habit=None, habit_emb=None):
        if self.flow_matching:
            return self.flow_matching_losses(
                x_start,
                cond_frame,
                cond,
                habit_one_hot,
                t,
                ref_habit=ref_habit,
                habit_emb=habit_emb,
            )
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(
            x_noisy,
            cond_frame,
            cond,
            habit_one_hot,
            t,
            cond_drop_prob=self.cond_drop_prob,
            ref_habit=ref_habit,
            habit_emb=habit_emb,
        )
        target = noise if self.predict_epsilon else x_start
        loss_dict = self._get_pva_loss(x_recon, target, cond_frame)
        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict

    def flow_matching_losses(self, x_start, cond_frame, cond, habit_one_hot, t, ref_habit=None, habit_emb=None):
        x_t, target_velocity, t_unit = self._flow_match_xt(x_start, t)
        time_cond = t_unit * max(self.n_timestep - 1, 1)
        model_output = self.model(
            x_t,
            cond_frame,
            cond,
            habit_one_hot,
            time_cond,
            cond_drop_prob=self.cond_drop_prob,
            ref_habit=ref_habit,
            habit_emb=habit_emb,
        )
        if self.flow_predict_xstart:
            pred_x_start = model_output
            loss_dict = self._get_pva_loss(pred_x_start, x_start, cond_frame)
        else:
            pred_velocity = model_output
            loss_dict = {
                "flow_velocity": self.loss_fn(pred_velocity, target_velocity, reduction="none").mean()
                * self.flow_loss_weight
            }
            if self.use_pva_loss:
                pred_x_start = self._flow_output_to_xstart(x_t, pred_velocity, t_unit)
                loss_dict.update(self._get_pva_loss(pred_x_start, x_start, cond_frame))
        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict

    def loss(self, x, cond_frame, cond, habit_one_hot, t_override=None, ref_habit=None, habit_emb=None):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond_frame, cond, habit_one_hot, t, ref_habit=ref_habit, habit_emb=habit_emb)

    def forward(self, x, cond_frame, cond, habit_one_hot, t_override=None, ref_habit=None, habit_emb=None):
        return self.loss(
            x,
            cond_frame,
            cond,
            habit_one_hot,
            t_override=t_override,
            ref_habit=ref_habit,
            habit_emb=habit_emb,
        )
