from __future__ import annotations

import numpy as np
import torch


def kalman_smooth(
    sequence: np.ndarray,
    device: torch.device,
    observation_variance: float = 3e-7,
    process_variance: float = 1e-5,
) -> np.ndarray:
    values = torch.as_tensor(sequence, dtype=torch.float32, device=device)
    original_shape = values.shape
    values = values.reshape(values.shape[0], -1)
    frame_count, feature_dim = values.shape
    identity = torch.eye(feature_dim, device=device)
    process_covariance = process_variance * identity
    observation_covariance = observation_variance * identity

    filtered = torch.zeros_like(values)
    filtered_covariance = torch.zeros(
        (frame_count, feature_dim, feature_dim), device=device
    )
    predicted_state = values[0]
    predicted_covariance = identity

    for index in range(frame_count):
        if index:
            predicted_state = filtered[index - 1]
            predicted_covariance = filtered_covariance[index - 1] + process_covariance
        innovation = values[index] - predicted_state
        innovation_covariance = predicted_covariance + observation_covariance
        gain = torch.linalg.solve(
            innovation_covariance.T, predicted_covariance.T
        ).T
        filtered[index] = predicted_state + gain @ innovation
        filtered_covariance[index] = (identity - gain) @ predicted_covariance

    smoothed = torch.zeros_like(filtered)
    smoothed_covariance = torch.zeros_like(filtered_covariance)
    smoothed[-1] = filtered[-1]
    smoothed_covariance[-1] = filtered_covariance[-1]
    for index in range(frame_count - 2, -1, -1):
        next_covariance = filtered_covariance[index] + process_covariance
        gain = torch.linalg.solve(next_covariance.T, filtered_covariance[index].T).T
        smoothed[index] = filtered[index] + gain @ (smoothed[index + 1] - filtered[index])
        smoothed_covariance[index] = filtered_covariance[index] + gain @ (
            smoothed_covariance[index + 1] - next_covariance
        ) @ gain.T

    return smoothed.reshape(original_shape).cpu().numpy()
