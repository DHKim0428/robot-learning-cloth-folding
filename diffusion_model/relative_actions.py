"""Relative-action helpers for custom DiffusionPolicy experiments.

This module intentionally lives outside LeRobot's native diffusion workflow.
It lets the custom trainer use arm-joint deltas while keeping the gripper as an
absolute open/close command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

# This module may be imported both as `relative_actions` by local scripts and as
# `diffusion_model.relative_actions` by tests/tools. LeRobot's registry is
# process-global, so make registration idempotent for that case.
ProcessorStepRegistry.unregister("diffusion_relative_actions")
ProcessorStepRegistry.unregister("diffusion_absolute_actions")


def current_state(state: Tensor) -> Tensor:
    """Return latest/current state as `(B, D)` from `(D)`, `(B, D)`, or `(B, T, D)`."""
    if state.ndim == 1:
        return state.unsqueeze(0)
    if state.ndim == 2:
        return state
    if state.ndim >= 3:
        return state[:, -1, :]
    raise ValueError(f"Unsupported observation.state shape: {tuple(state.shape)}")


def build_relative_mask(
    action_dim: int,
    action_names: list[str] | None,
    exclude_joints: list[str],
) -> list[bool]:
    """Return True for action dimensions that should become relative."""
    if not exclude_joints or action_names is None:
        return [True] * action_dim

    exclude_tokens = [str(name).lower() for name in exclude_joints if name]
    if not exclude_tokens:
        return [True] * action_dim

    mask: list[bool] = []
    for name in action_names[:action_dim]:
        action_name = str(name).lower()
        is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
        mask.append(not is_excluded)
    if len(mask) < action_dim:
        mask.extend([True] * (action_dim - len(mask)))
    return mask


def to_relative_actions(actions: Tensor, state: Tensor, mask: list[bool]) -> Tensor:
    """Convert absolute actions to deltas relative to the latest observation state."""
    state_ref = current_state(state).to(device=actions.device, dtype=actions.dtype)
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    offset = state_ref[..., :dims] * mask_t
    if actions.ndim == 3:
        offset = offset.unsqueeze(1)
    out = actions.clone()
    out[..., :dims] -= offset
    return out


def to_absolute_actions(actions: Tensor, state: Tensor, mask: list[bool]) -> Tensor:
    """Convert relative action deltas back to absolute joint targets."""
    state_ref = current_state(state).to(device=actions.device, dtype=actions.dtype)
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    offset = state_ref[..., :dims] * mask_t
    if actions.ndim == 3:
        offset = offset.unsqueeze(1)
    out = actions.clone()
    out[..., :dims] += offset
    return out


@dataclass
@ProcessorStepRegistry.register("diffusion_relative_actions")
class DiffusionRelativeActionsStep(ProcessorStep):
    """Preprocessor step converting diffusion action chunks to relative actions."""

    enabled: bool = True
    exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_names: list[str] | None = None
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    def _mask(self, action_dim: int) -> list[bool]:
        return build_relative_mask(action_dim, self.action_names, self.exclude_joints)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if isinstance(observation, dict) else None
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        action = transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return transition

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = to_relative_actions(
            action,
            state,
            self._mask(action.shape[-1]),
        )
        return new_transition

    def get_cached_state(self) -> Tensor | None:
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("diffusion_absolute_actions")
class DiffusionAbsoluteActionsStep(ProcessorStep):
    """Postprocessor step converting relative diffusion predictions to absolute actions."""

    enabled: bool = True
    relative_step: DiffusionRelativeActionsStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None:
            raise RuntimeError("DiffusionAbsoluteActionsStep requires a paired relative_step.")

        state = self.relative_step.get_cached_state()
        if state is None:
            raise RuntimeError("No cached observation.state available for relative-action inversion.")

        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = to_absolute_actions(
            action,
            state,
            self.relative_step._mask(action.shape[-1]),
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def compute_relative_action_stats(
    dataset,
    *,
    action_delta_indices: list[int],
    exclude_joints: list[str],
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """Compute action stats after converting valid action chunks to relative space.

    The dataset files are not modified. Stats are computed from the loaded
    HuggingFace table view, so episode filters/sanitization already applied to
    `dataset` are respected.
    """
    del num_workers  # The current implementation is vectorized single-process.

    hf_dataset = dataset.hf_dataset
    features = dataset.features
    action_names = list(features[ACTION].get("names") or [])
    action_dim = int(features[ACTION]["shape"][0])
    mask = np.array(
        build_relative_mask(action_dim, action_names, exclude_joints),
        dtype=np.float32,
    )

    actions = np.asarray(hf_dataset[ACTION], dtype=np.float32)
    states = np.asarray(hf_dataset[OBS_STATE], dtype=np.float32)
    episode_indices = np.asarray(hf_dataset["episode_index"])
    absolute_indices = np.asarray(hf_dataset["index"])

    offsets = np.asarray(action_delta_indices, dtype=np.int64)
    valid_chunks: list[np.ndarray] = []
    batch_size = 8192

    for start in range(0, len(hf_dataset), batch_size):
        stop = min(start + batch_size, len(hf_dataset))
        base_rows = np.arange(start, stop, dtype=np.int64)
        target_rows = base_rows[:, None] + offsets[None, :]

        in_bounds = (target_rows >= 0) & (target_rows < len(hf_dataset))
        valid = in_bounds.all(axis=1)
        if not valid.any():
            continue

        candidate_rows = target_rows[valid]
        candidate_base = base_rows[valid]
        same_episode = (
            episode_indices[candidate_rows] == episode_indices[candidate_base][:, None]
        ).all(axis=1)
        contiguous_abs = (
            absolute_indices[candidate_rows]
            == absolute_indices[candidate_base][:, None] + offsets[None, :]
        ).all(axis=1)
        keep = same_episode & contiguous_abs
        if not keep.any():
            continue

        kept_rows = candidate_rows[keep]
        kept_base = candidate_base[keep]
        chunks = actions[kept_rows].copy()
        chunks[:, :, :action_dim] -= states[kept_base, None, :action_dim] * mask[None, None, :]
        valid_chunks.append(chunks.reshape(-1, action_dim))

    if not valid_chunks:
        raise RuntimeError(
            "No valid relative-action chunks found. Check action_delta_indices and episode data."
        )

    values = np.concatenate(valid_chunks, axis=0)
    return {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "q01": np.quantile(values, 0.01, axis=0),
        "q10": np.quantile(values, 0.10, axis=0),
        "q90": np.quantile(values, 0.90, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }
