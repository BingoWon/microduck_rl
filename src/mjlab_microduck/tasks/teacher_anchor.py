"""Keep PPO close to a frozen teacher outside the dynamic jump phases."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from mjlab.rl import RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel


@dataclass
class PpoWithTeacherAnchorCfg(RslRlPpoAlgorithmCfg):
    teacher_anchor_coeff: float = 0.5
    teacher_anchor_phase0_only: bool = True
    symmetry_cfg: dict | None = None


class TeacherAnchoredMLPModel(MLPModel):
    """Inject the gradient of a teacher-action MSE into PPO's actor update."""

    def set_teacher(
        self,
        teacher: MLPModel,
        coeff: float,
        phase0_only: bool,
    ) -> None:
        object.__setattr__(self, "_anchor_teacher", teacher)
        self._anchor_coeff = coeff
        self._anchor_phase0_only = phase0_only

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        output = super().forward(obs, masks, hidden_state, stochastic_output)
        if stochastic_output and torch.is_grad_enabled():
            self._anchor_obs = obs
        return output

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        teacher = getattr(self, "_anchor_teacher", None)
        obs = getattr(self, "_anchor_obs", None)
        if teacher is not None and obs is not None and torch.is_grad_enabled():
            with torch.no_grad():
                target = teacher(obs)
            mean = self.output_mean
            mask = torch.ones(mean.shape[0], dtype=torch.bool, device=mean.device)
            if self._anchor_phase0_only:
                actor_obs = obs[self.obs_groups[0]]
                mask = actor_obs[:, 50].abs() < 0.5
            count = mask.sum()
            if count:
                delta = mean.detach() - target
                anchor_loss = delta[mask].square().mean()
                anchor_grad = (
                    2.0
                    * self._anchor_coeff
                    * delta
                    * mask.unsqueeze(-1)
                    / (count * mean.shape[-1])
                )
                mean.register_hook(lambda grad: grad + anchor_grad)
                self._anchor_losses.append(anchor_loss)
        self._anchor_obs = None
        return super().get_output_log_prob(outputs)


class TeacherAnchoredPPO(PPO):
    def __init__(
        self,
        *args,
        teacher_anchor_coeff: float = 0.5,
        teacher_anchor_phase0_only: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.actor, TeacherAnchoredMLPModel):
            raise TypeError("TeacherAnchoredPPO requires TeacherAnchoredMLPModel")
        self.teacher_anchor_coeff = teacher_anchor_coeff
        self.teacher_anchor_phase0_only = teacher_anchor_phase0_only
        self._teacher = None
        self.actor._anchor_losses = []

    def set_teacher_from_actor(self) -> None:
        if self._teacher is not None:
            return
        self._install_teacher(copy.deepcopy(self.actor).state_dict())

    def _install_teacher(self, state_dict: dict) -> None:
        teacher = copy.deepcopy(self.actor)
        object.__setattr__(teacher, "_anchor_teacher", None)
        teacher.load_state_dict(state_dict)
        teacher.requires_grad_(False)
        teacher.eval()
        self._teacher = teacher
        self.actor.set_teacher(
            teacher,
            self.teacher_anchor_coeff,
            self.teacher_anchor_phase0_only,
        )

    def update(self) -> dict[str, float]:
        self.actor._anchor_losses.clear()
        losses = super().update()
        if self.actor._anchor_losses:
            losses["teacher_anchor"] = torch.stack(
                self.actor._anchor_losses
            ).mean().item()
        return losses

    def save(self) -> dict:
        saved = super().save()
        if self._teacher is not None:
            saved["teacher_actor_state_dict"] = self._teacher.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        teacher_state = loaded_dict.get("teacher_actor_state_dict")
        if teacher_state is not None:
            self._install_teacher(teacher_state)
        return load_iteration
