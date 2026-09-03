#!/usr/bin/env python3
"""Interactive kinematic pose editor for the Microduck."""

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction

from mjlab_microduck.tasks import mdp

TASK = "Mjlab-SingleLegJump-Flat-MicroDuck"
REPO = Path(__file__).resolve().parents[2]
STATE_BANK = (
    REPO / "src/mjlab_microduck/tasks/data/single_leg_jump_reset_states.json"
)
SAVE_DIR = REPO / "pose-lab-saves"
JOINT_NAMES = (
    "左髋偏航",
    "左髋侧摆",
    "左髋俯仰",
    "左膝",
    "左踝",
    "颈部俯仰",
    "头部俯仰",
    "头部偏航",
    "头部侧摆",
    "右髋偏航",
    "右髋侧摆",
    "右髋俯仰",
    "右膝",
    "右踝",
)
JOINT_LIMITS_DEG = (
    (-25.0, 30.0),
    (-22.0, 22.0),
    (-90.0, 90.0),
    (-90.0, 90.0),
    (-90.0, 90.0),
    (-90.0, 60.0),
    (-90.0, 90.0),
    (-170.0, 170.0),
    (-25.0, 25.0),
    (-30.0, 25.0),
    (-22.0, 22.0),
    (-90.0, 90.0),
    (-90.0, 90.0),
    (-90.0, 90.0),
)
MIRROR_PERM = (9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4)
MIRROR_SIGN = (-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1)


class ZeroPolicy:
    def __init__(self, action_dim: int):
        self.action = torch.zeros(1, action_dim)

    def __call__(self, _obs) -> torch.Tensor:
        return self.action


def mean_joint_pos(states: list[dict]) -> torch.Tensor:
    return torch.tensor(
        [state["joint_pos"] for state in states],
        dtype=torch.float32,
    ).mean(dim=0)


class PoseController:
    def __init__(self, env: ManagerBasedRlEnv):
        payload = json.loads(STATE_BANK.read_text(encoding="utf-8"))["states"]
        self.env = env
        self.robot = env.scene["robot"]
        self.env_ids = torch.tensor([0], device=env.device)
        self.servo_ids = mdp._servo_joint_ids(env, self.robot)
        self.references = {}
        self.deltas = {}
        for side in ("left", "right"):
            standing_states = payload[side]["standing"]
            standing_mean = mean_joint_pos(standing_states)
            compressed_mean = mean_joint_pos(payload[side]["compressed"])
            reference = min(
                standing_states,
                key=lambda state: torch.square(
                    torch.tensor(state["joint_pos"]) - standing_mean
                ).sum(),
            )
            self.references[side] = reference
            self.deltas[side] = compressed_mean - standing_mean

        self.joint_pos = self.robot.data.joint_pos.clone()
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.root_pose = torch.zeros(1, 7, device=env.device)
        self.anchor_side = "left"
        self.anchor_position = torch.zeros(3, device=env.device)
        self.reference_root_z = 0.0
        self.current_q = torch.zeros(14, device=env.device)
        self.set_preset("left", 0.0)

    def _site_id(self, side: str) -> int:
        return self.robot.find_sites(f"{side}_foot")[0][0]

    def _write(self) -> None:
        self.joint_pos[:, self.servo_ids] = self.current_q
        self.robot.write_joint_state_to_sim(
            self.joint_pos,
            self.joint_vel,
            env_ids=self.env_ids,
        )
        self.robot.write_root_link_pose_to_sim(
            self.root_pose,
            env_ids=self.env_ids,
        )
        self.env.sim.forward()
        self.env.scene.update(dt=0.0)
        if self.anchor_side != "free":
            site_id = self._site_id(self.anchor_side)
            self.root_pose[:, :3] += (
                self.anchor_position - self.robot.data.site_pos_w[0, site_id]
            ).unsqueeze(0)
            self.robot.write_root_link_pose_to_sim(
                self.root_pose,
                env_ids=self.env_ids,
            )
            self.env.sim.forward()
            self.env.scene.update(dt=0.0)

    def set_preset(self, side: str, crouch_factor: float) -> None:
        reference = self.references[side]
        self.current_q = torch.tensor(
            reference["joint_pos"],
            device=self.env.device,
        )
        support_ids = (2, 3, 4) if side == "left" else (11, 12, 13)
        self.current_q[list(support_ids)] += (
            crouch_factor * self.deltas[side][list(support_ids)].to(self.env.device)
        )
        self.root_pose = torch.tensor(
            [[*reference["root_pos"], *reference["root_quat"]]],
            device=self.env.device,
        )
        self.root_pose[:, :3] += self.env.scene.terrain.env_origins[self.env_ids]
        self.anchor_side = side
        self.robot.write_root_link_pose_to_sim(
            self.root_pose,
            env_ids=self.env_ids,
        )
        self.joint_pos[:, self.servo_ids] = self.current_q
        self.robot.write_joint_state_to_sim(
            self.joint_pos,
            self.joint_vel,
            env_ids=self.env_ids,
        )
        self.env.sim.forward()
        self.env.scene.update(dt=0.0)
        self.anchor_position = self.robot.data.site_pos_w[
            0, self._site_id(side)
        ].clone()
        self.reference_root_z = float(self.robot.data.root_link_pos_w[0, 2])
        self._write()

    def set_joint_deg(self, index: int, value: float) -> None:
        self.current_q[index] = math.radians(value)
        self._write()

    def set_anchor(self, side: str) -> None:
        self.anchor_side = side
        if side != "free":
            self.anchor_position = self.robot.data.site_pos_w[
                0, self._site_id(side)
            ].clone()
        self._write()

    def mirror(self) -> None:
        source = self.current_q.clone()
        self.current_q = torch.stack(
            [
                source[MIRROR_PERM[i]] * MIRROR_SIGN[i]
                for i in range(14)
            ],
        )
        self.anchor_side = (
            "right"
            if self.anchor_side == "left"
            else "left"
            if self.anchor_side == "right"
            else "free"
        )
        if self.anchor_side != "free":
            self.anchor_position = self.robot.data.site_pos_w[
                0, self._site_id(self.anchor_side)
            ].clone()
        self._write()

    def metrics(self) -> dict:
        left = self.robot.data.site_pos_w[0, self._site_id("left")]
        right = self.robot.data.site_pos_w[0, self._site_id("right")]
        return {
            "root_drop_mm": (
                self.reference_root_z
                - float(self.robot.data.root_link_pos_w[0, 2])
            )
            * 1000.0,
            "foot_height_delta_mm": float((right[2] - left[2]) * 1000.0),
        }

    def save(self) -> Path:
        SAVE_DIR.mkdir(exist_ok=True)
        path = SAVE_DIR / (
            f"pose-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "joint_pos": self.current_q.cpu().tolist(),
                    "root_pose": self.root_pose[0].cpu().tolist(),
                    "anchor_side": self.anchor_side,
                    **self.metrics(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


class PoseLabViewer(ViserPlayViewer):
    def __init__(self, *args, controller: PoseController, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = controller
        self.joint_handles = []
        self.status = None
        self.updating_handles = False

    def setup(self) -> None:
        super().setup()
        self._pause_button.disabled = True
        self._step_button.disabled = True
        with self._server.gui.add_folder("姿态实验室"):
            preset = self._server.gui.add_button_group(
                "预设姿态",
                (
                    "左腿站立",
                    "左腿下蹲",
                    "右腿站立",
                    "右腿下蹲",
                    "镜像当前",
                ),
            )
            crouch = self._server.gui.add_slider(
                "下蹲进度",
                min=0.0,
                max=3.0,
                step=0.01,
                initial_value=0.0,
                hint="0为站立，2为深蹲参考，可连续拖动",
            )
            anchor = self._server.gui.add_dropdown(
                "固定位置",
                ("左脚", "右脚", "自由"),
                initial_value="左脚",
            )
            save = self._server.gui.add_button(
                "保存当前姿态",
                icon=viser.Icon.DEVICE_FLOPPY,
            )
            self.status = self._server.gui.add_html("")

        for title, indices in (
            ("左腿关节", range(5)),
            ("头部关节", range(5, 9)),
            ("右腿关节", range(9, 14)),
        ):
            with self._server.gui.add_folder(title):
                for index in indices:
                    lo, hi = JOINT_LIMITS_DEG[index]
                    initial = math.degrees(
                        float(self.controller.current_q[index])
                    )
                    handle = self._server.gui.add_slider(
                        JOINT_NAMES[index],
                        min=lo,
                        max=hi,
                        step=0.1,
                        initial_value=min(max(initial, lo), hi),
                    )

                    @handle.on_update
                    def _joint(event, joint_index=index) -> None:
                        if not self.updating_handles:
                            self.request_action(
                                "CUSTOM",
                                ("joint", joint_index, event.target.value),
                            )

                    self.joint_handles.append(handle)

        @preset.on_click
        def _preset(event) -> None:
            mapping = {
                "左腿站立": ("preset", "left", 0.0),
                "左腿下蹲": ("preset", "left", 2.0),
                "右腿站立": ("preset", "right", 0.0),
                "右腿下蹲": ("preset", "right", 2.0),
                "镜像当前": ("mirror",),
            }
            self.request_action("CUSTOM", mapping[event.target.value])

        @crouch.on_update
        def _crouch(event) -> None:
            side = (
                self.controller.anchor_side
                if self.controller.anchor_side in ("left", "right")
                else "left"
            )
            self.request_action(
                "CUSTOM",
                ("preset", side, event.target.value),
            )

        @anchor.on_update
        def _anchor(event) -> None:
            value = {"左脚": "left", "右脚": "right", "自由": "free"}[
                event.target.value
            ]
            self.request_action("CUSTOM", ("anchor", value))

        @save.on_click
        def _save(_) -> None:
            self.request_action("CUSTOM", ("save",))

        self._sync_controls()
        self._scene.request_update()

    def _sync_controls(self) -> None:
        self.updating_handles = True
        for index, handle in enumerate(self.joint_handles):
            handle.value = math.degrees(float(self.controller.current_q[index]))
        self.updating_handles = False
        metrics = self.controller.metrics()
        anchor = {
            "left": "左脚",
            "right": "右脚",
            "free": "自由",
        }[self.controller.anchor_side]
        if self.status is not None:
            self.status.content = (
                f"<b>物理已暂停</b><br>"
                f"固定：{anchor}<br>"
                f"身体下降：{metrics['root_drop_mm']:.1f} mm<br>"
                f"右脚相对左脚高度：{metrics['foot_height_delta_mm']:.1f} mm"
            )

    def _handle_custom_action(self, action, payload) -> bool:
        if action is not ViewerAction.CUSTOM or not isinstance(payload, tuple):
            return super()._handle_custom_action(action, payload)
        kind = payload[0]
        if kind == "joint":
            self.controller.set_joint_deg(payload[1], payload[2])
        elif kind == "preset":
            self.controller.set_preset(payload[1], payload[2])
        elif kind == "anchor":
            self.controller.set_anchor(payload[1])
        elif kind == "mirror":
            self.controller.mirror()
        elif kind == "save":
            path = self.controller.save()
            if self.status is not None:
                self.status.content = f"<b>已保存</b><br>{path}"
            return True
        else:
            return False
        self._sync_controls()
        self._scene.request_update()
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 3600.0
    cfg.events.pop("push_robot", None)
    for name in tuple(cfg.terminations):
        cfg.terminations.pop(name)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    controller = PoseController(env)
    wrapped = RslRlVecEnvWrapper(env)
    server = viser.ViserServer(port=args.port, label="Microduck 姿态实验室")
    viewer = PoseLabViewer(
        wrapped,
        ZeroPolicy(wrapped.num_actions),
        viser_server=server,
        controller=controller,
    )
    viewer.pause()
    viewer.run()
    wrapped.close()


if __name__ == "__main__":
    main()
