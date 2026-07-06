from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import yaml
from mjlab.utils.lab_api import math as math_utils

from instinct_mj.motion_reference.motion_reference_part2link_data import (
    Part2LinkMotionReferenceData,
    Part2LinkMotionReferenceState,
    Part2LinkMotionSequence,
)
from instinct_mj.motion_reference.utils import estimate_angular_velocity, estimate_velocity, pose_interpolate_bilinear

from .amass_motion import AmassMotion

if TYPE_CHECKING:
    from .part2link_motion_cfg import Part2LinkMotionCfg


class Part2LinkMotion(AmassMotion):
    """Processing G1 sitting Part2Link motion files with object and sparse contact-map data."""

    cfg: Part2LinkMotionCfg

    def _refresh_motion_file_list(self):
        if self.cfg.metadata_yaml is None:
            return super()._refresh_motion_file_list()

        with open(os.path.expanduser(self.cfg.metadata_yaml)) as f:
            yaml_data = yaml.safe_load(f)

        motion_files = yaml_data["motion_files"]
        self._all_motion_files = [os.path.join(self.cfg.path, item["motion_file"]) for item in motion_files]
        self._init_motion_weights(torch.tensor([float(item.get("weight", 1.0)) for item in motion_files]))
        print(f"[{self.__class__.__name__} Motion] {len(self._all_motion_files)} files found")

    def _read_retargetted_motion_file(self, filepath: str) -> Part2LinkMotionSequence:
        raw_data = np.load(filepath, mmap_mode="r", allow_pickle=True)
        framerate = raw_data["framerate"].item()
        joint_names = (
            raw_data["joint_names"] if isinstance(raw_data["joint_names"], list) else raw_data["joint_names"].tolist()
        )
        joint_pos = torch.as_tensor(raw_data["joint_pos"], device=self.buffer_device, dtype=torch.float)
        root_trans = torch.as_tensor(raw_data["base_pos_w"], device=self.buffer_device, dtype=torch.float)
        root_quat = torch.as_tensor(raw_data["base_quat_w"], device=self.buffer_device, dtype=torch.float)

        retargetted_joints_to_output_joints_ids = [joint_names.index(j_name) for j_name in self.sim_joint_names]
        joint_pos = joint_pos[:, retargetted_joints_to_output_joints_ids]

        object_pos_w = torch.as_tensor(
            raw_data[f"{self.cfg.object_key}_pos"], device=self.buffer_device, dtype=torch.float
        ).unsqueeze(1)
        object_quat_w = torch.as_tensor(
            raw_data[f"{self.cfg.object_key}_quat"], device=self.buffer_device, dtype=torch.float
        ).unsqueeze(1)
        object_validity = torch.ones((object_pos_w.shape[0], 1), device=self.buffer_device, dtype=torch.bool)

        sparse_contact_robot_body_names = raw_data["sparse_contact_robot_body_names"].tolist()
        sparse_contact_robot_interest_names = raw_data["sparse_contact_robot_interest_names"].tolist()
        sparse_contact_part_names = raw_data["sparse_contact_part_names"].tolist()

        sparse_link_pos_w = torch.as_tensor(
            raw_data["sparse_contact_link_pos_w"], device=self.buffer_device, dtype=torch.float
        )
        sparse_center_vector = torch.as_tensor(
            raw_data["sparse_contact_link_part_center_vector_w"], device=self.buffer_device, dtype=torch.float
        )
        sparse_link_part_proximity = torch.as_tensor(
            raw_data["sparse_contact_link_part_proximity"], device=self.buffer_device, dtype=torch.bool
        )
        sparse_point_proximity = torch.as_tensor(
            raw_data["sparse_contact_link_part_point_proximity"], device=self.buffer_device, dtype=torch.bool
        )
        sparse_part_points = torch.as_tensor(
            raw_data["sparse_contact_part_points_local"], device=self.buffer_device, dtype=torch.float
        )
        sparse_point_mask = torch.as_tensor(
            raw_data["sparse_contact_part_point_mask"], device=self.buffer_device, dtype=torch.bool
        )
        sparse_part_centers = torch.as_tensor(
            raw_data["sparse_contact_part_centers_local"], device=self.buffer_device, dtype=torch.float
        )

        if sparse_point_proximity.shape[-1] == 1 and sparse_part_points.shape[1] != 1:
            sparse_point_proximity = sparse_point_proximity.expand(
                *sparse_point_proximity.shape[:-1], sparse_part_points.shape[1]
            )

        sparse_relation = torch.as_tensor(
            raw_data["sparse_contact_relation_matrix"], device=self.buffer_device, dtype=torch.float
        )
        if sparse_relation.ndim == 2:
            sparse_relation = sparse_relation.unsqueeze(0).repeat(sparse_center_vector.shape[0], 1, 1)
        if sparse_part_points.ndim == 3:
            sparse_part_points = sparse_part_points.unsqueeze(0).repeat(sparse_center_vector.shape[0], 1, 1, 1)
        if sparse_point_mask.ndim == 2:
            sparse_point_mask = sparse_point_mask.unsqueeze(0).repeat(sparse_center_vector.shape[0], 1, 1)
        if sparse_part_centers.ndim == 2:
            sparse_part_centers = sparse_part_centers.unsqueeze(0).repeat(sparse_center_vector.shape[0], 1, 1)

        return self._pack_retargetted_motion_sequence(
            root_trans,
            root_quat,
            joint_pos,
            framerate,
            object_pos_w,
            object_quat_w,
            object_validity,
            sparse_link_pos_w,
            sparse_center_vector,
            sparse_relation,
            sparse_link_part_proximity,
            sparse_point_proximity,
            sparse_part_points,
            sparse_point_mask,
            sparse_part_centers,
            sparse_contact_robot_body_names,
            sparse_contact_robot_interest_names,
            sparse_contact_part_names,
        )

    def _nearest_resample_sparse(self, value: torch.Tensor, source_framerate: float, target_framerate: float) -> torch.Tensor:
        time_max = (value.shape[0] - 1) / source_framerate
        num_frames = np.ceil(time_max * target_framerate)
        frame_idx = torch.arange(num_frames, dtype=torch.float, device=value.device) / target_framerate * source_framerate
        frame_idx = frame_idx[frame_idx <= (value.shape[0] - 1)]
        nearest_idx = frame_idx.round().long().clamp(max=value.shape[0] - 1)
        return value[nearest_idx]

    def _pack_retargetted_motion_sequence(
        self,
        root_trans: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        framerate: float | int,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
        object_validity: torch.Tensor,
        sparse_link_pos_w: torch.Tensor,
        sparse_center_vector: torch.Tensor,
        sparse_relation: torch.Tensor,
        sparse_link_part_proximity: torch.Tensor,
        sparse_point_proximity: torch.Tensor,
        sparse_part_points: torch.Tensor,
        sparse_point_mask: torch.Tensor,
        sparse_part_centers: torch.Tensor,
        sparse_contact_robot_body_names: list[str],
        sparse_contact_robot_interest_names: list[str],
        sparse_contact_part_names: list[str],
    ) -> Part2LinkMotionSequence:
        if self.cfg.motion_interpolate_func:
            source_framerate = float(framerate)
            root_trans, root_quat, joint_pos = self.cfg.motion_interpolate_func(
                root_trans, root_quat, joint_pos, framerate, self.cfg.motion_target_framerate
            )
            if source_framerate != self.cfg.motion_target_framerate:
                object_pos_w, object_quat_w, object_validity = pose_interpolate_bilinear(
                    object_pos_w,
                    object_quat_w,
                    object_validity,
                    source_framerate,
                    self.cfg.motion_target_framerate,
                )
                sparse_center_vector = self._nearest_resample_sparse(
                    sparse_center_vector, source_framerate, self.cfg.motion_target_framerate
                )
                time_max = (sparse_link_pos_w.shape[0] - 1) / source_framerate
                num_frames = np.ceil(time_max * self.cfg.motion_target_framerate)
                frame_idx = (
                    torch.arange(num_frames, dtype=torch.float, device=sparse_link_pos_w.device)
                    / self.cfg.motion_target_framerate
                    * source_framerate
                )
                frame_idx = frame_idx[frame_idx <= (sparse_link_pos_w.shape[0] - 1)]
                front_frame_idx = frame_idx.floor().long()
                back_frame_idx = frame_idx.ceil().long()
                ratio = frame_idx - front_frame_idx.float()
                sparse_link_pos_w = (
                    ratio.view(-1, 1, 1) * sparse_link_pos_w[back_frame_idx]
                    + (1 - ratio).view(-1, 1, 1) * sparse_link_pos_w[front_frame_idx]
                )
                sparse_relation = self._nearest_resample_sparse(
                    sparse_relation, source_framerate, self.cfg.motion_target_framerate
                )
                sparse_link_part_proximity = self._nearest_resample_sparse(
                    sparse_link_part_proximity, source_framerate, self.cfg.motion_target_framerate
                )
                sparse_point_proximity = self._nearest_resample_sparse(
                    sparse_point_proximity, source_framerate, self.cfg.motion_target_framerate
                )
                sparse_part_points = self._nearest_resample_sparse(
                    sparse_part_points, source_framerate, self.cfg.motion_target_framerate
                )
                sparse_point_mask = self._nearest_resample_sparse(
                    sparse_point_mask, source_framerate, self.cfg.motion_target_framerate
                )
                sparse_part_centers = self._nearest_resample_sparse(
                    sparse_part_centers, source_framerate, self.cfg.motion_target_framerate
                )
            framerate = self.cfg.motion_target_framerate

        link_pos_quat_b = self.forward_kinematics_func(joint_pos)
        link_pos_b = link_pos_quat_b[..., :3]
        link_quat_b = link_pos_quat_b[..., 3:]

        # Part2Link contact maps anchor the root trajectory at their first sparse robot link.
        anchor_name = sparse_contact_robot_body_names[0]
        anchor_name_for_fk = anchor_name
        if anchor_name_for_fk not in self.link_of_interests and anchor_name == "pelvis_contour_link":
            anchor_name_for_fk = "pelvis"
        anchor_idx = self.link_of_interests.index(anchor_name_for_fk)
        root_trans = sparse_link_pos_w[:, 0] - math_utils.quat_apply(root_quat, link_pos_b[:, anchor_idx])

        if self.cfg.velocity_estimation_method is not None:
            joint_vel = estimate_velocity(
                joint_pos.unsqueeze(0), 1 / framerate, self.cfg.velocity_estimation_method
            ).squeeze(0)
            base_lin_vel_w = estimate_velocity(
                root_trans.unsqueeze(0), 1 / framerate, self.cfg.velocity_estimation_method
            ).squeeze(0)
            base_ang_vel_w = estimate_angular_velocity(
                root_quat.unsqueeze(0), 1 / framerate, self.cfg.velocity_estimation_method
            ).squeeze(0)
        else:
            joint_vel = torch.zeros_like(joint_pos)
            base_lin_vel_w = torch.zeros_like(root_trans)
            base_ang_vel_w = torch.zeros_like(root_trans)

        if self.cfg.object_velocity_estimation_method is not None:
            object_lin_vel_w = estimate_velocity(
                object_pos_w.permute(1, 0, 2), 1 / framerate, self.cfg.object_velocity_estimation_method
            ).permute(1, 0, 2)
            object_ang_vel_w = estimate_angular_velocity(
                object_quat_w.permute(1, 0, 2), 1 / framerate, self.cfg.object_velocity_estimation_method
            ).permute(1, 0, 2)
        else:
            object_lin_vel_w = torch.zeros_like(object_pos_w)
            object_ang_vel_w = torch.zeros_like(object_pos_w)

        link_pos_w, link_quat_w = math_utils.combine_frame_transforms(
            root_trans.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
            root_quat.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
            link_pos_b,
            link_quat_b,
        )

        if self.cfg.velocity_estimation_method is not None:
            link_lin_vel_b = estimate_velocity(
                link_pos_b.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            link_ang_vel_b = estimate_angular_velocity(
                link_quat_b.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            link_lin_vel_w = estimate_velocity(
                link_pos_w.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            link_ang_vel_w = estimate_angular_velocity(
                link_quat_w.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
        else:
            link_lin_vel_b = torch.zeros_like(link_pos_b)
            link_ang_vel_b = torch.zeros_like(link_pos_b)
            link_lin_vel_w = torch.zeros_like(link_pos_w)
            link_ang_vel_w = torch.zeros_like(link_pos_w)

        return Part2LinkMotionSequence(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            base_pos_w=root_trans,
            base_lin_vel_w=base_lin_vel_w,
            base_quat_w=root_quat,
            base_ang_vel_w=base_ang_vel_w,
            link_pos_b=link_pos_b,
            link_quat_b=link_quat_b,
            link_lin_vel_b=link_lin_vel_b,
            link_ang_vel_b=link_ang_vel_b,
            link_lin_vel_w=link_lin_vel_w,
            link_ang_vel_w=link_ang_vel_w,
            link_pos_w=link_pos_w,
            link_quat_w=link_quat_w,
            framerate=torch.as_tensor(framerate),
            buffer_length=torch.as_tensor(joint_pos.shape[0]),
            object_pos_w=object_pos_w,
            object_quat_w=object_quat_w,
            object_lin_vel_w=object_lin_vel_w,
            object_ang_vel_w=object_ang_vel_w,
            object_validity=object_validity,
            scene_object_names=[self.cfg.object_key],
            sparse_contact_link_part_center_vector_w=sparse_center_vector,
            sparse_contact_relation_matrix=sparse_relation,
            sparse_contact_link_part_proximity=sparse_link_part_proximity,
            sparse_contact_link_part_point_proximity=sparse_point_proximity,
            sparse_contact_part_points_local=sparse_part_points,
            sparse_contact_part_point_mask=sparse_point_mask,
            sparse_contact_part_centers_local=sparse_part_centers,
            sparse_contact_robot_body_names=sparse_contact_robot_body_names,
            sparse_contact_robot_interest_names=sparse_contact_robot_interest_names,
            sparse_contact_part_names=sparse_contact_part_names,
        )

    def _load_motion_sequences(self):
        print(f"[Part2Link Motion] Loading motion files, should be {len(self._all_motion_files)} in total...")
        all_motion_sequences = list(map(self._read_motion_file, range(len(self._all_motion_files))))
        for motion in all_motion_sequences:
            assert isinstance(motion, Part2LinkMotionSequence), "Part2LinkMotion yields Part2LinkMotionSequence"
        all_motion_sequences = cast(list[Part2LinkMotionSequence], all_motion_sequences)
        print(f"[Part2Link Motion] All {len(all_motion_sequences)} motion files loaded.")
        print(
            "[Part2Link Motion] buffer lengths statistics:"
            f" mean: {np.array([motion.buffer_length for motion in all_motion_sequences]).mean()},"
            f" max: {np.array([motion.buffer_length for motion in all_motion_sequences]).max()},"
            f" min: {np.array([motion.buffer_length for motion in all_motion_sequences]).min()},"
        )

        first_motion = all_motion_sequences[0]
        self._all_motion_sequences = Part2LinkMotionSequence.make_emtpy_concat_batch(
            buffer_lengths=[int(motion.buffer_length) for motion in all_motion_sequences],
            num_joints=self.articulation_view.num_joints,
            num_links=self.num_link_to_ref,
            num_objects=1,
            num_sparse_links=first_motion.sparse_contact_link_part_center_vector_w.shape[1],
            num_sparse_parts=first_motion.sparse_contact_link_part_center_vector_w.shape[2],
            num_sparse_points=first_motion.sparse_contact_part_points_local.shape[2],
            device=self.buffer_device,
        )
        self._sparse_contact_robot_body_names = list(first_motion.sparse_contact_robot_body_names)
        self._sparse_contact_robot_interest_names = list(first_motion.sparse_contact_robot_interest_names)
        self._sparse_contact_part_names = list(first_motion.sparse_contact_part_names)

        for motion_id, motion in enumerate(all_motion_sequences):
            for attr in self._all_motion_sequences.attrs_with_frame_dim:
                getattr(self._all_motion_sequences, attr)[motion_id, : motion.buffer_length] = getattr(motion, attr)
            for attr in self._all_motion_sequences.attrs_only_batch_dim:
                getattr(self._all_motion_sequences, attr)[motion_id] = getattr(motion, attr)

    def fill_init_reference_state(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        env_origins: torch.Tensor,
        state_buffer: Part2LinkMotionReferenceState,
    ) -> None:
        super().fill_init_reference_state(env_ids, env_origins, state_buffer)

        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
        motion_ids = self._assigned_env_motion_selection[assigned_ids]
        frame_selection = torch.floor(
            self._motion_buffer_start_time_s[assigned_ids] * self._all_motion_sequences.framerate[motion_ids]
        ).to(torch.long)

        object_pos_w = self._all_motion_sequences.object_pos_w[motion_ids, frame_selection][..., 0, :]
        object_quat_w = self._all_motion_sequences.object_quat_w[motion_ids, frame_selection][..., 0, :]
        object_lin_vel_w = self._all_motion_sequences.object_lin_vel_w[motion_ids, frame_selection][..., 0, :]
        object_ang_vel_w = self._all_motion_sequences.object_ang_vel_w[motion_ids, frame_selection][..., 0, :]
        object_validity = self._all_motion_sequences.object_validity[motion_ids, frame_selection][..., 0]

        object_pos_w += self._get_motion_based_origin(env_origins, env_ids)
        state_buffer.object_pos_w[env_ids, 0] = object_pos_w.to(self.output_device)
        state_buffer.object_quat_w[env_ids, 0] = object_quat_w.to(self.output_device)
        state_buffer.object_lin_vel_w[env_ids, 0] = object_lin_vel_w.to(self.output_device)
        state_buffer.object_ang_vel_w[env_ids, 0] = object_ang_vel_w.to(self.output_device)
        state_buffer.object_validity[env_ids, 0] = object_validity.to(self.output_device)

    def fill_motion_data(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor,
        data_buffer: Part2LinkMotionReferenceData,
    ) -> None:
        super().fill_motion_data(env_ids, sample_timestamp, env_origins, data_buffer)

        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
        motion_ids = self._assigned_env_motion_selection[assigned_ids]
        frame_selections = torch.round(
            (self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1) + sample_timestamp.to(self.buffer_device))
            * self._all_motion_sequences.framerate[motion_ids].unsqueeze(-1)
        )
        frame_selections = torch.where(
            data_buffer.validity[env_ids].to(self.buffer_device),
            frame_selections,
            self._all_motion_sequences.buffer_length[motion_ids].unsqueeze(-1) - 1,
        ).to(torch.long)

        num_frames = frame_selections.shape[1]
        assigned_ids_across_frame = assigned_ids.unsqueeze(-1).expand(-1, num_frames).flatten()
        frame_selections_flat = frame_selections.flatten()
        motion_selection = self._assigned_env_motion_selection[assigned_ids_across_frame]

        object_attrs = ["object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"]
        for attr in object_attrs:
            source_data = getattr(self._all_motion_sequences, attr)[motion_selection, frame_selections_flat][
                ..., 0, :
            ].reshape(len(env_ids), num_frames, *getattr(data_buffer, attr).shape[3:])
            if attr == "object_pos_w":
                source_data += self._get_motion_based_origin(env_origins, env_ids).unsqueeze(1).to(self.buffer_device)
            getattr(data_buffer, attr)[env_ids, :, 0] = source_data.to(self.output_device)

        source_validity = self._all_motion_sequences.object_validity[motion_selection, frame_selections_flat][
            ..., 0
        ].reshape(len(env_ids), num_frames)
        data_buffer.object_validity[env_ids, :, 0] = source_validity.to(self.output_device)
        data_buffer.object_validity[env_ids, :, 0] &= data_buffer.validity[env_ids]

        sparse_attrs = [
            "sparse_contact_link_part_center_vector_w",
            "sparse_contact_relation_matrix",
            "sparse_contact_link_part_proximity",
            "sparse_contact_link_part_point_proximity",
            "sparse_contact_part_points_local",
            "sparse_contact_part_point_mask",
            "sparse_contact_part_centers_local",
        ]
        for attr in sparse_attrs:
            target = getattr(data_buffer, attr)
            source_data = getattr(self._all_motion_sequences, attr)[motion_selection, frame_selections_flat].reshape(
                len(env_ids), num_frames, *target.shape[2:]
            )
            target[env_ids] = source_data.to(self.output_device)

        data_buffer.sparse_contact_robot_body_names = list(self._sparse_contact_robot_body_names)
        data_buffer.sparse_contact_robot_interest_names = list(self._sparse_contact_robot_interest_names)
        data_buffer.sparse_contact_part_names = list(self._sparse_contact_part_names)
