from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch

import instinct_mj.utils.torch as torch_utils

from .motion_reference_hoi_data import HoiMotionReferenceData, HoiMotionReferenceState, HoiMotionSequence


@dataclass
class Part2LinkMotionSequence(HoiMotionSequence):
    """HOI motion sequence with sparse Part2Link contact-map tensors."""

    sparse_contact_link_part_center_vector_w: torch.Tensor = None  # type: ignore
    sparse_contact_relation_matrix: torch.Tensor = None  # type: ignore
    sparse_contact_link_part_proximity: torch.Tensor = None  # type: ignore
    sparse_contact_link_part_point_proximity: torch.Tensor = None  # type: ignore
    sparse_contact_part_points_local: torch.Tensor = None  # type: ignore
    sparse_contact_part_point_mask: torch.Tensor = None  # type: ignore
    sparse_contact_part_centers_local: torch.Tensor = None  # type: ignore
    sparse_contact_robot_body_names: list[str] = None  # type: ignore
    sparse_contact_robot_interest_names: list[str] = None  # type: ignore
    sparse_contact_part_names: list[str] = None  # type: ignore

    attrs_with_frame_dim: tuple = HoiMotionSequence.attrs_with_frame_dim + (
        "sparse_contact_link_part_center_vector_w",
        "sparse_contact_relation_matrix",
        "sparse_contact_link_part_proximity",
        "sparse_contact_link_part_point_proximity",
        "sparse_contact_part_points_local",
        "sparse_contact_part_point_mask",
        "sparse_contact_part_centers_local",
    )

    @staticmethod
    def make_emtpy_concat_batch(
        buffer_lengths: Sequence[int],
        num_joints: int,
        num_links: int,
        num_objects: int,
        num_sparse_links: int,
        num_sparse_parts: int,
        num_sparse_points: int,
        device=torch.device("cpu"),
    ) -> Part2LinkMotionSequence:
        base = HoiMotionSequence.make_emtpy_concat_batch(
            buffer_lengths=buffer_lengths,
            num_joints=num_joints,
            num_links=num_links,
            num_objects=num_objects,
            device=device,
        )
        base_kwargs = {
            field.name: getattr(base, field.name)
            for field in fields(HoiMotionSequence)
            if hasattr(base, field.name) and field.name not in ("attrs_with_frame_dim", "attrs_only_batch_dim")
        }
        return Part2LinkMotionSequence(
            **base_kwargs,
            sparse_contact_link_part_center_vector_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_sparse_links, num_sparse_parts, 3), device=device
            ),
            sparse_contact_relation_matrix=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_sparse_links, num_sparse_parts), device=device
            ),
            sparse_contact_link_part_proximity=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_sparse_links, num_sparse_parts), dtype=torch.bool, device=device
            ),
            sparse_contact_link_part_point_proximity=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths,
                data_shape=(num_sparse_links, num_sparse_parts, num_sparse_points),
                dtype=torch.bool,
                device=device,
            ),
            sparse_contact_part_points_local=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_sparse_parts, num_sparse_points, 3), device=device
            ),
            sparse_contact_part_point_mask=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_sparse_parts, num_sparse_points), dtype=torch.bool, device=device
            ),
            sparse_contact_part_centers_local=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_sparse_parts, 3), device=device
            ),
        )


@dataclass
class Part2LinkMotionReferenceData(HoiMotionReferenceData):
    """Motion-reference data exposed to Part2Link rewards and observations."""

    sparse_contact_link_part_center_vector_w: torch.Tensor = None  # type: ignore
    sparse_contact_relation_matrix: torch.Tensor = None  # type: ignore
    sparse_contact_link_part_proximity: torch.Tensor = None  # type: ignore
    sparse_contact_link_part_point_proximity: torch.Tensor = None  # type: ignore
    sparse_contact_part_points_local: torch.Tensor = None  # type: ignore
    sparse_contact_part_point_mask: torch.Tensor = None  # type: ignore
    sparse_contact_part_centers_local: torch.Tensor = None  # type: ignore
    sparse_contact_robot_body_names: list[str] = None  # type: ignore
    sparse_contact_robot_interest_names: list[str] = None  # type: ignore
    sparse_contact_part_names: list[str] = None  # type: ignore

    @staticmethod
    def make_empty(
        num_envs: int,
        num_frames: int,
        num_joints: int,
        num_links: int,
        num_objects: int,
        device=torch.device("cpu"),
        scene_object_names: list[str] = None,
        num_sparse_links: int = 4,
        num_sparse_parts: int = 4,
        num_sparse_points: int = 5,
    ) -> Part2LinkMotionReferenceData:
        base = HoiMotionReferenceData.make_empty(
            num_envs=num_envs,
            num_frames=num_frames,
            num_joints=num_joints,
            num_links=num_links,
            num_objects=num_objects,
            device=device,
            scene_object_names=scene_object_names,
        )
        base_kwargs = {
            field.name: getattr(base, field.name)
            for field in fields(HoiMotionReferenceData)
            if hasattr(base, field.name)
        }
        return Part2LinkMotionReferenceData(
            **base_kwargs,
            sparse_contact_link_part_center_vector_w=torch.zeros(
                num_envs, num_frames, num_sparse_links, num_sparse_parts, 3, device=device
            ),
            sparse_contact_relation_matrix=torch.zeros(
                num_envs, num_frames, num_sparse_links, num_sparse_parts, device=device
            ),
            sparse_contact_link_part_proximity=torch.zeros(
                num_envs, num_frames, num_sparse_links, num_sparse_parts, dtype=torch.bool, device=device
            ),
            sparse_contact_link_part_point_proximity=torch.zeros(
                num_envs,
                num_frames,
                num_sparse_links,
                num_sparse_parts,
                num_sparse_points,
                dtype=torch.bool,
                device=device,
            ),
            sparse_contact_part_points_local=torch.zeros(
                num_envs, num_frames, num_sparse_parts, num_sparse_points, 3, device=device
            ),
            sparse_contact_part_point_mask=torch.zeros(
                num_envs, num_frames, num_sparse_parts, num_sparse_points, dtype=torch.bool, device=device
            ),
            sparse_contact_part_centers_local=torch.zeros(num_envs, num_frames, num_sparse_parts, 3, device=device),
            sparse_contact_robot_body_names=[],
            sparse_contact_robot_interest_names=[],
            sparse_contact_part_names=[],
        )


Part2LinkMotionReferenceState = HoiMotionReferenceState
