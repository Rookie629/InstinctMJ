from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import instinct_mj.tasks.interaction.mdp.part2link as part2link_mdp

DATASET_ROOT = Path(
    os.getenv(
        "INSTINCT_PART2LINK_DATASET_ROOT",
        "/home/yangke/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object",
    )
)
SAMPLE_NPZ = DATASET_ROOT / "sofa" / "replay_contact_map.g1_retargeted.npz"
SOFA_JSON = DATASET_ROOT / "sparse_contact_maps" / "sofa.json"
G1_XML = Path(__file__).parents[1] / "src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml"


@pytest.mark.skipif(not SAMPLE_NPZ.exists(), reason="Part2Link sample NPZ is stored outside the repo")
def test_part2link_npz_sparse_metadata_and_native_joint_order():
    data = np.load(SAMPLE_NPZ, allow_pickle=True)

    assert data["sparse_contact_link_part_center_vector_w"].shape[-3:] == (4, 4, 3)
    assert data["sparse_contact_relation_matrix"].shape == (4, 4)
    assert data["sparse_contact_part_points_local"].shape == (4, 5, 3)
    assert data["sparse_contact_part_point_mask"].shape == (4, 5)
    assert data["sparse_contact_robot_body_names"].tolist() == [
        "pelvis_contour_link",
        "torso_link",
        "left_rubber_hand_link",
        "right_rubber_hand_link",
    ]
    assert data["sparse_contact_part_names"].tolist() == ["seat", "armrest_left", "armrest_right", "back"]

    root = ET.parse(G1_XML).getroot()
    native_joint_names = [
        joint.attrib.get("name")
        for joint in root.iter("joint")
        if joint.attrib.get("name") and joint.attrib.get("type", "hinge").lower() not in {"free", "ball"}
    ]
    source_joint_names = data["joint_names"].tolist()
    reorder = [source_joint_names.index(name) for name in native_joint_names]
    assert len(reorder) == 29
    assert native_joint_names[:3] == ["waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint"]


@pytest.mark.skipif(not SOFA_JSON.exists() or not SAMPLE_NPZ.exists(), reason="Part2Link metadata is outside the repo")
def test_sofa_json_is_validation_only_for_current_alias_mismatch():
    npz = np.load(SAMPLE_NPZ, allow_pickle=True)
    with open(SOFA_JSON) as f:
        sofa_json = json.load(f)

    json_parts = [part2link_mdp._PART_ALIAS.get(name, name) for name in sofa_json["object_parts_order"]]
    npz_parts = npz["sparse_contact_part_names"].tolist()
    assert json_parts != npz_parts


class _FakeScene(dict):
    @property
    def entities(self):
        return self


class _FakeRobot:
    def __init__(self, body_pos_w: torch.Tensor):
        self.data = SimpleNamespace(body_link_pos_w=body_pos_w)

    def find_bodies(self, names, preserve_order=False):
        del preserve_order
        return list(range(len(names))), list(names)


def _fake_env_for_part2link(current_link_pos_w: torch.Tensor, relation_value: float = 1.0):
    device = current_link_pos_w.device
    num_envs, num_links, _ = current_link_pos_w.shape
    num_parts = 1
    num_points = 1
    variant_name = "chair_14_alpha_1p00"

    data = SimpleNamespace(
        validity=torch.ones(num_envs, 1, dtype=torch.bool, device=device),
        sparse_contact_robot_body_names=["pelvis_contour_link"] * num_links,
        sparse_contact_part_names=["seat"],
        sparse_contact_link_part_center_vector_w=current_link_pos_w[:, None, :, None, :].clone(),
        sparse_contact_relation_matrix=torch.full((num_envs, 1, num_links, num_parts), relation_value, device=device),
        sparse_contact_link_part_proximity=torch.ones(num_envs, 1, num_links, num_parts, dtype=torch.bool, device=device),
        sparse_contact_link_part_point_proximity=torch.ones(
            num_envs, 1, num_links, num_parts, num_points, dtype=torch.bool, device=device
        ),
    )
    motion_ref = SimpleNamespace(
        data=data,
        ALL_INDICES=torch.arange(num_envs, device=device),
        aiming_frame_idx=torch.zeros(num_envs, dtype=torch.long, device=device),
        cfg=SimpleNamespace(scene_object_names=["box"]),
    )
    scene = _FakeScene(
        {
            "motion_reference": motion_ref,
            "robot": _FakeRobot(current_link_pos_w),
            variant_name: SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.zeros(num_envs, 3, device=device),
                    root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4),
                    root_link_lin_vel_w=torch.zeros(num_envs, 3, device=device),
                    root_link_ang_vel_w=torch.zeros(num_envs, 3, device=device),
                )
            ),
        }
    )
    env = SimpleNamespace(num_envs=num_envs, device=device, scene=scene)
    catalog = part2link_mdp.ObjectVariantCatalog(
        variants=[
            part2link_mdp.ObjectVariant(
                name=variant_name,
                chair_name="chair_14",
                alpha=1.0,
                mesh_path=Path("dummy.obj"),
                contact_points_path=Path("dummy.npz"),
                alpha_index=0,
            )
        ],
        centers_local=torch.zeros(1, num_parts, 3, device=device),
        points_local=torch.zeros(1, num_parts, num_points, 3, device=device),
        point_valid_mask=torch.ones(1, num_parts, num_points, dtype=torch.bool, device=device),
        part_names=["seat"],
        chair_names=["chair_14"],
        variant_names=[variant_name],
    )
    env._part2link_object_variant_state = part2link_mdp.ObjectVariantRuntimeState(
        catalog=catalog,
        active_variant_ids=torch.zeros(num_envs, dtype=torch.long, device=device),
        active_alpha=torch.ones(num_envs, device=device),
        active_scale=torch.ones(num_envs, device=device),
        active_precision_scale=torch.ones(num_envs, device=device),
        active_centers_local=torch.zeros(num_envs, num_parts, 3, device=device),
        active_points_local=torch.zeros(num_envs, num_parts, num_points, 3, device=device),
        active_point_valid_mask=torch.ones(num_envs, num_parts, num_points, dtype=torch.bool, device=device),
        reference_position_offsets=torch.zeros(num_envs, 3, device=device),
    )
    return env


def test_part2link_vector_reward_is_one_when_current_matches_gt():
    current_link_pos_w = torch.tensor([[[0.2, 0.0, 0.1]]], dtype=torch.float32)
    env = _fake_env_for_part2link(current_link_pos_w, relation_value=1.0)
    reward = part2link_mdp.part2link_vector_guidance_gauss(env, metadata_root=None)
    assert torch.allclose(reward, torch.ones_like(reward), atol=1.0e-5)


def test_part2link_forbidden_penalty_positive_inside_threshold():
    current_link_pos_w = torch.tensor([[[0.05, 0.0, 0.0]]], dtype=torch.float32)
    env = _fake_env_for_part2link(current_link_pos_w, relation_value=-1.0)
    penalty = part2link_mdp.part2link_forbidden_contact_penalty(env, metadata_root=None, forbidden_distance_threshold=0.10)
    assert penalty.item() > 0.0
