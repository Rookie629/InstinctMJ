from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg
from mjlab.managers import ObservationTermCfg as ObsTermCfg
from mjlab.managers import RewardTermCfg as RewTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg as DoneTermCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer.viewer_config import ViewerConfig
from instinct_mj.monitors import MonitorTermCfg

import instinct_mj.tasks.interaction.mdp as interaction_mdp
import instinct_mj.tasks.shadowing.perceptive_hoi.perceptive_env_cfg as perceptual_cfg
from instinct_mj.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_MJCF_PATH,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuator_cfgs,
)
from instinct_mj.motion_reference import Part2LinkMotionReferenceData, Part2LinkMotionReferenceState
from instinct_mj.motion_reference.motion_files.part2link_motion_cfg import Part2LinkMotionCfg as Part2LinkMotionCfgBase
from instinct_mj.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg
from instinct_mj.motion_reference.utils import motion_interpolate_bilinear

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

# Resolve project-root-relative data directory.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))
)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "datasets", "interaction")

DEFAULT_DATASET_ROOT = os.path.join(_DATA_DIR, "output_npz_29dof_with_object")
PART2LINK_DATASET_ROOT = os.path.expanduser(os.getenv("INSTINCT_PART2LINK_DATASET_ROOT", DEFAULT_DATASET_ROOT))
PART2LINK_METADATA_YAML = os.path.join(PART2LINK_DATASET_ROOT, "metadata.yaml")
PART2LINK_METADATA_ROOT = os.path.join(PART2LINK_DATASET_ROOT, "sparse_contact_maps")
SITTING_EXTENSION_ROOT = os.path.join(PART2LINK_DATASET_ROOT, "sofa_exntend_obj")
SITTING_ASSET_CACHE = os.getenv("INSTINCT_PART2LINK_ASSET_CACHE")
SITTING_COLLISION_CACHE = os.getenv("INSTINCT_PART2LINK_COLLISION_CACHE")
SITTING_COLLISION_VISUAL_MODE = os.getenv("INSTINCT_PART2LINK_COLLISION_VISUAL_MODE", "mesh")
SITTING_PART2LINK_ENV_SPACING = 8.0
SITTING_VARIANT_SCALE_RANGE = (0.8, 1.2)
SITTING_VARIANT_PRECISION_SCALE_RANGE = (0.8, 1.4)

SITTING_VARIANT_CHAIR_NAMES_DEFAULT = (
    "chair_14",
    "chair_15",
    "chair_17",
    "chair_18",
    "chair_20",
    "chair_22",
    "chair_28",
    "chair_30",
    "chair_32",
    "chair_33",
    "chair_37",
    "chair_39",
    "chair_40",
    "chair_41",
    "chair_43",
    "chair_44",
    "chair_46",
    "chair_48",
    "chair_51",
    "chair_55",
)

G1_29DOF_LINKS = [
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "waist_yaw_link",
    "waist_roll_link",
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "left_rubber_hand",
    "right_rubber_hand",
]


def _parse_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or default


def _parse_alpha_values(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return default
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    return parsed or default


SITTING_VARIANT_ALPHA_VALUES = _parse_alpha_values(
    os.getenv("SITTING_PART2LINK_ALPHA_VALUES"),
    (1.0, 0.8, 0.5, 0.0),
)
SITTING_VARIANT_ALPHA_LABEL = "_".join(f"{alpha:.2f}".replace(".", "p") for alpha in SITTING_VARIANT_ALPHA_VALUES)
SITTING_VARIANT_CHAIR_NAMES = _parse_csv(
    os.getenv("SITTING_PART2LINK_CHAIR_NAMES"),
    SITTING_VARIANT_CHAIR_NAMES_DEFAULT,
)


def _load_stage_catalog():
    return interaction_mdp.load_object_variant_catalog(
        extension_root=SITTING_EXTENSION_ROOT,
        chair_names=SITTING_VARIANT_CHAIR_NAMES,
        alpha_values=SITTING_VARIANT_ALPHA_VALUES,
        asset_cache=SITTING_ASSET_CACHE,
        device="cpu",
        single_object_type="box",
    )


def _part2link_object_entity_names() -> tuple[str, ...]:
    return tuple(_load_stage_catalog().variant_names)


def _make_part2link_camera_mesh_prim_paths() -> list[str]:
    return (
        ["/World/ground"]
        + [f"/World/envs/env_.*/Robot/{link_name}" for link_name in G1_29DOF_LINKS]
        + [f"/World/envs/env_.*/{object_name}" for object_name in _part2link_object_entity_names()]
    )


def _make_g1_part2link_scene_sensors(*, motion_reference) -> tuple:
    sensors = list(perceptual_cfg.make_hoi_scene_sensors(motion_reference=motion_reference))
    camera_cfg = next(sensor_cfg for sensor_cfg in sensors if sensor_cfg.name == "camera")
    camera_cfg.mesh_prim_paths = _make_part2link_camera_mesh_prim_paths()
    return tuple(sensors)


def _resolve_part2link_collision_assets(variant) -> tuple[str | None, tuple[str, ...]]:
    if not SITTING_COLLISION_CACHE:
        return None, tuple()
    collision_dir = Path(SITTING_COLLISION_CACHE).expanduser().resolve() / variant.name
    visual_path = collision_dir / "visual.obj"
    if not visual_path.exists():
        raise FileNotFoundError(
            f"Missing CoACD visual mesh for {variant.name}. Expected {visual_path}."
        )
    collision_paths = tuple(str(path) for path in sorted(collision_dir.glob("collision_*.obj")))
    if not collision_paths:
        raise FileNotFoundError(
            f"Missing CoACD collision meshes for {variant.name}. "
            f"Expected files matching {collision_dir / 'collision_*.obj'}."
        )
    return str(visual_path), collision_paths


def _make_part2link_object_spec(mesh_file_path: str, collision_mesh_file_paths: tuple[str, ...] = tuple()):
    def spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        mesh = spec.add_mesh(name="object_mesh", file=os.path.expanduser(mesh_file_path), scale=(1.0, 1.0, 1.0))
        body = spec.worldbody.add_body(name="object", mocap=True)
        if collision_mesh_file_paths:
            show_collision_as_visual = SITTING_COLLISION_VISUAL_MODE == "collision"
            if not show_collision_as_visual:
                body.add_geom(
                    name="object_visual",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh.name,
                    contype=0,
                    conaffinity=0,
                    density=0.0,
                    group=2,
                    rgba=(0.2, 0.55, 0.85, 1.0),
                )
            mass_per_collision = 1.0 / len(collision_mesh_file_paths)
            for index, collision_mesh_file_path in enumerate(collision_mesh_file_paths):
                collision_mesh = spec.add_mesh(
                    name=f"object_collision_mesh_{index:03d}",
                    file=os.path.expanduser(collision_mesh_file_path),
                    scale=(1.0, 1.0, 1.0),
                )
                body.add_geom(
                    name=f"object_collision_{index:03d}",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=collision_mesh.name,
                    mass=mass_per_collision,
                    group=2 if show_collision_as_visual else 3,
                    rgba=(0.2, 0.55, 0.85, 0.8) if show_collision_as_visual else (1.0, 0.55, 0.05, 0.35),
                    friction=(1.0, 0.005, 0.0001),
                )
        else:
            body.add_geom(
                name="object_geom",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh.name,
                mass=1.0,
                rgba=(0.2, 0.55, 0.85, 1.0),
                friction=(1.0, 0.005, 0.0001),
            )
        return spec

    return spec_fn


def _make_part2link_entities(*, include_reference: bool = False) -> dict[str, EntityCfg]:
    entities: dict[str, EntityCfg] = {
        "robot": deepcopy(G1_CFG),
    }
    if include_reference:
        robot_reference = deepcopy(G1_CFG)
        robot_reference.collisions = (
            CollisionCfg(
                geom_names_expr=(".*",),
                contype=0,
                conaffinity=0,
            ),
        )
        entities["robot_reference"] = robot_reference

    for variant in _load_stage_catalog().variants:
        visual_mesh_path, collision_mesh_paths = _resolve_part2link_collision_assets(variant)
        entities[variant.name] = EntityCfg(
            spec_fn=_make_part2link_object_spec(
                visual_mesh_path or str(variant.mesh_path),
                collision_mesh_paths,
            )
        )
    return entities


@dataclass(kw_only=True)
class Part2LinkMotionCfg(Part2LinkMotionCfgBase):
    path: str = field(default_factory=lambda: PART2LINK_DATASET_ROOT)
    metadata_yaml: str | None = field(default_factory=lambda: PART2LINK_METADATA_YAML)
    object_key: str = "box"
    ensure_link_below_zero_ground: bool = False
    motion_start_from_middle_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    motion_start_height_offset: float = 0.0
    motion_bin_length_s: float | None = 1.0
    buffer_device: str = "output_device"
    motion_interpolate_func: object = field(default_factory=lambda: motion_interpolate_bilinear)
    velocity_estimation_method: str = "frontbackward"
    object_velocity_estimation_method: str | None = "frontbackward"
    env_starting_stub_sampling_strategy: str = "concat_motion_bins"


motion_reference_cfg = MotionReferenceManagerCfg(
    name="motion_reference",
    entity_name="robot",
    robot_model_path=G1_MJCF_PATH,
    data_class_type=Part2LinkMotionReferenceData,
    state_class_type=Part2LinkMotionReferenceState,
    scene_object_names=["box"],
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.1,
    update_period=0.02,
    num_frames=10,
    data_start_from="current_time",
    visualizing_robot_offset=(2.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    visualizing_marker_types=["relative_links", "links"],
    motion_buffers={
        "Part2LinkMotion": Part2LinkMotionCfg(),
    },
    mp_split_method="None",
)
motion_reference_cfg_play = deepcopy(motion_reference_cfg)
motion_reference_cfg_play.debug_vis = True
motion_reference_cfg_play.reference_entity_name = "robot_reference"


def make_part2link_observations():
    observations = perceptual_cfg.make_hoi_observations()
    critic_terms = observations["critic"].terms
    object_entities = _part2link_object_entity_names()
    critic_terms.update(
        {
            "object_pos": ObsTermCfg(
                func=interaction_mdp.object_position,
                params={
                    "object_entity_names": object_entities,
                    "robot_cfg": SceneEntityCfg("robot"),
                    "in_base_frame": True,
                },
            ),
            "object_ori": ObsTermCfg(
                func=interaction_mdp.object_orientation_tannorm,
                params={
                    "object_entity_names": object_entities,
                    "robot_cfg": SceneEntityCfg("robot"),
                    "in_base_frame": True,
                },
            ),
            "object_pos_ref": ObsTermCfg(
                func=interaction_mdp.object_reference_position,
                params={
                    "reference_cfg": SceneEntityCfg("motion_reference"),
                    "robot_cfg": SceneEntityCfg("robot"),
                    "object_name": "box",
                    "in_base_frame": True,
                },
            ),
            "object_ori_ref": ObsTermCfg(
                func=interaction_mdp.object_reference_orientation_tannorm,
                params={
                    "reference_cfg": SceneEntityCfg("motion_reference"),
                    "robot_cfg": SceneEntityCfg("robot"),
                    "object_name": "box",
                    "in_base_frame": True,
                },
            ),
            "object_pos_err": ObsTermCfg(
                func=interaction_mdp.object_position_error,
                params={
                    "object_entity_names": object_entities,
                    "reference_cfg": SceneEntityCfg("motion_reference"),
                    "robot_cfg": SceneEntityCfg("robot"),
                    "object_name": "box",
                    "in_base_frame": True,
                },
            ),
            "object_lin_vel": ObsTermCfg(
                func=interaction_mdp.object_linear_velocity,
                params={
                    "object_entity_names": object_entities,
                    "robot_cfg": SceneEntityCfg("robot"),
                    "in_base_frame": True,
                },
            ),
            "object_ang_vel": ObsTermCfg(
                func=interaction_mdp.object_angular_velocity,
                params={
                    "object_entity_names": object_entities,
                    "robot_cfg": SceneEntityCfg("robot"),
                    "in_base_frame": True,
                },
            ),
            "seat_object_contact": ObsTermCfg(func=interaction_mdp.seat_object_contact),
        }
    )
    return observations


def make_part2link_rewards():
    rewards = perceptual_cfg.make_hoi_rewards()
    rewards["base_position_imitation_gauss"].weight = 0.75
    rewards["base_rot_imitation_gauss"].weight = 0.5
    rewards["link_pos_imitation_gauss"].weight = 0.5
    rewards["link_rot_imitation_gauss"].weight = 0.5
    rewards["link_lin_vel_imitation_gauss"].weight = 0.5
    rewards["link_ang_vel_imitation_gauss"].weight = 0.5
    rewards["undesired_contacts"].params["asset_cfg"].body_names = [
        r"^(?!pelvis_contour_link$)(?!left_hip_roll_link$)(?!right_hip_roll_link$)"
        r"(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)"
        r"(?!right_wrist_yaw_link$).+$"
    ]
    rewards["seat_object_contact_ref_phase"] = RewTermCfg(
        func=interaction_mdp.object_contact_reference_phase,
        weight=3.0,
        params={
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "object_name": "box",
            "threshold": 0.12,
            "normalize": True,
            "print_reason": False,
            "debug_label": "seat_object_contact",
        },
    )
    rewards["part2link_vector_guidance_gauss"] = RewTermCfg(
        func=interaction_mdp.part2link_vector_guidance_gauss,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "object_name": "sofa",
            "metadata_root": PART2LINK_METADATA_ROOT,
            "tracking_sigma": 0.25,
            "tracking_tolerance": 0.08,
            "precision_scale_range": SITTING_VARIANT_PRECISION_SCALE_RANGE,
            "debug_vis": False,
            "debug_vis_max_envs": 1,
            "debug_vis_show_contact_points": True,
            "debug_vis_show_part_centers": True,
            "debug_vis_point_radius": 0.025,
            "debug_vis_center_radius": 0.035,
            "debug_vis_line_radius": 0.008,
            "debug_vis_ignore_contact_phase": False,
        },
    )
    rewards["part2link_forbidden_contact_penalty"] = RewTermCfg(
        func=interaction_mdp.part2link_forbidden_contact_penalty,
        weight=-1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "object_name": "sofa",
            "metadata_root": PART2LINK_METADATA_ROOT,
            "forbidden_distance_threshold": 0.10,
        },
    )
    return rewards


def make_part2link_events():
    events = perceptual_cfg.make_hoi_events()
    object_entities = _part2link_object_entity_names()
    part2link_catalog = _load_stage_catalog()
    events["reset_rigid_objects_state_by_reference"] = EventTermCfg(
        func=interaction_mdp.reset_object_variant_by_reference,
        mode="reset",
        params={
            "object_entity_names": object_entities,
            "catalog": part2link_catalog,
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "object_name": "box",
            "scale_distribution_params": SITTING_VARIANT_SCALE_RANGE,
            "precision_scale_range": SITTING_VARIANT_PRECISION_SCALE_RANGE,
            "base_lin_vel_ratio": 1.0,
            "base_ang_vel_ratio": 1.0,
            "apply_env_spacing": False,
        },
    )
    events["update_rigid_objects_state_by_reference"] = EventTermCfg(
        func=interaction_mdp.update_object_variant_by_reference,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "object_entity_names": object_entities,
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "object_name": "box",
            "invalid_object_pos": (0.0, 0.0, -100.0),
        },
    )
    events["reset_robot"].params["randomize_pose_range"].update({"x": (-0.03, 0.03), "y": (-0.03, 0.03)})
    events["reset_robot"].params["randomize_joint_pos_range"] = (-0.2, 0.2)
    return events


def make_part2link_curriculum():
    curriculum = perceptual_cfg.make_hoi_curriculum()
    curriculum["tracking_sigma_annealing"] = CurriculumTermCfg(
        func=interaction_mdp.TrackingSigmaCurriculum,
        params={
            "reward_group_name": None,
            "term_names": ["part2link_vector_guidance_gauss"],
            "initial_sigmas": [0.35],
            "final_sigmas": [0.12],
            "start_step": 0,
            "end_step": 3_000_000,
            "min_sigma": 0.1,
        },
    )
    curriculum["object_alpha_curriculum"] = CurriculumTermCfg(
        func=interaction_mdp.ObjectAlphaCurriculum,
        params={
            "initial_alpha": 1.0,
            "final_alpha": 0.0,
            "start_step": 0,
            "end_step": 3_000_000,
        },
    )
    return curriculum


def make_part2link_terminations():
    terminations = perceptual_cfg.make_hoi_terminations()
    terminations["illegal_reset_contact"].params["asset_cfg"].body_names = [
        r"^(?!pelvis_contour_link$)(?!left_hip_roll_link$)(?!right_hip_roll_link$)"
        r"(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)"
        r"(?!right_wrist_yaw_link$).+$"
    ]
    terminations["object_nonseat_contact"] = DoneTermCfg(
        func=interaction_mdp.any_object_filtered_contact,
        time_out=False,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "distance_threshold": 0.08,
            "print_reason": False,
        },
    )
    return terminations


@dataclass(kw_only=True)
class G1InteractionSittingPart2LinkShadowingEnvCfg(perceptual_cfg.PerceptiveHoiShadowingEnvCfg):
    headless: bool = False
    """Disable GUI-dependent debug visualizations (camera rays, depth overlay) for headless runs."""
    scene: perceptual_cfg.PerceptiveHoiShadowingSceneCfg = field(
        default_factory=lambda: perceptual_cfg.PerceptiveHoiShadowingSceneCfg(
            num_envs=4096,
            env_spacing=SITTING_PART2LINK_ENV_SPACING,
            entities=_make_part2link_entities(),
            sensors=_make_g1_part2link_scene_sensors(
                motion_reference=deepcopy(motion_reference_cfg),
            ),
        )
    )
    observations: dict = field(default_factory=make_part2link_observations)
    rewards: dict = field(default_factory=make_part2link_rewards)
    events: dict = field(default_factory=make_part2link_events)
    curriculum: dict = field(default_factory=make_part2link_curriculum)
    terminations: dict = field(default_factory=make_part2link_terminations)

    def __post_init__(self):
        super().__post_init__()
        robot_cfg = self.scene.entities["robot"]
        motion_reference_cfg = next(
            sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "motion_reference"
        )
        robot_cfg.articulation.actuators = beyondmimic_g1_29dof_actuator_cfgs
        self.actions["joint_pos"].scale = beyondmimic_action_scale
        self.sim.njmax = None
        self.sim.nconmax = None
        self.sim.mujoco.jacobian = "sparse"
        self.scene.env_spacing = SITTING_PART2LINK_ENV_SPACING
        self.rewards["undesired_contacts"] = None

        self.observations["critic"].terms["link_pos"].params[
            "asset_cfg"
        ].body_names = motion_reference_cfg.link_of_interests
        self.observations["critic"].terms["link_rot"].params[
            "asset_cfg"
        ].body_names = motion_reference_cfg.link_of_interests

        motion_buffer = next(iter(motion_reference_cfg.motion_buffers.values()))
        self.run_name = "G1InteractionSittingPart2LinkShadowing" + "".join(
            [
                "_alphaObjectPart2Link",
                f"_alphas{SITTING_VARIANT_ALPHA_LABEL}",
                (
                    "_concatMotionBins"
                    if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins"
                    else "_independentMotionBins"
                ),
            ]
        )
        self.terminations["out_of_border"] = None

        self.monitors = {
            "part2link_debug": MonitorTermCfg(func=interaction_mdp.Part2LinkDebugVisualizer),
        }

        camera_cfg = next(sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "camera")
        camera_cfg.debug_vis = not self.headless
        self.observations["policy"].terms["depth_image"].params["debug_vis"] = not self.headless


@dataclass(kw_only=True)
class G1InteractionSittingPart2LinkShadowingEnvCfg_PLAY(G1InteractionSittingPart2LinkShadowingEnvCfg):
    headless: bool = False
    scene: perceptual_cfg.PerceptiveHoiShadowingSceneCfg = field(
        default_factory=lambda: perceptual_cfg.PerceptiveHoiShadowingSceneCfg(
            num_envs=1,
            env_spacing=2.5,
            entities=_make_part2link_entities(include_reference=True),
            sensors=_make_g1_part2link_scene_sensors(
                motion_reference=deepcopy(motion_reference_cfg_play),
            ),
        )
    )
    viewer: ViewerConfig = field(
        default_factory=lambda: ViewerConfig(
            lookat=(0.0, 0.0, 0.0),
            distance=2.1213,
            elevation=45.0,
            azimuth=0.0,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 2.5
        motion_reference_cfg = next(
            sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "motion_reference"
        )
        camera_cfg = next(sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "camera")
        motion_buffer = next(iter(motion_reference_cfg.motion_buffers.values()))
        motion_buffer.motion_start_from_middle_range = [0.0, 0.0]
        motion_buffer.motion_bin_length_s = None
        motion_buffer.env_starting_stub_sampling_strategy = "independent"
        self.curriculum["beyond_adaptive_sampling"] = None
        self.events["bin_fail_counter_smoothing"] = None
        self.rewards["part2link_vector_guidance_gauss"].params["debug_vis"] = True
        self.rewards["part2link_vector_guidance_gauss"].params["debug_vis_max_envs"] = 1
        self.rewards["part2link_vector_guidance_gauss"].params["debug_vis_show_contact_points"] = True
        self.rewards["part2link_vector_guidance_gauss"].params["debug_vis_show_part_centers"] = True
        self.rewards["part2link_vector_guidance_gauss"].params["debug_vis_ignore_contact_phase"] = False
        self.events["reset_rigid_objects_state_by_reference"].params["scale_distribution_params"] = (1.0, 1.0)
        self.events["add_joint_default_pos"] = None
        self.events["base_com"] = None
        self.events["physics_material"] = None
        self.terminations["base_pos_too_far"] = None
        self.terminations["base_pg_too_far"] = None
        self.terminations["link_pos_too_far"] = None
        self.terminations["out_of_border"] = None
        self.events["reset_robot"].params["randomize_pose_range"]["x"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["y"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["z"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_joint_pos_range"] = (0.0, 0.0)
        motion_reference_cfg.visualizing_robot_offset = (0.0, 0.0, 0.0)
        self.viewer.entity_name = "robot_reference"
