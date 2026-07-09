from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field

from mjlab.entity import EntityCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, SceneEntityCfg
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.tasks.shadowing.perceptive_hoi.perceptive_env_cfg as perceptual_cfg
from instinct_mj.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_MJCF_PATH,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuator_cfgs,
)
from instinct_mj.envs.mdp.curriculums.object_difficulty import (
    ObjectScaleCurriculum,
)
from instinct_mj.envs.mdp.events.object_variant import (
    ObjectVariantCatalog,
    reset_object_variant_by_reference,
    update_object_variant_by_reference,
)
from instinct_mj.monitors import ActuatorMonitorTerm, MonitorTermCfg, ShadowingBasePosMonitorTerm
from instinct_mj.motion_reference import HoiMotionReferenceData, HoiMotionReferenceState
from instinct_mj.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg
from instinct_mj.motion_reference.motion_files.omomo_motion_cfg import OmomoMotionCfg as OmomoMotionCfgBase
from instinct_mj.motion_reference.utils import motion_interpolate_bilinear
from instinct_mj.tasks.shadowing.perceptive_hoi.mdp.object_variant_hoi import (
    OMOMO_OBJECT_TYPES,
    build_hoi_multi_variant_catalog,
    build_hoi_single_mesh_catalog,
    make_hoi_variant_camera_mesh_prim_paths,
    make_hoi_variant_entities,
)

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

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
]

OMOMO_DATASET_PATH = "~/Datasets/OMOMO/retargeted"

# ---------------------------------------------------------------------------
# Multi-variant object support — gated behind INSTINCT_HOI_VARIANT_ROOT
# ---------------------------------------------------------------------------

HOI_VARIANT_ROOT = os.getenv("INSTINCT_HOI_VARIANT_ROOT", None)
HOI_VARIANT_ASSET_CACHE = os.getenv("INSTINCT_HOI_VARIANT_ASSET_CACHE", None)
_enable_multi_variant = HOI_VARIANT_ROOT is not None


def _load_hoi_catalog() -> ObjectVariantCatalog:
    """Load HOI object variant catalog.

    Single-mesh mode (default): degenerate catalog with 1 variant per type.
    Multi-variant mode (``INSTINCT_HOI_VARIANT_ROOT`` set): discover variants
    from disk.
    """
    if _enable_multi_variant:
        return build_hoi_multi_variant_catalog(
            extension_root=HOI_VARIANT_ROOT,
            object_types=OMOMO_OBJECT_TYPES,
            asset_cache=HOI_VARIANT_ASSET_CACHE,
        )
    return build_hoi_single_mesh_catalog()


_HOI_CATALOG = _load_hoi_catalog()


def _make_hoi_camera_mesh_prim_paths() -> list[str]:
    return make_hoi_variant_camera_mesh_prim_paths(_HOI_CATALOG, link_names=G1_29DOF_LINKS)


def _make_g1_hoi_scene_sensors(*, motion_reference) -> tuple:
    sensors = list(perceptual_cfg.make_hoi_scene_sensors(motion_reference=motion_reference))
    camera_cfg = next(sensor_cfg for sensor_cfg in sensors if sensor_cfg.name == "camera")
    camera_cfg.mesh_prim_paths = _make_hoi_camera_mesh_prim_paths()
    return tuple(sensors)


def _make_hoi_entities(*, include_reference: bool = False) -> dict[str, EntityCfg]:
    """Build entity dict from the HOI catalog (catalog-driven)."""
    return make_hoi_variant_entities(
        _HOI_CATALOG,
        include_reference=include_reference,
        robot_cfg=G1_CFG,
    )


@dataclass(kw_only=True)
class OmomoMotionCfg(OmomoMotionCfgBase):
    path: object = field(default_factory=lambda: os.path.expanduser(OMOMO_DATASET_PATH))
    ensure_link_below_zero_ground: bool = False
    motion_start_from_middle_range: list = field(default_factory=lambda: [0.0, 0.0])
    motion_start_height_offset: float = 0.0
    motion_bin_length_s: float | None = 1.0
    buffer_device: str = "output_device"
    motion_interpolate_func: object = field(default_factory=lambda: motion_interpolate_bilinear)
    velocity_estimation_method: str = "frontbackward"
    env_starting_stub_sampling_strategy: str = "concat_motion_bins"


motion_reference_cfg = MotionReferenceManagerCfg(
    name="motion_reference",
    entity_name="robot",
    robot_model_path=G1_MJCF_PATH,
    data_class_type=HoiMotionReferenceData,
    state_class_type=HoiMotionReferenceState,
    scene_object_names=_HOI_CATALOG.object_types,
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
    # set the robot_reference directly at where they are in the scene
    # DO NOT FORGET to change this when in actual training
    visualizing_robot_offset=(2.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    visualizing_marker_types=["relative_links", "links"],
    motion_buffers={
        "OmomoMotion": OmomoMotionCfg(),
    },
    mp_split_method="None",
)
motion_reference_cfg_play = deepcopy(motion_reference_cfg)
motion_reference_cfg_play.debug_vis = True
motion_reference_cfg_play.reference_entity_name = "robot_reference"


@dataclass(kw_only=True)
class G1PerceptiveHoiShadowingEnvCfg(perceptual_cfg.PerceptiveHoiShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveHoiShadowingSceneCfg = field(
        default_factory=lambda: perceptual_cfg.PerceptiveHoiShadowingSceneCfg(
            num_envs=4096,
            entities=_make_hoi_entities(),
            sensors=_make_g1_hoi_scene_sensors(
                motion_reference=deepcopy(motion_reference_cfg),
            ),
        )
    )

    def __post_init__(self):
        super().__post_init__()

        robot_cfg = self.scene.entities["robot"]
        motion_reference_cfg = next(
            sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "motion_reference"
        )

        robot_cfg.articulation.actuators = beyondmimic_g1_29dof_actuator_cfgs
        # self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 0.3
        self.actions["joint_pos"].scale = beyondmimic_action_scale
        self.sim.njmax = 700
        self.sim.nconmax = 128
        self.sim.mujoco.jacobian = "sparse"

        MOTION_NAME = list(motion_reference_cfg.motion_buffers.keys())[0]
        motion_buffer = motion_reference_cfg.motion_buffers[MOTION_NAME]

        # match key links for observation terms
        self.observations["critic"].terms["link_pos"].params[
            "asset_cfg"
        ].body_names = motion_reference_cfg.link_of_interests
        self.observations["critic"].terms["link_rot"].params[
            "asset_cfg"
        ].body_names = motion_reference_cfg.link_of_interests

        self.run_name = "g1PerceptiveHoi" + "".join(
            [
                (
                    "_concatMotionBins"
                    if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins"
                    else "_independentMotionBins"
                ),
            ]
        )
        # HOI task does not use terrain constraints.
        self.terminations["out_of_border"] = None

        # ------------------------------------------------------------------
        # Object domain randomization (always active)
        # ------------------------------------------------------------------
        object_dr = perceptual_cfg.make_hoi_object_dr_events(
            object_names=_HOI_CATALOG.variant_names,
        )
        self.events.update(object_dr)

        # ------------------------------------------------------------------
        # Multi-variant mode: variant switching + curriculum
        # ------------------------------------------------------------------
        if _enable_multi_variant:
            self.events["reset_rigid_objects_state_by_reference"] = EventTermCfg(
                func=reset_object_variant_by_reference,
                mode="reset",
                params={
                    "object_entity_names": _HOI_CATALOG.variant_names,
                    "catalog": _HOI_CATALOG,
                    "scale_distribution_params": (0.8, 1.2),
                    "precision_scale_range": (0.8, 1.4),
                },
            )
            self.events["update_rigid_objects_state_by_reference"] = EventTermCfg(
                func=update_object_variant_by_reference,
                mode="interval",
                interval_range_s=(0.02, 0.02),
                params={
                    "object_entity_names": _HOI_CATALOG.variant_names,
                    "invalid_object_pos": (0.0, 0.0, -100.0),
                },
            )
            # NOTE: alpha curriculum is disabled because HOI variants do not
            # carry alpha-morph data (all are alpha=1.0).  Enable only when
            # variant meshes with multiple alpha values are available.
            # Scale curriculum adjusts the scale-distribution range seen by
            # the reset event, but does NOT physically scale meshes (MJ
            # mocap bodies lack runtime scale).  It is still useful as
            # reward-precision randomization.
            self.curriculum["object_scale_curriculum"] = CurriculumTermCfg(
                func=ObjectScaleCurriculum,
                params={
                    "initial_scale_range": (0.95, 1.05),
                    "final_scale_range": (0.8, 1.2),
                    "start_step": 0,
                    "end_step": 1_000_000,
                },
            )


@dataclass(kw_only=True)
class G1PerceptiveHoiShadowingEnvCfg_PLAY(G1PerceptiveHoiShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveHoiShadowingSceneCfg = field(
        default_factory=lambda: perceptual_cfg.PerceptiveHoiShadowingSceneCfg(
            num_envs=1,
            env_spacing=2.5,
            entities=_make_hoi_entities(include_reference=True),
            sensors=_make_g1_hoi_scene_sensors(
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

        motion_reference_cfg = next(
            sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "motion_reference"
        )
        camera_cfg = next(sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "camera")

        # deactivate adaptive sampling and start from the 0.0s of the motion
        MOTION_NAME = list(motion_reference_cfg.motion_buffers.keys())[0]
        motion_reference_cfg.motion_buffers[MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        motion_reference_cfg.motion_buffers[MOTION_NAME].motion_bin_length_s = None
        motion_reference_cfg.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy = "independent"
        # BeyondConcatMotionAdaptiveWeighting requires _motion_bin_weights, which is only created when motion_bin_length_s is set
        self.curriculum["beyond_adaptive_sampling"] = None
        self.events["bin_fail_counter_smoothing"] = None
        # self.scene.motion_reference.motion_buffers[MOTION_NAME].path = (
        #     "/localhdd/Datasets/NoKov-Marslab-Motions-instinctnpz/20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall"
        # )
        camera_cfg.debug_vis = True
        self.observations["policy"].terms["depth_image"].params["debug_vis"] = True

        # change reset robot event with more pitch_down randomization (since the robot is facing -y axis)
        # self.events.reset_robot.params["randomize_pose_range"]["roll"] = (0.0, 0.6)

        # remove some terimation terms
        self.terminations["base_pos_too_far"] = None
        self.terminations["base_pg_too_far"] = None
        self.terminations["link_pos_too_far"] = None
        self.terminations["out_of_border"] = None

        # put the reference in scene and move the robot elsewhere and visualize the reference
        self.events["reset_robot"].params["position_offset"] = [0.0, 0.0, 0.0]
        motion_reference_cfg.visualizing_robot_offset = (0.0, 0.0, 0.0)
        self.viewer.entity_name = "robot_reference"

        # remove some randomizations (robot DR)
        self.events["add_joint_default_pos"] = None
        self.events["base_com"] = None
        self.events["physics_material"] = None

        # remove object DR events
        for key in list(self.events.keys()):
            if key.startswith("object_friction_") or key.startswith("object_com_"):
                self.events[key] = None
        self.events["reset_robot"].params["randomize_pose_range"]["x"] = [0.0] * 2  # (+-0.6)
        self.events["reset_robot"].params["randomize_pose_range"]["y"] = [0.0] * 2  # (+-0.6)
        self.events["reset_robot"].params["randomize_pose_range"]["z"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["roll"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["pitch"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["yaw"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["x"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["y"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["z"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["roll"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["pitch"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["yaw"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_joint_pos_range"] = (0.0, 0.0)

        # add some additional monitor terms
        self.monitors["shadowing_position_stats"] = MonitorTermCfg(
            func=ShadowingBasePosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
            ),
        )
        self.monitors["right_ankle_pitch_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_ankle_pitch.*"),
            ),
        )
        self.monitors["left_ankle_pitch_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_ankle_pitch.*"),
            ),
        )
        self.monitors["right_knee_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_knee.*"),
            ),
        )
        self.monitors["left_knee_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_knee.*"),
            ),
        )
