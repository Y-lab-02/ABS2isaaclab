# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from dataclasses import MISSING
import torch
import numpy as np

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.assets import (
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.sensors import CameraCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RecorderTermCfg as RecTerm
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderManagerBaseCfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdplo
import isaaclab_tasks.manager_based.navigation.mdp as mdpna
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


##
# Pre-defined configs
##

from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip
from isaaclab.terrains.config.rough import GO1_TERRAINS_CFG  # isort: skip


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    pose_command = mdpna.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(10.0, 10.0),
        object_avoid = True,
        object_nums = 8,
        debug_vis=True,
        ranges=mdpna.UniformPose2dCommandCfg.Ranges(pos_x=(1.5, 7.5), pos_y=(-2.0, 2.0), heading=(-0.3, 0.3)),
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        contact = ObsTerm(func=mdplo.obs_contact,
                          params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
                          )
        # base_lin_vel = ObsTerm(func=mdplo.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdplo.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdplo.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_command = ObsTerm(func=mdplo.generated_commands, params={"command_name": "pose_command"})
        joint_pos = ObsTerm(func=mdplo.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdplo.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdplo.last_action)
        obs_timer_left = ObsTerm(func=mdplo.obs_time_out)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


    # rewards for navigation
    reach_pos_target_soft = RewTerm(
        func=mdplo.reach_pos_target_soft,
        weight=50.,
        params={"duration": 2.0, "position_target_sigma_soft": 2.0, "command_name": "pose_command"},
    )
        
    reach_pos_target_tight = RewTerm(
        func=mdplo.reach_pos_target_tight,
        weight=50,
        params={"duration": 1.0, "position_target_sigma_tight": 0.5, "command_name": "pose_command"},
    )
    
    reach_heading_target = RewTerm(
        func=mdplo.reach_heading_target,
        weight=50,
        params={"duration": 2.0, "near_goal_threshold": 2.0, "heading_target_sigma": 1.0, "command_name": "pose_command"},
    )

    velo_dir = RewTerm(
        func=mdplo.velo_dir,
        weight=45,
        params={"dist_threshold": 0.5, "velo_normalization": 4.5, "command_name": "pose_command"},
    )

    # penalties
    termination = RewTerm(
        func=mdplo.termination,
        weight=-100,
        params={"dist_threshold": 0.5, "duration": 1.0, "command_name": "pose_command"},
    )
    stand_still_pos = RewTerm(
        func=mdplo.stand_still_pose,
        weight=-0.5,
        params={"dist_threshold": 0.5, "duration": 1.0, "command_name": "pose_command"},
    )

    fly = RewTerm(
        func=mdplo.fly,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    nomove = RewTerm(
        func=mdplo.nomove,
        weight=-0.2,
        params={"dist_threshold": 2.0, "command_name": "pose_command"},
    )
  
    # due to the usage of implicit autuator, this reward applied_torque_limits is currently unworkable
    # applied_torque_limits = RewTerm(func=mdplo.applied_torque_limits, weight=-20.0) 
    
    lin_vel_z_l2 = RewTerm(func=mdplo.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdplo.ang_vel_xy_l2, weight=-0.0005)
    flat_orientation_l2 = RewTerm(func=mdplo.flat_orientation_l2, weight=-0.2)
    dof_torques_l2 = RewTerm(func=mdplo.joint_torques_l2, weight=-0.0002)
    joint_vel_l2 = RewTerm(func=mdplo.joint_vel_l2, weight=-0.00005) 
    joint_vel_limits = RewTerm(func=mdplo.joint_vel_limits, weight=-0.2, params={"soft_ratio": 0.9})
    joint_pos_limits =  RewTerm(func=mdplo.joint_pos_limits, weight=-0.2)
    dof_acc_l2 = RewTerm(func=mdplo.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdplo.action_rate_l2, weight=-0.01)

    feet_air_time = RewTerm(
        func=mdplo.feet_air_time,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "pose_command",
            "threshold": 0.5,
            "give_reward_threshold": 0.5
        },
    )
    undesired_contacts = RewTerm(
        func=mdplo.undesired_contacts,
        weight=-100.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]), "threshold": 0.1},
    )

    feet_slide = RewTerm(
        func=mdplo.feet_slide,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), 
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )

    feet_collision = RewTerm(
        func=mdplo.feet_collision,
        weight=-100.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdplo.pos_time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdplo.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdplo.terrain_levels_pos,
        params={"position_target_sigma_soft": 2.0, "position_target_sigma_tight": 0.5},
        ) 




@configclass
class UnitreeGo1FlatPosEnvCfg(LocomotionVelocityRoughEnvCfg):

    commands: CommandsCfg = CommandsCfg()
 
    observations: ObservationsCfg = ObservationsCfg()

    rewards: RewardsCfg = RewardsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 9.0

        # randomlization for timer       
        self.randomize_timer_minus = 2.0

        self.train_test_ra = False

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        self.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO1_TERRAINS_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

        # objects 
        # object collection
        object_cfgs = {  
            f"object_{i}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/object_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=0.4,
                    height=1,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(20.0+5.0*i, 20.0+5.0*i, 0)),
            )
            for i in range(8)
        }

        self.scene.object_collection = RigidObjectCollectionCfg(rigid_objects=object_cfgs)

        # simulated sensor observation

        self.observations.policy.objects = ObsTerm(
            func=mdplo.object_detection,
            params={"num_objects": 8, "cylinder_radius": 0.4, "theta_start": - np.pi/4, "theta_end": np.pi/4 + 0.0001,
                    "theta_step": np.pi/20, "sensor_x0": -0.05, "sensor_y0": 0, "log2": True, "min_dist": 0.1, "max_dist": 6},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        
        # no height scan
        self.scene.height_scanner = None
        # no terrain curriculum
        # self.curriculum.terrain_levels = None

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.events.reset_objects = EventTerm( 
            func=mdplo.reset_root_state_uniform_pos,
            mode="reset",
            params={
                "pose_range": {"x": (-3.0, 8.0), "y": (-2.5, 2.5)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "asset_cfg": SceneEntityCfg(name='object_collection'),
            },
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"

        self.viewer = ViewerCfg(
            eye=(7.0, -7.0, 6.0),     # 摄像头位于右下上方
            lookat=(4.0, 0.0, 0.5),   # 视线指向场地中心略偏下方
            resolution=(1280, 720),      # 分辨率
        )


class UnitreeGo1FlatPosEnvCfg_PLAY(UnitreeGo1FlatPosEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


        #change parameters for testing

        object_cfgs = {  
            f"object_{i}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/object_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=0.4,
                    height=1,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True,kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000),
                    activate_contact_sensors=True,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(20.0+5.0*i, 20.0+5.0*i, 0)),
            )
            for i in range(8)
        }
        object_cfgs["object_0"].init_state = RigidObjectCfg.InitialStateCfg(pos=(6.5, -1.46, 0))
        object_cfgs["object_1"].init_state = RigidObjectCfg.InitialStateCfg(pos=(4.12, 0.51, 0))
        object_cfgs["object_2"].init_state = RigidObjectCfg.InitialStateCfg(pos=(3.53, -1.11, 0))
        object_cfgs["object_3"].init_state = RigidObjectCfg.InitialStateCfg(pos=(5.53, 1.64, 0))
        object_cfgs["object_4"].init_state = RigidObjectCfg.InitialStateCfg(pos=(6.90, -0.73, 0))
        object_cfgs["object_5"].init_state = RigidObjectCfg.InitialStateCfg(pos=(5.91, 1.51, 0))
        object_cfgs["object_6"].init_state = RigidObjectCfg.InitialStateCfg(pos=(2.05, 0.33, 0))
        object_cfgs["object_7"].init_state = RigidObjectCfg.InitialStateCfg(pos=(2.09, 0.30, 0))
      

        self.scene.object_collection = RigidObjectCollectionCfg(rigid_objects=object_cfgs)

        self.events.reset_objects = None









        


      


        

