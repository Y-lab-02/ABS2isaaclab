# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from dataclasses import MISSING
import numpy as np

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdplo
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
    
    velocity_command = mdplo.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(100.0, 100.0),
        debug_vis=True,
        ranges=mdplo.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 3.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-3.0, 3.0)
        ),
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
        velocity_command = ObsTerm(func=mdplo.generated_commands, params={"command_name": "velocity_command"})
        joint_pos = ObsTerm(func=mdplo.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdplo.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdplo.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # rewards for recovery

    walkback = RewTerm(
        func=mdplo.walkback,
        weight=30,
        params={"walkback_sigma": 0.5**2,  "command_name": "velocity_command"},
    )

    posture = RewTerm(
        func=mdplo.posture,
        weight=-2
        )
    
    yaw_rate = RewTerm(
        func=mdplo.yaw_rate,
        weight=-5.0,
        params={"command_name": "velocity_command"},
    )

    alive = RewTerm(
        func=mdplo.is_alive,
        weight=20
        )

    # penalties
    termination = RewTerm(
        func=mdplo.is_terminated,
        weight=-100
    )

    lin_vel_z_l2 = RewTerm(func=mdplo.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdplo.ang_vel_xy_l2, weight=-0.0005)
    flat_orientation_l2 = RewTerm(func=mdplo.flat_orientation_l2, weight=-0.2)
    dof_torques_l2 = RewTerm(func=mdplo.joint_torques_l2, weight=-0.0002)
    joint_vel_l2 = RewTerm(func=mdplo.joint_vel_l2, weight=-0.00005) 
    joint_vel_limits = RewTerm(func=mdplo.joint_vel_limits, weight=-0.2, params={"soft_ratio": 0.9})
    joint_pos_limits =  RewTerm(func=mdplo.joint_pos_limits, weight=-0.2)
    dof_acc_l2 = RewTerm(func=mdplo.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdplo.action_rate_l2, weight=-0.01)
    
    # due to the usage of implicit autuator, this reward applied_torque_limits is currently unworkable
    # applied_torque_limits = RewTerm(func=mdplo.applied_torque_limits, weight=-20.0) 

    feet_air_time = RewTerm(
        func=mdplo.feet_air_time,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "velocity_command",
            "threshold": 0.5,
            "give_reward_threshold": 0.5,

        },
    )
    feet_slide = RewTerm(
        func=mdplo.feet_slide,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), 
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdplo.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdplo.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdplo.terrain_levels_rec,
        params={"walkback_sigma": 0.5**2},
        )


@configclass
class UnitreeGo1FlatRecEnvCfg(LocomotionVelocityRoughEnvCfg):

    commands: CommandsCfg = CommandsCfg()
 
    observations: ObservationsCfg = ObservationsCfg()

    rewards: RewardsCfg = RewardsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 2.0

        # randomlization for timer       
        self.randomize_timer_minus = 0.


        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        self.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO1_TERRAINS_CFG,
        max_init_terrain_level=5,
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
        self.events.base_external_force_torque.params["torque_range"] = (-3.,3.)
        self.events.reset_base.params = {
            "pose_range": {"x": (0., 0.), "y": (0., 0.),"roll": (-3.14/6, 3.14/6),"y": (-3.14/6, 3.14/6), "yaw": (0., 0.)},
            "velocity_range": {
                "x": (-0.5, 5.5), # fast forward
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.reset_robot_joints = EventTerm(
            func=mdplo.reset_joints_rec,
            mode="reset",
            params={
                "position_range_scale": (0.75, 1.25),
                "velocity_range_uniform": (-8., 8.),
            },
        )
        

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"


class UnitreeGo1FlatRecEnvCfg_PLAY(UnitreeGo1FlatRecEnvCfg):
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

