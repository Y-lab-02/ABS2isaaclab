# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLPosEnv, ManagerBasedRLEnv


def reach_pos_target_soft(
    env: ManagerBasedRLPosEnv, command_name: str,  duration: float, position_target_sigma_soft: float
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    mask = env.timer_left <= duration
    reward = (1. /(1. + torch.square(distance / position_target_sigma_soft))) * mask/duration
    return reward


def reach_pos_target_tight(
    env: ManagerBasedRLPosEnv, command_name: str,  duration: float, position_target_sigma_tight: float
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    mask = env.timer_left <= duration
    reward = (1. /(1. + torch.square(distance / position_target_sigma_tight))) * mask/duration
    return reward

def reach_heading_target(
    env: ManagerBasedRLPosEnv, command_name: str,  duration: float, near_goal_threshold: float, heading_target_sigma: float
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    near_goal = distance < near_goal_threshold
    angle_diff = torch.abs(command[:, 3])
    mask = env.timer_left <= duration 
    heading_reward = (1. /(1. + torch.square(angle_diff / heading_target_sigma))) 
    reward = heading_reward * near_goal * mask/duration
    return reward

def velo_dir(
    env: ManagerBasedRLPosEnv, command_name: str, dist_threshold: float, velo_normalization: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_direction = wrap_to_pi(torch.atan2(command[:, 1], command[:, 0]))
    good_dir = torch.abs(target_direction) <  torch.deg2rad(torch.tensor(105.0))   # within around 105 degree
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    reward = asset.data.root_lin_vel_b[:,0].clip(min=0.0) * good_dir * (distance>dist_threshold)/velo_normalization \
               + 1.0 * (distance<dist_threshold)

    return reward

def stand_still_pose(
    env: ManagerBasedRLPosEnv, command_name: str, dist_threshold: float, duration: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    target_single = torch.tensor([0., 0., 0., 0., 1.0, 1.0, 1.0, 1.0, -1.8, -1.8, -1.8, -1.8],device=env.device)
    good_pos = target_single.unsqueeze(0).expand(env.num_envs, -1)
    mask = env.timer_left <= duration
    reward = torch.sum(torch.abs(asset.data.joint_pos - good_pos), dim=1) * mask/duration * (distance < dist_threshold)

    return reward


def fly(
    env: ManagerBasedRLPosEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    Contactsensor = env.scene.sensors[sensor_cfg.name]
    reward = torch.sum(Contactsensor.data.current_contact_time[:,sensor_cfg.body_ids], dim=-1) <0.5 *1.0 *(env.episode_length_buf * env.step_dt >0.5)
    return reward

def termination(
    # Terminal penalty; 5x penalty for dying in the rew_duration near goal 
    env: ManagerBasedRLPosEnv, command_name: str, dist_threshold: float, duration: float ) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    mask = env.timer_left <= duration
    reward = env.termination_manager.terminated * (1.0 + 4.0 * mask/duration) * (distance <dist_threshold)
    return reward


def nomove(
    env: ManagerBasedRLPosEnv, command_name: str, dist_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    asset = env.scene[asset_cfg.name]
    static = torch.logical_and(torch.norm(asset.data.root_lin_vel_b[:,:2], dim=-1) < 0.1, torch.abs(asset.data.root_ang_vel_b[:,2]) < 0.1)
    target_direction = wrap_to_pi(torch.atan2(command[:, 1], command[:, 0]))
    bad_dir = torch.abs(target_direction) > 1.83   # within around 105 degree

    reward = static * bad_dir * 1.0 * (distance > dist_threshold)
    return reward

def feet_collision( env: ManagerBasedRLPosEnv, sensor_cfg: SceneEntityCfg)-> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_hor = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2].norm(dim=-1)
    contact_force_ver = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    foot_hor_col = torch.any(contact_force_hor > 2 * contact_force_ver + 10.0, dim=-1)
    return foot_hor_col.float() * 1.0


# for recovery policy

def walkback( 
        env: ManagerBasedRLEnv, command_name: str, walkback_sigma:float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    lin_ver_error = torch.square((asset.data.root_com_lin_vel_b[:,0]-command[:,0]))+ \
        torch.square(asset.data.root_com_lin_vel_b[:,1]-command[:,1])
    
    return torch.exp(-lin_ver_error/walkback_sigma)


def posture(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:
    
    asset = env.scene[asset_cfg.name]
    target_single = torch.tensor([0., 0., 0., 0., 1.0, 1.0, 1.0, 1.0, -1.8, -1.8, -1.8, -1.8],device=env.device)
    good_pos = target_single.unsqueeze(0).expand(env.num_envs, -1)
    reward = torch.sum(torch.abs(asset.data.joint_pos - good_pos), dim=1)
    
    return reward

def yaw_rate(
        env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:
    
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    reward = torch.square(asset.data.root_com_ang_vel_b[:,2]-command[:,2])
    return reward



