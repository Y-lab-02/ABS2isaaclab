# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv,ManagerBasedRLPosEnv


def pos_time_out(env: ManagerBasedRLPosEnv) -> torch.Tensor:
    """Terminate used for random timeer."""
    return env.timer_left <= 0

def obs_time_out(env: ManagerBasedRLPosEnv) -> torch.Tensor:
    return env.timer_left.unsqueeze(-1)/ env.cfg.episode_length_s

def obs_contact(env: ManagerBasedRLPosEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    Contactsensor = env.scene.sensors[sensor_cfg.name]
    return (Contactsensor .data.current_contact_time[:, sensor_cfg.body_ids] > 0).float() *2.0 - 1.0


def object_detection(env: ManagerBasedRLPosEnv,num_objects: int, cylinder_radius: float, theta_start: float, 
                     theta_end: float, theta_step: float, sensor_x0: float, sensor_y0: float, 
                     log2: bool, min_dist: float, max_dist: float) -> torch.Tensor:
    """concatenate all sensor data to one observation."""

    robot_cfg = SceneEntityCfg(f"robot")
    robot = env.scene[robot_cfg.name]
    robot_yaw_only = math_utils.yaw_quat(robot.data.root_quat_w)

    object_pos = robot.data.root_pos_w.clone().unsqueeze(1) # dimension of object_pos now is [num_env, 1, 3]

    ray2d_thetas = torch.arange(start = theta_start, end = theta_end, step = theta_step, device = env.device).repeat(env.num_envs,1)
    ray2d_obs = torch.zeros_like(ray2d_thetas)
    ray2d_obs[:] = 99999.9

    for i in range(num_objects):

        object_cfg = SceneEntityCfg(name="object_collection")
        object_collections = env.scene[object_cfg.name]
        object_pos_relative = object_collections.data.object_pos_w[:,i,:] - robot.data.root_pos_w

        if i == 0:
            object_pos[:,i,:] = object_pos_relative
        else:
            object_pos = torch.cat((object_pos,object_pos_relative.unsqueeze(1)), dim=1)

        object_pos[:,i,:] = math_utils.quat_rotate_inverse(robot_yaw_only, object_pos[:,i,:])

        this_ray2d_obs = math_utils.circle_ray_query(sensor_x0, sensor_y0, ray2d_thetas,object_pos[:,i,0:2], radius= cylinder_radius, min_= min_dist, max_= max_dist)
        ray2d_obs = torch.minimum(ray2d_obs, this_ray2d_obs)

    if log2 == True:
        ray2d_ = torch.log2(ray2d_obs)
    else:
        ray2d_ = ray2d_obs
    
    return ray2d_
    


def reset_root_state_uniform_pos(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    
    # extract the used quantities (to enable type-hinting)
 
    asset = env.scene[asset_cfg.name]
    robot = env.scene[robot_cfg.name]
    
    # get default root state
    root_states = asset.data.default_object_state[env_ids].clone()
    root_states_robot = robot.data.default_root_state[env_ids].clone()
    root_states[:,:,0:3]=root_states_robot[:,None,0:3]

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids),asset.num_objects,6), device=asset.device)
    xs = rand_samples[:, :, 0]
    ys = rand_samples[:, :, 1]
    too_near = (xs**2 + ys**2) < 1.1**2
    rand_samples[:, :, 0] += too_near * torch.sign(xs) * 0.9
    rand_samples[:, :, 1] += too_near * torch.sign(ys) * 0.9        

    positions = root_states[:,:, 0:3] + env.scene.env_origins[env_ids,None,:] + rand_samples[:,:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:,:, 3], rand_samples[:,:, 4], rand_samples[:,:,5])
    orientations = math_utils.quat_mul(root_states[:,:,3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_vel_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids),asset.num_objects, 6), device=asset.device)

    velocities = root_states[:,:,7:13] + rand_vel_samples

    # set into the physics simulation
    asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_object_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_uniform_pos_play(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    asset = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_object_state[env_ids].clone()
    # poses
    positions = root_states[:,:, 0:3] + env.scene.env_origins[env_ids,None,:]
    orientations = root_states[:,:, 3:7]
    # velocities
    # set into the physics simulation
    asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)

def reset_root_state_uniform_pos_play_train_ra(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    init_obst_xy = [-3., 8., -2.5, 2.5]
    
):
    asset = env.scene[asset_cfg.name]   
    

    root_states = asset.data.default_object_state[env_ids].clone()
    root_states[:,:,0:2]= 0 # [ids,8,2]


    moves = env.objects_local_pos.clone()

    ids_to_update = env_ids[env.resample_obj_pos[env_ids]]   

    # only sample the objects that need to be reset
    moves[ids_to_update, :, 0] = moves[ids_to_update, :, 0].uniform_(init_obst_xy[0], init_obst_xy[1])
    moves[ids_to_update, :, 1] = moves[ids_to_update, :, 1].uniform_(init_obst_xy[2], init_obst_xy[3])

    moves[moves == 0.] = 0.01

    xs = moves[:, :, 0]
    ys = moves[:, :, 1]
    too_near = (xs**2 + ys**2) < 1.1**2
    moves[:, :, 0] += too_near * torch.sign(xs) * 0.9
    moves[:, :, 1] += too_near * torch.sign(ys) * 0.9   

    env.objects_local_pos = moves
    env.resample_obj_pos[ids_to_update] = False
    
    positions = root_states[:,:, 0:3] + env.scene.env_origins[env_ids,None,:]

    positions[:,:,:2] += moves[env_ids,:,:]
    
    orientations = root_states[:,:, 3:7]
    # velocities
    # set into the physics simulation
    asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)

def reset_root_state_uniform_pos_play_test_ra_pos(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg,
    xmin: float, xmax: float, ymin: float, ymax: float,  n_obj: int,
    safedist = 0.75
):
    assert ymax - ymin > 2*safedist
    assert xmax > xmin + 0.1

    asset = env.scene[asset_cfg.name]   
    root_states = asset.data.default_object_state[env_ids].clone()

    obj_pos_sampled = torch.zeros(len(env_ids),n_obj,2).to("cuda")# [ids,8,2]

    # step 1
    nodes = torch.zeros(len(env_ids),4,2).to("cuda")
    nodes[:,0,0] = xmin # x0 = xmin
    nodes[:,1,0] = xmin * 0.67 + xmax * 0.33 # x1
    nodes[:,2,0] = xmin * 0.33 + xmax * 0.67 # x2
    nodes[:,3,0] = xmax # x3 = xmax
    nodes[:,0,1] = ymin * 0.5 + ymax * 0.5 # y0 in middle
    nodes[:,3,1] = ymin * 0.5 + ymax * 0.5 # y3 in middle
    nodes[:,1:3,1] = nodes[:,1:3,1].uniform_(ymin+safedist, ymax-safedist)

    # step 2
    A = torch.stack([nodes[:,:,0]**3, nodes[:,:,0]**2, nodes[:,:,0], torch.ones_like(nodes[:,:,0])], dim=2)
    coefficients = torch.linalg.lstsq(A, nodes[:,:,1].unsqueeze(2)).solution  # (n,4,1)

    # step 3
    obj_pos_sampled[:,:,0] = obj_pos_sampled[:,:,0].uniform_(xmin, xmax) #(n,4)
    obj_pos_sampled[:,:,1] = obj_pos_sampled[:,:,1].uniform_(ymin, ymax)

    # step 4
    y_curve = coefficients[:,0] * obj_pos_sampled[:,:,0]**3 + coefficients[:,1] * obj_pos_sampled[:,:,0]**2 \
                + coefficients[:,2] * obj_pos_sampled[:,:,0] + coefficients[:,3] #(n,4)
    diffy = obj_pos_sampled[:,:,1] - y_curve
    diffy[diffy==0.] = 0.001
    obj_pos_sampled[:,:,1] = obj_pos_sampled[:,:,1] * (torch.abs(diffy)>=safedist) + (torch.sign(diffy)*safedist + y_curve) * (torch.abs(diffy)<safedist)
    
    obj_pos_sampled[obj_pos_sampled == 0.] = 0.01
    xs = obj_pos_sampled[:, :, 0]
    ys = obj_pos_sampled[:, :, 1]

    too_near = ((xs)**2 + (ys)**2) < 1.1**2

    obj_pos_sampled[:, :, 0] += too_near * torch.sign(xs) * 0.9
    obj_pos_sampled[:, :, 1] += too_near * torch.sign(ys) * 0.9  


    obj_pos_sampled +=  env.scene.env_origins[env_ids,None,:2]

    obj_pos_rest = root_states[:,:, 2:7]
    
    asset.write_object_pose_to_sim(torch.cat([obj_pos_sampled, obj_pos_rest], dim=-1), env_ids=env_ids)
    
    
def reset_joints_rec(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range_scale: tuple[float, float],
    velocity_range_uniform: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids, asset_cfg.joint_ids].clone()

    # sample position and velocity
    joint_pos *= math_utils.sample_uniform(*position_range_scale, joint_pos.shape, joint_pos.device)
    joint_vel = math_utils.sample_uniform(*velocity_range_uniform,joint_vel.shape,joint_pos.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=asset_cfg.joint_ids,
    )




