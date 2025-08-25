# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaacsim.core.version import get_version

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.managers import SceneEntityCfg

from isaaclab.ui.widgets import ManagerLiveVisualizer

from .common import VecEnvStepReturn
from .manager_based_rl_env import ManagerBasedRLEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ManagerBasedRLPosEnv(ManagerBasedRLEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """
    
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """

         # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)

        print("[INFO]: Joint names: ", self.scene["robot"].joint_names)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager

        # initial timer
        # put it here because timer needs to know the information of num_envs which is known after super().__init__(cfg=cfg) , 
        # but in super().__init__(cfg=cfg) the observation manager needs to know timer_left
        self.timer_left = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * self.cfg.episode_length_s
        self.timer_left.uniform_(self.cfg.episode_length_s - self.cfg.randomize_timer_minus, self.cfg.episode_length_s)  

        self.ra_obs = torch.zeros((self.num_envs, 19), dtype=torch.float, device=self.device)

        self.objects_local_pos = torch.zeros((self.num_envs,8,2),dtype=torch.float, device=self.device)

        self.resample_obj_pos = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device )
       

        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")


    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.timer_left -= self.step_dt
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)


        # record observations data for RA_training

        root_lin_vel = self.scene["robot"].data.root_com_lin_vel_b  # shape: (num_envs, 3)

        root_ang_vel = self.scene["robot"].data.root_com_ang_vel_b  # shape: (num_envs, 3)

        command_ra = self.command_manager.get_command("pose_command")[:, :2] # shape: (num_envs, 2)

        obs_dict = self.observation_manager.compute_group("policy")
        objects_obs = obs_dict[:, 51:62] # shape: (num_envs, 11)

        self.ra_obs = torch.cat([root_lin_vel, root_ang_vel, command_ra, objects_obs], dim=-1)

        # record and compute gs, ls and dones for RA_training

        collision = self.termination_manager.get_term("base_contact")

        sensor_cfg = SceneEntityCfg(name="contact_forces", body_names=".*_foot")

        contact_sensor = self.scene.sensors[sensor_cfg.name]
        contact_force_hor = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2].norm(dim=-1)
        contact_force_ver = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        foot_hor_col = torch.any(contact_force_hor > 2 * contact_force_ver + 10.0, dim=-1)
        
        collision = torch.logical_or(collision, foot_hor_col) 

        robot_pos = self.scene["robot"].data.root_pos_w  # [num_env, 3]
        object_pos_w = self.scene["object_collection"].data.object_pos_w  # [num_env, num_objects, 3]
        object_pos_relative = object_pos_w - robot_pos.unsqueeze(1)  # [num_env, num_objects, 3]
        object_dist = object_pos_relative.norm(dim=-1)  # [num_env, num_objects]

        _near_obj = torch.any(object_dist<0.95, dim=-1)

        _near_obj = torch.logical_and(_near_obj, self.scene["robot"].data.root_com_lin_vel_b[:,:2].norm(dim=-1) > 0.5)

        collision = torch.logical_and(collision, _near_obj)  # filter the weird collisions from simulator that cannot be explained

        if self.cfg.train_test_ra == True:
            self.reset_buf = torch.logical_or(collision, self.reset_time_outs)

        self.collision = collision

        self.gs = collision.float() * 2 - 1  # 1 for collision, -1 for not collision

        self.ls = torch.tanh( torch.log2(torch.norm(command_ra[:,:], dim=-1) / 0.65 + 1e-8) )

        self.dones = self.reset_buf

        self.foot_hor_col = foot_hor_col

        



        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    """
    Helper functions.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # reset timer
        self.timer_left[env_ids] = -self.cfg.randomize_timer_minus * torch.rand(len(env_ids), device=self.device) + self.cfg.episode_length_s  
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
