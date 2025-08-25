# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import time
import torch
import torch.nn as nn
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdplo
import isaaclab_tasks.manager_based.navigation.mdp as mdpna
from isaaclab.managers import SceneEntityCfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx, RslRlPpoActorCriticCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

def get_pos_integral(twist, tau):
    # taylored as approximation
    vx, vy, wz = twist[...,0], twist[...,1], twist[...,2]
    theta = wz * tau
    x = vx * tau - 0.5 * vy * wz * tau * tau
    y = vy * tau + 0.5 * vx * wz * tau * tau
    return x, y, theta

def _clip_grad(grad, thres):
    grad_norms = grad.norm(p=2, dim=-1).unsqueeze(-1) #(n,1)
    return grad * thres / torch.maximum(grad_norms, thres*torch.ones_like(grad_norms))



def main():
    """Play with RSL-RL agent."""

    train_ra = False

    test_ra = False

    test_pos = True




    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)




    if test_ra  or train_ra  or test_pos:
        resume_path =os.path.join('logs', "test_policy", 'agile_model_650' + '.pt')
        resume_path = os.path.abspath(resume_path)
    


    log_dir = os.path.dirname(resume_path)



    # change initializaitions

    # env_cfg.scene.env_spacing=20

    # env_cfg.episode_length_s = 6.0

    # env_cfg.scene.terrain.terrain_type = "plane"
    # env_cfg.scene.terrain.terrain_generator = None

    env_cfg.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.
    env_cfg.scene.terrain.terrain_generator.sub_terrains["rough"].proportion = 0.
    env_cfg.scene.terrain.terrain_generator.sub_terrains["low_obst"].proportion = 1.
    env_cfg.scene.terrain.terrain_generator.sub_terrains["low_obst"].num_obstacles = 150
    env_cfg.scene.terrain.terrain_generator.size = (20.0,20.0)
    env_cfg.scene.terrain.terrain_generator.num_rows = 1
    env_cfg.scene.terrain.terrain_generator.num_cols = 1



    env_cfg.viewer.eye = (12.0, -3.0, 4.0) 
    env_cfg.viewer.lookat=(6.0, 0.0, 0.5)





    env_cfg.curriculum.terrain_levels = None

    

    if train_ra or test_ra  or test_pos:

        env_cfg.randomize_timer_minus = 0

        env_cfg.observations.policy.base_ang_vel.noise = None 
        env_cfg.observations.policy.projected_gravity.noise = None
        env_cfg.observations.policy.joint_pos.noise = None 
        env_cfg.observations.policy.joint_vel.noise = None 
        env_cfg.observations.policy.objects.noise = None

        env_cfg.events.add_base_mass = None

        env_cfg.events.reset_base.params = {
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            }
        
        env_cfg.terminations.base_contact.params["sensor_cfg"].body_names = ["trunk","FR_thigh", "FL_thigh", "FR_calf" , "FL_calf"]

        env_cfg.events.reset_objects = EventTerm( 
                func=mdplo.reset_root_state_uniform_pos_play_train_ra,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg(name='object_collection')},
            )
        
        # env_cfg.events.reset_objects = EventTerm( 
        #         func=mdplo.reset_root_state_uniform_pos_play,
        #         mode="reset",
        #         params={"asset_cfg": SceneEntityCfg(name='object_collection')},
        #     )
        
        env_cfg.commands.pose_command.ranges.pos_x = (6.0, 7.5)
        env_cfg.commands.pose_command.ranges.pos_y = (-1.5, 1.5)
        env_cfg.commands.pose_command.ranges.heading = (0.0, 0.0)

        env_cfg.train_test_ra = True

    if test_ra  or test_pos:
        # env_cfg.events.reset_objects = EventTerm( 
        #         func=mdplo.reset_root_state_uniform_pos_play_test_ra_pos,
        #         mode="reset",
        #         params={"asset_cfg": SceneEntityCfg(name='object_collection'),
        #                 "xmin": 1.5, "xmax": 7., "ymin": -2., "ymax": 2.,  "n_obj": 8,
        #                 },
        #     )    

        env_cfg.events.reset_objects = EventTerm( 
                func=mdplo.reset_root_state_uniform_pos_play,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg(name='object_collection')},
            )   
        env_cfg.commands.pose_command.ranges.pos_x = (7.5, 7.5)
        env_cfg.commands.pose_command.ranges.pos_y = (0.0, 0.0)
        env_cfg.commands.pose_command.ranges.heading = (0.0, 0.0) 

    # create isaac environment
    
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    isaaclab_env = env.unwrapped
    isaaclab_env.render_mode = "rgb_array"

    # network for RA
    ra_vf = nn.Sequential(nn.Linear(19,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1), nn.Tanh())
    ra_vf.to(isaaclab_env.device)
    optimizer = torch.optim.SGD(ra_vf.parameters(), lr=0.002, momentum=0.0)

    # load RA network for training

    if train_ra:
        best_metric = 999.
        ra_path = os.path.join('logs', "RA", "unitree_go1_flat_pos", 'exported', 'RA', 'policy_ra' + '.pt')
        ra_path = os.path.abspath(ra_path)
        if os.path.isfile(ra_path):
            _load = input('load existing value? y/n\n')
            if _load != 'n':
                ra_vf = torch.load(ra_path)
                print('loaded value from', ra_path)
    
    standard_raobs_init = torch.Tensor([[0,0,0,0,0,0,6.,0]+[0.,0.,0.,1.,1.,2.,2.,1.,0.,0.,0.]]).to(isaaclab_env.device)
    standard_raobs_die = torch.Tensor([[5.,0,0,0,0,0,6.,0]+[-2.5]*11]).to(isaaclab_env.device)
    standard_raobs_turn = torch.Tensor([[0,0,0,0,0,2.0, 0.5,5.8]+[2.0]*6+[0.0]*5]).to(isaaclab_env.device)
    ra_obs = standard_raobs_init.clone().repeat(isaaclab_env.scene.num_envs,1)

    collision = torch.zeros(env.num_envs).to(isaaclab_env.device).bool()

    twist_eps = -0.15

    twist_tau = 0.05
    twist_lam = 10.
    twist_lr = 0.5
    twist_min = torch.tensor([-1.5,-0.3,-3.0]).cuda()
    twist_max = -twist_min

    queue_len = 1001
    batch_size = 200
    hindsight = 10
    s_queue = torch.zeros((queue_len,isaaclab_env.scene.num_envs,19), device = isaaclab_env.device, dtype=torch.float)
    g_queue = torch.zeros((queue_len,isaaclab_env.scene.num_envs), device = isaaclab_env.device, dtype=torch.float)
    g_hs_queue = g_queue.clone()
    g_hs_span = torch.zeros((2,isaaclab_env.scene.num_envs), device = isaaclab_env.device, dtype=torch.int) # start and end index of the latest finished episode
    l_queue = torch.zeros((queue_len,isaaclab_env.scene.num_envs), device = isaaclab_env.device, dtype=torch.float)
    done_queue = torch.zeros((queue_len,isaaclab_env.scene.num_envs), device = isaaclab_env.device, dtype=torch.bool)
    alive = torch.ones_like(isaaclab_env.termination_manager.compute())

    # ======= metrics begin =======
    total_n_collision, total_n_reach, total_n_timeout = 0, 0, 0
    total_n_episodic_recovery, total_n_episodic_recovery_success, total_n_episodic_recovery_fail = 0, 0, 0
    total_recovery_dist = 0
    total_recovery_timesteps = 0
    total_n_collision_when_ra_on, total_n_collision_when_ra_off = 0, 0

    episode_recovery_logging = torch.zeros(isaaclab_env.scene.num_envs).to(isaaclab_env.device).bool()
    current_recovery_status = torch.zeros(isaaclab_env.scene.num_envs).to(isaaclab_env.device).bool()

    total_n_done = 0
    total_episode = 0
    if train_ra  or test_ra  or test_pos:
        last_obs = isaaclab_env.observation_manager.compute_group("policy").clone()
        last_root_states = isaaclab_env.scene["robot"].data.root_state_w.clone() # in world frame
        last_position_targets = isaaclab_env.command_manager.get_command("pose_command")[:, :2].clone() # in robot b frame 

    episode_travel_dist = torch.zeros(isaaclab_env.scene.num_envs).to(isaaclab_env.device)
    episode_time = torch.zeros(isaaclab_env.scene.num_envs).to(isaaclab_env.device)

    episode_max_velo = torch.zeros(isaaclab_env.scene.num_envs).to(isaaclab_env.device)
    episode_max_velo_dist = 0
    episode_max_velo_dist_collision = 0
    episode_max_velo_dist_reach = 0
    episode_max_velo_dist_timeout = 0

    total_travel_dist = 0
    total_time = 0

    total_reach_dist = 0
    total_time = 0

    total_reach_dist = 0
    total_collision_dist = 0
    total_timeout_dist = 0

    total_reach_time = 0
    total_collision_time = 0
    total_timeout_time = 0
    # ======= metrics end =======

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)

        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    if test_ra :
        ra_path =  os.path.join('logs', "test_policy", 'policy_ra' + '.pt')
        ra_path = os.path.abspath(ra_path)
        ra_vf = torch.load(ra_path)
        print('loaded ra network from', ra_path)
        rec_policy_path = os.path.join('logs', "test_policy", 'recovery_model_680_jit' + '.pt')

        rec_policy = torch.jit.load(rec_policy_path).cuda()
        print('loaded recovery policy from',rec_policy_path)
        mode_running = True # if False: recovery


    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    
    timestep = 0


    # simulate environment
    while simulation_app.is_running():

        for i in range(300*int(isaaclab_env.max_episode_length)):
        # for i in range(600*int(isaaclab_env.max_episode_length)):
            start_time = time.time()
            current_recovery_status = torch.zeros(env.num_envs).to(env.device).bool()
            where_recovery = torch.zeros(env.num_envs).to(env.device).bool()

            if train_ra or test_ra  or test_pos :


                if train_ra:

                    if i % 1000 == 0 :
                        isaaclab_env.resample_obj_pos[:] = True

                        os.makedirs(os.path.dirname(ra_path), exist_ok=True)
                        torch.save(ra_vf, ra_path)    
                        print('\x1b[6;30;42m', 'saving ra model to', ra_path, '\x1b[0m' )
                
                    # run everything in inference mode
                    with torch.inference_mode():
                        # agent stepping
                        # actions = policy(obs)
                        actions = policy(obs.detach()) * alive.unsqueeze(1)
                        # env stepping
                        obs, _, _, _ = env.step(actions)

                
                if test_ra:

                    isaaclab_env.resample_obj_pos[:] = True
                   
                    # actions = policy(obs)
                    with torch.inference_mode():
                        actions = policy(obs.detach()) * alive.unsqueeze(1)
                    actions = actions.clone()
                    
                    v_pred = ra_vf(ra_obs)
                    start_v = ra_vf(standard_raobs_init).mean().item()
                    die_v = ra_vf(standard_raobs_die).mean().item() 
                    turn_v = ra_vf(standard_raobs_turn).mean().item()

                    recovery = (v_pred > -twist_eps).squeeze(-1)
                    where_recovery = torch.where(torch.logical_and(recovery, ~collision))[0]

                    if where_recovery.shape[0] > 0:
                        episode_recovery_logging[where_recovery] = True
                        current_recovery_status[where_recovery] = True

                        mode_running = False
                        twist_iter = torch.cat([isaaclab_env.scene["robot"].data.root_lin_vel_b[where_recovery, 0:2],isaaclab_env.scene["robot"].data.root_ang_vel_b[where_recovery, 2:3]], dim=-1)   
                        twist_iter.requires_grad=True
                        for _iter in range(10):
                        # for _iter in range(30):
                            twist_ra_obs = torch.cat([twist_iter[...,0:2], isaaclab_env.scene["robot"].data.root_lin_vel_b[where_recovery, 2:3],\
                                                       isaaclab_env.scene["robot"].data.root_ang_vel_b[where_recovery,0:2], twist_iter[...,2:3],\
                                                          isaaclab_env.command_manager.get_command("pose_command")[where_recovery, :2], isaaclab_env.observation_manager.compute_group("policy")[where_recovery, 51:62]],dim=-1)
                            x_iter, y_iter, _ = get_pos_integral(twist_iter, twist_tau)
                            ra_value = ra_vf(twist_ra_obs)
                            loss_separate = twist_lam * (ra_value + 2*twist_eps).clip(min=0).squeeze(-1) + 0.02*(((x_iter-isaaclab_env.command_manager.get_command("pose_command")[where_recovery, 0:1].squeeze(-1))**2) + ((y_iter-isaaclab_env.command_manager.get_command("pose_command")[where_recovery, 1:2].squeeze(-1))**2))
                            loss = loss_separate.sum()
                            loss.backward(retain_graph=True)
                            twist_iter.data = twist_iter.data - twist_lr * _clip_grad(twist_iter.grad.data, 1.0)
                            twist_iter.data = twist_iter.data.clip(min=twist_min, max=twist_max)
                            twist_iter.grad.zero_()

                        twist_iter = twist_iter.detach()
                        obs_rec = torch.cat((obs[where_recovery,:10], twist_iter, obs[where_recovery,14:50]), dim=-1)
                        actions[where_recovery] = rec_policy(obs_rec.detach())
                    else:
                        mode_running = True

                    with torch.inference_mode():
                        obs, _, _, _ = env.step(actions)
                    
                    obs = obs.detach().clone().requires_grad_(True)

                if test_pos:
                    with torch.inference_mode():
                        # agent stepping
                        actions = policy(obs)
                        # env stepping
                        obs, _, _, _ = env.step(actions)
                    

                # ra_obs = isaaclab_env.ra_obs.detach().clone().requires_grad_(True)
                ra_obs = isaaclab_env.ra_obs.clone()
                gs = isaaclab_env.gs
                ls = isaaclab_env.ls 
                dones = isaaclab_env.dones
                collision = isaaclab_env.collision

                # if isaaclab_env.foot_hor_col == True:
                #     _load = input('load existing value? y/n\n')
                #     print("[INFO] foot reason dones",isaaclab_env.foot_hor_col)

                where_done = torch.where(dones)[0]
                where_collision = torch.where(torch.logical_and(dones, collision))[0]
                distance_to_goal = torch.norm(last_position_targets[:,0:2], dim=-1) # when done is true, new env root states and position targets are updated, so we need to use last ones
                where_reach = torch.where(torch.logical_and(distance_to_goal < 0.65, torch.logical_and(dones, ~collision)))[0]
                where_timeout = torch.where(torch.logical_and(distance_to_goal >= 0.65, torch.logical_and(dones, ~collision)))[0]

                not_in_goal = torch.logical_and(distance_to_goal >= 0.65, ~dones)

                total_episode += where_done.shape[0]
                total_n_done += where_done.shape[0]
                total_n_collision += where_collision.shape[0]
                total_n_reach += where_reach.shape[0]
                total_n_timeout += where_timeout.shape[0]

                # dist and time
                if i > 0 and (not train_ra):
                    onestep_dist = torch.norm(isaaclab_env.scene["robot"].data.root_state_w[:,0:2] - last_root_states[:,0:2], dim=-1) * (not_in_goal).float()
                    episode_travel_dist += onestep_dist
                    episode_time += (- (obs[:,50] - last_obs[:,50]) * (not_in_goal).float()) * isaaclab_env.max_episode_length_s
                    episode_max_velo = torch.maximum(episode_max_velo, onestep_dist/0.02)
                    total_recovery_dist += onestep_dist[where_recovery].sum().item()
                    total_recovery_timesteps += where_recovery.shape[0]

                if where_done.shape[0] > 0 and (not train_ra):
                    # recovery
                    total_n_episodic_recovery += episode_recovery_logging[where_done].sum().item()
                    total_n_episodic_recovery_fail += (episode_recovery_logging[where_done] * collision[where_done]).sum().item()
                    total_n_episodic_recovery_success += (episode_recovery_logging[where_done] * ~collision[where_done]).sum().item()
                    episode_recovery_logging[where_done] = False

                    total_n_collision_when_ra_on += torch.logical_and(collision[where_done], current_recovery_status[where_done]).sum().item()
                    total_n_collision_when_ra_off += torch.logical_and(collision[where_done], ~current_recovery_status[where_done]).sum().item()

                    # max velocity in trajectory
                    episode_max_velo_dist += episode_max_velo[where_done].sum().item()
                    episode_max_velo_dist_collision += episode_max_velo[where_collision].sum().item()
                    episode_max_velo_dist_reach += episode_max_velo[where_reach].sum().item()
                    episode_max_velo_dist_timeout += episode_max_velo[where_timeout].sum().item()
                    episode_max_velo[where_done] = 0

                    # collision
                    collision_dist = episode_travel_dist[where_collision].sum().item()
                    collision_time = episode_time[where_collision].sum().item()
                    total_collision_dist += collision_dist
                    total_collision_time += collision_time
                    total_travel_dist += collision_dist
                    total_time += collision_time

                    # reach
                    reach_dist = episode_travel_dist[where_reach].sum().item()
                    reach_time = episode_time[where_reach].sum().item()
                    total_reach_dist += reach_dist
                    total_reach_time += reach_time
                    total_travel_dist += reach_dist
                    total_time += reach_time

                    # timeout
                    timeout_dist = episode_travel_dist[where_timeout].sum().item()
                    timeout_time = episode_time[where_timeout].sum().item()
                    total_timeout_dist += timeout_dist
                    total_timeout_time += timeout_time
                    total_travel_dist += timeout_dist
                    total_time += timeout_time

                    episode_time[where_done] = 0
                    episode_travel_dist[where_done] = 0
                    episode_time[where_collision] = 0
                    episode_travel_dist[where_collision] = 0

                    avg_collision_dist = total_collision_dist / (total_n_collision + 1e-8)
                    avg_collision_time = total_collision_time / (total_n_collision + 1e-8)
                    avg_reach_dist = total_reach_dist / (total_n_reach + 1e-8)
                    avg_reach_time = total_reach_time / (total_n_reach + 1e-8)
                    avg_timeout_dist = total_timeout_dist / (total_n_timeout + 1e-8)
                    avg_timeout_time = total_timeout_time / (total_n_timeout + 1e-8)

                    avg_total_dist = total_travel_dist / (total_episode + 1e-8)
                    avg_total_time = total_time / (total_episode + 1e-8)

                    avg_total_velocity = avg_total_dist / avg_total_time
                    avg_collision_velocity = avg_collision_dist / (avg_collision_time + 1e-8)
                    avg_reach_velocity = avg_reach_dist / (avg_reach_time + 1e-8)
                    avg_timeout_velocity = avg_timeout_dist / (avg_timeout_time + 1e-8)
                    avg_recovery_velocity = total_recovery_dist / ((total_recovery_timesteps + 1e-8)*0.02)

                if total_episode % 1 == 0 and total_episode > 0 and (not train_ra) and where_done.shape[0] > 0:
                    # _load = input('load existing value? y/n\n')
                    # print("[INFO] dones:", dones)
                    # print("[INFO] collision:", collision)
                    # print("[INFO] foot reason dones",isaaclab_env.foot_hor_col)

                    # print("[INFO] objects location",isaaclab_env.scene["object_collection"].data.object_link_pos_w[:,:,0:2]-isaaclab_env.scene.env_origins[:,None,0:2])
                    # print("[INFO] robot", isaaclab_env.scene["robot"].data.root_link_pos_w[:,0:2])



                    print("========= Episode {} =========".format(total_episode))
                    print('Total Episode:                         {}'.format(total_episode))
                    print('Total Collision:                       {}'.format(total_n_collision))
                    print('Total Reach:                           {}'.format(total_n_reach))
                    print('Total Timeout:                         {}'.format(total_n_timeout))
                    print('Total Collision + Reach + Timeout:      {}'.format(total_n_collision + total_n_reach + total_n_timeout))
                    print('Total Done:                            {}'.format(total_n_done))
                    print('Collision Rate:                        {:.2%}'.format(total_n_collision / total_episode))
                    print('Reach Rate:                            {:.2%}'.format(total_n_reach / total_episode))
                    print('Timeout Rate:                          {:.2%}'.format(total_n_timeout / total_episode))
                    print('Average Total Velocity:                {:.2f}'.format(avg_total_velocity))
                    print('Average Collision Velocity:            {:.2f}'.format(avg_collision_velocity))
                    print('Average Reach Velocity:                {:.2f}'.format(avg_reach_velocity))
                    print('Average Timeout Velocity:              {:.2f}'.format(avg_timeout_velocity))
                    print('Average Recovery Velocity:             {:.2f}'.format(avg_recovery_velocity))
                    print('Average in-trajectory Max Velocity:    {:.2f}'.format(episode_max_velo_dist / total_episode))
                    print('Average in-trajectory Max Velocity Collision: {:.2f}'.format(episode_max_velo_dist_collision / (total_n_collision + 1e-8)))
                    print('Average in-trajectory Max Velocity Reach: {:.2f}'.format(episode_max_velo_dist_reach / (total_n_reach + 1e-8)))
                    print('Average in-trajectory Max Velocity Timeout: {:.2f}'.format(episode_max_velo_dist_timeout / (total_n_timeout + 1e-8)))
                    # Recovery
                    print('Episode that activated recovery:         {}'.format(total_n_episodic_recovery))
                    print('Episode that activated recovery - safe:  {}'.format(total_n_episodic_recovery_success))
                    print('Episode that activated recovery - collision:    {}'.format(total_n_episodic_recovery_fail))
                    print('Episode that did not activate recovery - collision: {}'.format(total_n_collision - total_n_episodic_recovery_fail))
                    print('Episodic recovery activation rate:          {:.2%}'.format(total_n_episodic_recovery / total_episode))
                    print('Episodic recovery success rate (end up safe): {:.2%}'.format(total_n_episodic_recovery_success / (total_n_episodic_recovery + 1e-8)))
                    # Collision
                    print('RA activation rate for collision moments: {:.2%}'.format(total_n_collision_when_ra_on / (total_n_collision + 1e-8)))
                    print('RA deactivation rate for collision moments: {:.2%}'.format(total_n_collision_when_ra_off / (total_n_collision + 1e-8)))
                    
                
        
                last_obs = isaaclab_env.observation_manager.compute_group("policy").clone()
                last_root_states = isaaclab_env.scene["robot"].data.root_state_w.clone() # in world frame
                last_position_targets = isaaclab_env.command_manager.get_command("pose_command")[:, :2].clone() # in robot b frame 

                if train_ra:

                    s_queue[:-1] = s_queue[1:].clone()
                    g_queue[:-1] = g_queue[1:].clone()
                    l_queue[:-1] = l_queue[1:].clone()
                    done_queue[:-1] = done_queue[1:].clone()
                    s_queue[-1] = ra_obs.clone()
                    g_queue[-1] = gs.clone()  # note that g is obtained before done and reset and obs
                    l_queue[-1] = ls.clone()
                    done_queue[-1] = dones.clone()  # note that s is obtained after done and reset
                    
                    s_queue[-1] = ra_obs.clone()
                    g_queue[-1] = gs.clone()
                    l_queue[-1] = ls.clone()
                    done_queue[-1] = dones.clone()

                    ## hindsight for lipschitz ######
                    g_hs_queue[:-1] = g_hs_queue[1:].clone()
                    g_hs_queue[-1] = gs.clone()

                    ### calculate the span for potential hindsight lipschitz
                    g_hs_span[:] -= 1
                    g_hs_span[0][dones] = g_hs_span[1][dones].clone() + 1
                    g_hs_span[1][dones] = queue_len - 1
                    g_hs_span[0] = torch.maximum(g_hs_span[0], g_hs_span[1]-hindsight)
                    g_hs_span = g_hs_span * (g_hs_span>=0)
                    ### overwrite with hindsight for lipschitz if terminated witg gs > 0
                    range_tensor = torch.arange(queue_len).unsqueeze(1).to(env.device) #(t,1)
                    mask = (range_tensor >= g_hs_span[0:1]) & (range_tensor < g_hs_span[1:2]) #(t,n)
                    new_values = gs.clone().repeat(queue_len,1) #(t,n), broadcast last frame g to every timestep
                    mask = mask & (new_values>0) # and: dies at last frame
                    new_values -= (g_hs_span[1:2]-range_tensor)*2/hindsight*mask ## soften the values
                    g_hs_queue[mask] = new_values[mask].clone()

                    if i > queue_len and i % 20 == 0:
                        false_safe, false_reach, n_fail, n_reach, accu_loss = 0, 0, 0, 0, []
                        total_n_fail, total_n_reach = torch.logical_and(g_queue[1:]>0, done_queue[1:]).sum().item(), torch.logical_and(l_queue[:-1]<=0,done_queue[1:]).sum().item()
                        start_v = ra_vf(standard_raobs_init).mean().item()
                        die_v = ra_vf(standard_raobs_die).mean().item() 
                        turn_v = ra_vf(standard_raobs_turn).mean().item()
                        weight_end = 0.0 # ignore this, all samples are equal
                        gamma = 0.999999 #
                        print('weight of end %.3f'%(weight_end+1), 'total_n_fail',total_n_fail,'total_n_reach',total_n_reach,'gamma', gamma)
                        for _start in range(0, queue_len-1, batch_size):
                            vs_old = ra_vf(s_queue[_start:_start+batch_size]).squeeze(-1)
                            with torch.no_grad():
                                vs_new = ra_vf(s_queue[_start+1:_start+batch_size+1]).squeeze(-1) * (~done_queue[_start+1:_start+batch_size+1]) + 1.0 * done_queue[_start+1:_start+batch_size+1]
                                vs_discounted_old = gamma * torch.maximum(g_hs_queue[_start+1:_start+batch_size+1], torch.minimum(l_queue[_start:_start+batch_size],vs_new))\
                                                + (1-gamma) * torch.maximum(l_queue[_start:_start+batch_size], g_hs_queue[_start+1:_start+batch_size+1])
            
                            v_loss = 100*torch.mean(torch.square(vs_old - vs_discounted_old) * (1.0 + weight_end * (done_queue[_start+1:_start+batch_size+1]>0)))  # 
                            optimizer.zero_grad()
                            v_loss.backward()
                            torch.nn.utils.clip_grad_norm_(ra_vf.parameters(), 1.0)
                            optimizer.step()

                            false_safe += torch.logical_and(g_queue[_start+1:_start+batch_size+1]>0 , vs_old<=0).sum().item()
                            false_reach += torch.logical_and(l_queue[_start:_start+batch_size]<=0 , vs_old>0).sum().item()
                            n_fail += (g_queue[_start+1:_start+batch_size+1]>0).sum().item()
                            n_reach += (l_queue[_start:_start+batch_size]<=0).sum().item() 
                            accu_loss.append(v_loss.item())

                        new_loss = np.mean(accu_loss)

                        print('value RA loss %.4f, false safe rate %.2f in %d, false reach rate %.2f in %d, standard values init %.2f die %.2f turn %.2f, step %d'%\
                                        (new_loss, false_safe/(n_fail+1e-8), n_fail, false_reach/(n_reach+1e-8), n_reach, start_v, die_v, turn_v, i), end='   \n')

                        if false_safe/(n_fail+1e-8) < best_metric and die_v > 0.2 and start_v<-0.1 and turn_v<-0.1 and i > 3000:
                            best_metric = false_safe/(n_fail+1e-8)


                            os.makedirs(os.path.dirname(ra_path), exist_ok=True)
                            torch.save(ra_vf, ra_path)
                            
                            print('\x1b[6;30;42m', 'saving ra model to', ra_path, '\x1b[0m' )
            else:


                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    # env stepping
                    obs, _, _, _ = env.step(actions)
                

            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
        
        break
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
