# Learning Project: Migrating ABS(Agile but safe) Strategies to IsaacLab

>  **Disclaimer**: This repository is a personal, non-commercial academic project.  
It is developed solely for the purpose of personal learning and experimentation.  
The author make no claim of ownership over the original ABS or IsaacLab codebases.
>
> 
> This codebase is under inherited license in ABS [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en), and inherited license in Isaaclab  [BSD-3 License](/LICENSE) and [Apache 2.0](/LICENSE-mimic). The usage for commercial purposes is not allowed, e.g., to make demos to advertise commercial products.
> 
>The provided Dockerfiles rely on NVIDIA Isaac Sim base images 
(distributed via NGC). Usage of these images requires accepting the 
[NVIDIA EULA](https://docs.nvidia.com/isaac/isaac-sim/latest/eula.html). 
This repository does not redistribute Isaac Sim images; users must pull 
them directly from NGC if needed. see also:https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html
> 
>All usage of this repository must comply with the above upstream licenses.  
>The author of this repository is **not responsible** for any misuse or violation of the original licenses by third parties.

This work re-implements and adapts the core **Agile But Safe (ABS)** strategies and **RA value training** pipeline from the [ABS project](https://github.com/LeCAR-Lab/ABS), migrating from **IsaacGym** to **IsaacLab**.

The system follows the **IsaacLab Manager-Based architecture**, with logic and evaluation protocols aligned with the ABS methodology, but adjusted to match IsaacLab conventions (e.g., event/reset configuration, sensor pipeline).



---

## Policy Behavior Comparison

<table>
<thead>
<tr>
  <th></th>
  <th align="center"><b>ABS Policy</b><br><i>Safer with recovery</i></th>
  <th align="center"><b>Agile Policy</b><br><i>Fast and aggressive</i></th>
</tr>
</thead>

<tbody>

<tr>
  <td><b> Flat Terrain</b></td>
  <td><img src="videos and pictures/rl-video-ABS-flat.gif" width="320"/></td>
  <td><img src="videos and pictures/rl-video-agile-flat.gif" width="320"/></td>
</tr>

<tr>
  <td><b> Rough Terrain</b></td>
  <td><img src="videos and pictures/rl-video-ABS-rough.gif" width="320"/></td>
  <td><img src="videos and pictures/rl-video-agile-rough.gif" width="320"/></td>
</tr>

<tr>
  <td><b> Low Obstacles</b></td>
  <td><img src="videos and pictures/rl-video-ABS-lowobst.gif" width="320"/></td>
  <td><img src="videos and pictures/rl-video-agile-lowobst.gif" width="320"/></td>
</tr>

</tbody>
</table>

---

## Evaluation Results

The following comparison between ABS and Agile policies is based on over 50k test episodes on flat terrain.

<table>
<tr>
  <td align="center"><b>ABS Policy</b></td>
  <td align="center"><b>Agile Policy</b></td>
</tr>
<tr>
  <td><img src="videos and pictures/ABS_flat.png" width="400"/></td>
  <td><img src="videos and pictures/agile_flat.png" width="400"/></td>
</tr>
</table>

The following visualization shows RA values over 2D position grids under different commanded velocities. Warmer colors (red) indicate higher risk, and cooler colors (green) indicate safety.

<img src="videos and pictures/ra_value_map.png" width="600"/>



## environment description

The framework of Isaaclab https://isaac-sim.github.io/IsaacLab/v2.1.0/

The training and testing of this repo is based on isaaclab docker with version 2.1.0 https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html

python version: 3.10.15,  numpy version: 1.26.4, torch version: 2.5.1 + cu118

###  training and playing

# agile policy
```bash
   ./isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Pos-Unitree-Go1-v0 --headless --max_iterations=800
```
# recovery policy
```bash
   ./isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Rec-Unitree-Go1-v0 --headless --max_iterations=800
```
# RA network and testing
  **Note**:The bool flag needs to be set in play.py
  

```bash
   ./isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Pos-Unitree-Go1-Play-v0 --headless --num_envs=1 --video --enable_cameras --video_length=5000 
```
--video can be used to make videos


##  Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [ABS](https://github.com/LeCAR-Lab/ABS) 
- [IsaacLab](https://github.com/isaac-sim/IsaacLab) 



