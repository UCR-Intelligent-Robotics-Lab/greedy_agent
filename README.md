# PIMbot

Official implementations and results of IROS 2023 paper [PIMbot: Policy and Incentive Manipulation for Multi-Robot Reinforcement Learning in Social Dilemmas](https://arxiv.org/pdf/2307.15944).

## Overview
PIMbot introduces two forms of reward function manipulation in multi-agent reinforcement learning (RL) social dilemmas:

1.	Policy Manipulation: Adjusting the decision-making strategies of robots to influence task outcomes.

2.	Incentive Manipulation: Modifying reward structures to change robot behavior within a social dilemma.

## Setup

- Goto ./lio folder.
- Python 3.6
- Tensorflow >= 1.12
- OpenAI Gym == 0.10.9
- Clone and `pip install` [Sequential Social Dilemma](https://github.com/011235813/sequential_social_dilemma_games), which is a fork from the [original](https://github.com/eugenevinitsky/sequential_social_dilemma_games) open-source implementation.
- Clone and `pip install` [LOLA](https://github.com/alshedivat/lola) if you wish to run this baseline.
- Clone this repository and run `$ pip install -e .` from the root.

## Navigation
* `./*.ipynb` - Plot and visualization scripts.
* `./*.py` - Plot scripts.
* `./*.png` - Experimental results in PIMbot paper.
* `./lio/alg/` - Implementation of LIO and PG/AC baselines
* `./lio/env/` - Implementation of the Escape Room game and wrappers around the SSD environment.
* `./lio/results/` - Results of training will be stored in subfolders here. Each independent training run will create a subfolder that contains the final Tensorflow model, and reward log files. For example, 5 parallel independent training runs would create `results/cleanup/10x10_lio_0`,...,`results/cleanup/10x10_lio_4` (depending on configurable strings in config files).
* `./lio/utils/` - Utility methods.

## Examples

### Train LIO on Escape Room

* Set config values in `alg/config_room_lio.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_multiprocess.py lio er`. Default settings conduct 5 parallel runs with different seeds.
* For a single run, execute `$ python train_lio.py er`.

### Train LIO on Cleanup

* Set config values in `alg/config_ssd_lio.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_multiprocess.py lio ssd`.
* For a single run, execute `$ python train_ssd.py`.

## Setup and run on Jetson Nano (2025)
### Pre-req packages
- JetPack 5.1.2 (L4T 35.4.1)
- Python 3.8.20
- Miniconda
- OpenAI Gym 0.26.2
- opencv-python 4.11.0.86
- ray==2.2.0
### Repos needed:
- Ray from the original sequential social dilemma: https://github.com/natashamjaques/ray.git
- LOLA from our version: https://github.com/UCR-Intelligent-Robotics-Lab/lola
- Sequential Social Dilemma from our version: https://github.com/UCR-Intelligent-Robotics-Lab/sequential_social_dilemma_games.git
### Install
First, make sure you have cloned all repos and setup the python environment. We use miniconda to automate the packages for Nano env (Miniconda3-py312_25.1.1-2-Linux-aarch64.sh)
```bash
conda env create -f environment_nano.yml
```
Then, follow the steps in the [Sequential Social Dilemma](https://github.com/UCR-Intelligent-Robotics-Lab/sequential_social_dilemma_games.git) to setup. The ray repo is the one you have already cloned in the repos needed part.

Next, Make sure you followed our [LOLA](https://github.com/UCR-Intelligent-Robotics-Lab/lola) setup in the cloned repo

Finally, check if you can run the train_lio.py er for the escape room case using the code below (er means escape room, 4 means 4 agents):
```bash
LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so python train_lio_er.py 4
LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so python train_lio_ipd.py 4
```

## Citation

Please cite our paper if you are inspired by PIMbot in your work:

<pre>
@inproceedings{nikkhoo2023pimbot,
  title={Pimbot: Policy and incentive manipulation for multi-robot reinforcement learning in social dilemmas},
  author={Nikkhoo, Shahab and Li, Zexin and Samanta, Aritra and Li, Yufei and Liu, Cong},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5630--5636},
  year={2023},
  organization={IEEE}
}
</pre>

## Acknowledgement

Code is implemented based on [Learning to Incentivize Other Learning Agents](https://github.com/011235813/lio). We would like to thank the authors for making their code public.

## License

See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT

## New Baselines and Benchmarks (Reciprocators and ADMO)

We have extended this repository to incorporate the **Reciprocators** baseline and a novel **Spatial Stag Hunt** benchmark. Additionally, we have integrated the **Adaptive Multi-Objective (ADMO)** attack strategy for these new settings.

### 1. Reciprocators Integration
The [Reciprocators](https://arxiv.org/abs/2410.21360) framework has been fully integrated and modified to run natively on the environments from this repository (Escape Room and IPD). 
- **Codebase:** Included in the `reciprocators/` directory.
- **Architectural Changes:** The `StateEncoder` in `reciprocators/src/agents/components.py` was modified to support 1D vector observations natively (falling back from the original Conv2D implementation), allowing seamless evaluation on existing grid and matrix games without external heavy simulators.
- **Execution:** 
  You can find the run scripts in the `scripts/` folder or run them directly:
  ```bash
  cd reciprocators
  python run_er.py --episodes 5000 --device cuda
  python run_ipd.py --episodes 5000 --device cuda
  ```

### 2. Spatial Stag Hunt Benchmark
We introduced a custom, SOTA Multi-Agent Reinforcement Learning (MARL) benchmark: **Spatial Stag Hunt**. This environment extends beyond simple matrix games (like IPD) and 1D coordination lines (like Escape Room) by introducing a 2D 5x5 spatial sparse environment containing 2 learning agents, 2 stationary Hares, and 1 mobile Stag.

- **LIO Implementation:** Evaluated under `lio/env/staghunt.py` and `lio/alg/lio/train_lio_staghunt.py`.
- **Reciprocators Implementation:** Evaluated under `reciprocators/run_staghunt.py`.

### 3. ADMO Attacks
We extended the Adaptive Multi-Objective (ADMO) attack to exploit both the new Spatial Stag Hunt environment and the new Reciprocators agent baseline.
- **LIO ADMO:** Added `lio/alg/lio/train_lio_staghunt_admo.py`.
- **Reciprocators ADMO:** Added an ADMO controller (`reciprocators/admo.py`) adapted for PPO-style intrinsic reward manipulation without explicit environmental reward exchange. Implemented in `run_er_admo.py`, `run_ipd_admo.py`, and `run_staghunt_admo.py`.

### Scripts
A comprehensive suite of `autorun` bash scripts has been added to the `scripts/` folder. All old scripts were prefixed properly and we added autorun variants for LIO, Reciprocators, and their respective ADMO attacks.
