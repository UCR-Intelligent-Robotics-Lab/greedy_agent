# REFiNE (This is the tensorflow v1 for PC)

Official implementations and results of the paper [REFiNE: Reward and Energy Fairness for Robust Multi-Agent Coordination in RL-driven Robots].

## Overview
REFiNE is a fully self-contained TensorFlow v1 codebase on reward & energy fairness in multi-agent reinforcement learning. It implements:

Core algorithms:
- LIO (Learning to Incentivize Others) for baseline comparisons
- EIA (Exogenous Incentive Adjustment) to validate robustness
- REFiNE (a novel Reward & Energy First Notably Equilibrated Incentivization)

Environment wrappers:
- Escape Room (ER) and Iterated Prisoner’s Dilemma (IPD) for social-dilemma tests

Experiment orchestration:
- Config‐driven training scripts (train_*.py )
- run_experiments_er.py or run_experiments_ipd.py to launch multi-seed, parallel runs for ER or IPD experiments
- save experiment result logs in results/er * or results/ipd * for the *th experiment on ER or IPD setting

Visualization & analysis
- Post-processing scripts for reward, fairness, and energy metrics
- Figure generators to reproduce all paper plots

Everything lives under the lio/ folder, with clear subdirectories for algorithm code (alg/) vs. env wrappers (env/), making it easy to locate, extend, or plug in your own MARL variants.




## Setup

- Goto ./lio folder.
- Python 3.6
- Tensorflow >= 1.12
- OpenAI Gym == 0.10.9
- Clone and `pip install` [Sequential Social Dilemma](https://github.com/011235813/sequential_social_dilemma_games), which is a fork from the [original](https://github.com/eugenevinitsky/sequential_social_dilemma_games) open-source implementation.
- Clone and `pip install` [LOLA](https://github.com/alshedivat/lola) if you wish to run this baseline.
- Clone this repository and run `$ pip install -e .` from the root.

```markdown
- **Navigation**
  - `README.md` ← this file  
  - `environment.yml` ← conda env spec (TensorFlow 1, Gym, etc.)  
  - `requirements.txt` ← pip dependencies  
  - `run.sh` ← helper script to launch experiments  
  - **lio/**
    - **alg/**
      - `config_room_lio.py` ← LIO on Escape Room (ER)  
      - `config_room_REFINE.py` ← REFiNE on ER  
      - `config_ipd_lio.py` ← LIO on Iterated Prisoner’s Dilemma (IPD)
      - `config_ipd_REFINE.py` ← REFiNE on Iterated Prisoner’s Dilemma (IPD)
      - `train_.py` ← training scripts (e.g. train_REFINE_eia_er.py)
      - `run_trained_.py` ← evaluation scripts for trained models
      - `run_experiments_.py` ← unified entry point for reproducible runs on ER or IPD
    - **env/**
      - `room_agent.py`  
      - `room_symmetric_*.py`  
      - `ipd_wrapper.py`  
  - `LICENSE` ← MIT license  
```

## Examples

### Train REFiNE on Escape Room

* Set config values in `alg/config_room_REFiNE.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_REFiNE_eia_er.py er i`. Default settings saves results in results/er i, meaning the ith experiment.

### Train REFiNE on Iterated Prisoner’s Dilemma

* Set config values in `alg/config_ipd_REFiNE.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_REFiNE_eia_ipd.py ipd i`. Default settings saves results in results/ipd i, meaning the ith experiment.

### Run trained REFiNE model on ER (4,2) that is gained in experiment er i


* `cd` into the `alg` folder
* Execute training script `$ python run_trained_REFiNE_eia_er.py`. 

### Evaluation & Visualization

* Plotting scripts are available in plot_ER42_REFiNE.py for ER (4,2) case

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
- [LOLA baseline](https://github.com/alshedivat/lola)
- [Sequential Social Dilemma](https://github.com/eugenevinitsky/sequential_social_dilemma_games)
### Install
First, make sure you have cloned all repos and setup the python environment. We use miniconda to automate the packages for Nano env (Miniconda3-py312_25.1.1-2-Linux-aarch64.sh)
```bash
conda env create -f environment_nano.yml
```
Then, follow the steps in the [Sequential Social Dilemma](https://github.com/eugenevinitsky/sequential_social_dilemma_games) to setup. The ray repo is the one you have already cloned in the repos needed part.

Next, Make sure you followed [LOLA baseline](https://github.com/alshedivat/lola) setup in the cloned repo

Finally, check if you can run the train_lio_er.py er for the escape room case using the code below (er means escape room, 4 means 4 agents):
```bash
LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so python train_lio_er.py er 4
```



## Acknowledgement

Code is implemented based on [Learning to Incentivize Other Learning Agents](https://github.com/011235813/lio). We would like to thank the authors for making their code public.

## License

See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT
