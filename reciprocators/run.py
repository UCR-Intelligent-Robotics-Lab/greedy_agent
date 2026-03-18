import argparse
import json
import os
import pathlib
import random
from time import sleep
import multiprocessing as mp

from omegaconf import OmegaConf
import torch

from src.training.coins_trainer import CoinsTrainer
from src.training.ipd_trainer import IPDTrainer

CUDA_LIST = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]


def run_trainer(trainer_class, config: dict, device: torch.device, max_episodes: int, results_dir: str = None):
    """Train function for use with multiprocessing."""
    trainer = trainer_class(config, device, results_dir)
    trainer.train(max_episodes)


def find_free_device(nested_process_list):
    """
    Find a free device to run a process on.
    :param nested_process_list: nested list of processes, where the first index specifies the device and the second the
      which process on that device (there may be multiple processes per device).
    """
    # Want to prioritize devices with the least number of processes
    num_free_processes = [sum([p is None for p in device_processes]) for device_processes in nested_process_list]
    if sum(num_free_processes) == 0:
        return -1
    device_idx = num_free_processes.index(max(num_free_processes))
    process_idx = nested_process_list[device_idx].index(None)
    return device_idx, process_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="test")
    parser.add_argument("-g", "--game", type=str, required=True)
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-e", "--episodes", type=int, required=True)
    parser.add_argument("-d", "--device", nargs='+', type=str, default=None)
    parser.add_argument("-pd", "--per_device", type=int, default=1)
    parser.add_argument("-r", "--replicate", type=int, default=1)
    parser.add_argument("-dd", "--results_dir", type=str, default="results")
    args = parser.parse_args()

    game_name = args.game.lower()
    if game_name == "coins":
        trainer_class = CoinsTrainer
    elif game_name == "ipd":
        trainer_class = IPDTrainer
    else:
        raise ValueError(f"Game {game_name} not recognized.")
    if args.device[0] == "all":
        device_list = CUDA_LIST if CUDA_LIST else [torch.device("cpu")]
        print("Using all available devices:", device_list)
    elif args.device is not None:
        device_list = [torch.device(d) for d in args.device]
        print("Using devices:", device_list)
    else:
        device_list = [torch.device("cpu")]
        print("No device specified, using CPU")
    num_processes_per_device = args.per_device

    config = OmegaConf.load(args.config)
    seed_start = config.random_seed

    base_name = args.name
    config.name = base_name
    max_episodes = args.episodes
    results_dir = os.path.join(args.results_dir, base_name)

    processes = [[None] * num_processes_per_device for _ in range(len(device_list))]

    for i in range(args.replicate):
        if args.replicate > 1:
            config.name = f"{base_name}_{i}"
            config.random_seed = i + seed_start

        while True:
            if find_free_device(processes) == -1:
                # If no devices are free, check
                for device_id in range(len(processes)):
                    for process_id in range(len(processes[device_id])):
                        if not processes[device_id][process_id].is_alive():
                            processes[device_id][process_id] = None
                sleep(1)
            else:
                free_device_idx, free_process_idx = find_free_device(processes)
                print(f"Starting process {free_process_idx} on device {device_list[free_device_idx]}...")
                expt_dir = os.path.join(results_dir, config.name)
                if not os.path.isdir(expt_dir):
                    pathlib.Path(expt_dir).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(expt_dir, "commandline_args.txt"), "w") as f:
                        json.dump(args.__dict__, f, indent=2)
                    with open(os.path.join(expt_dir, "config.yaml"), "w") as f:
                        OmegaConf.save(config, f)
                break

        process_args = (trainer_class, config, device_list[free_device_idx], max_episodes, expt_dir)

        processes[free_device_idx][free_process_idx] = mp.Process(target=run_trainer, args=process_args)
        processes[free_device_idx][free_process_idx].start()
