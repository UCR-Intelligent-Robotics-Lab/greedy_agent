# Reciprocal Reward Influence Encourages Cooperation From Self-Interested Agents
This is the official implementation of Reciprocator agents.

After cloning this repository and installing `python==3.10.13`, you can install the required dependencies by running:
```
pip install -r requirements.txt
pip install -e .
```

To run the experiments, you can use the following command template (with an example for Coins):
```
python run.py -n coins_reciprocator_vs_naive -g coins -c configs/coins.yaml -e 500 -d all -r 8 -dd results
```
## GPU/CPU Setup & Usage

The implementation has been modified to automatically fall back to CPU if no GPU is found on your machine when specifying `all` devices. 

To run the codebase natively with the required dependencies, use the following commands:
```bash
cd reciprocators
pip install -r requirements.txt
pip install -e .

# Example to run the Coins game on all available GPUs (or fallback to CPU):
python run.py -n coins_reciprocator_vs_naive -g coins -c configs/coins.yaml -e 500 -d all -r 8 -dd results
```


## Running Reciprocators on LIO Environments (Escape Room and IPD)

We have added support to run the `Reciprocators` agent against the LIO base implementations of `EscapeRoom` and `Iterated Prisoners Dilemma`. The environments automatically run on GPU natively if available, or fall back to CPU.

```bash
cd reciprocators

# Run Reciprocators in the Escape Room environment
python run_er.py --episodes 5000 --device cuda

# Run Reciprocators in the Iterated Prisoners Dilemma environment
python run_ipd.py --episodes 5000 --device cuda
```

