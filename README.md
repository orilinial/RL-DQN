# RL-DQN
This repository holds a deep RL solutions for solving OpenAI-gym's TAXI and ACROBOT environments.
All the files below have arguments which can be changed (but all set to the our choice of default parameters).
To see all arguments for each script run: <SCRIPT NAME>.py --help
Example for running a script: 'python train_taxi_dqn.py'

## Taxi environment:
### Train scripts:
DQN: train_taxi_dqn.py
Vanilla Policy Gradient: train_taxi_pg.py
These files are scripts for training the model for the taxi environment.
The models used are in - model_taxi.py

### Evaluation Scripts:
DQN: eval_taxi_dqn.py
Vanilla Policy Gradient: eval_taxi_pg.py
These files are scripts for evaluating the models for the taxi environment.

### Evaluation rewards are in the files:
DQN: eval_reward_dqn_taxi.npy
Policy Gradients: eval_reward_pg_taxi.npy
Plots can be seen using the file plot.py --path <PATH TO NPY FILE>



## Acrobot:
Train files:

The file eval_reward_pg_taxi.npy
