# RL-DQN
This repository holds deep RL solutions for solving OpenAI's gym environments: TAXI and ACROBOT. </br>

###### Scripts Usage:
All the files below have arguments which can be changed (but all set to the our choice of default parameters). </br>
To see all arguments for each script run: `<SCRIPT NAME>.py --help` </br>
Example for running a script: `python train_taxi_dqn.py`

## Taxi environment:
### Train scripts:
To train the model that solves the TAXI environment, run the following scripts: </br>
Using the DQN method: `train_taxi_dqn.py` </br>
Using the Vanilla Policy Gradient method: `train_taxi_pg.py` </br>
</br>
The models architecture is specified in: `model_taxi.py`

### Evaluation scripts:
To test the models that solves the TAXI environment, run the following scripts: </br>
DQN: `eval_taxi_dqn.py` </br>
Vanilla Policy Gradient: `eval_taxi_pg.py` </br>
</br>
These scripts use the saved models: `dqn_taxi_model.pkl` and `pg_taxi_model.pkl`. </br>
</br>
To see the accumulated reward graphs, aas a function of training episode, use the data in:</br>
DQN: `eval_reward_dqn_taxi.npy` </br>
Policy Gradients: `eval_reward_pg_taxi.npy` </br>
These files hold the accumulated reward achieved for evaluation, when we ran evaluation every 10 training episodes.
Plots can be seen using: `plot.py --path <PATH TO NPY FILE>`

## Acrobot:
As in taxi, we provide a train script (using DQN) and a test script: </br>
Train script: `train_acrobot.py` </br>
Test script: `eval_reward_pg_taxi.npy`</br>
</br>
The model trained is `acrobot_model.pkl`, and the architecture is in `model_acrobot.py`. </br>
</br>
To see the accumulated reward graphs, use the data in:</br>
Test: `acrobot_reward_eval.npy` </br>
Train: `acrobot_reward_train.npy` </br>
The _test_ file holds the accumulated reward achieved for evaluation, when we ran evaluation every 10 training episodes. </br>
The _train_ file holds the accumulated reward achieved for every train episode.


