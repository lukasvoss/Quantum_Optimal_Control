import sys
import os
import json

from needed_files.quantumenvironment import QuantumEnvironment
from needed_files.helper_functions import load_agent_from_yaml_file
from needed_files.ppo import make_train_ppo
from needed_files.q_env_config import q_env_config as gate_q_env_config

from braket.tracking import Tracker
from braket.jobs import hybrid_job, save_job_result
from braket.jobs.metrics import log_metric
from braket.aws import AwsDevice

from braket.aws import AwsSession
from qiskit_braket_provider import AWSBraketProvider


aws_session = AwsSession(default_bucket="amazon-braket-us-west-1-lukasvoss")


def calibrate_gate():
    print("Job started!!!!!")

    braket_task_costs = Tracker().start()
    ppo_params, network_params  = load_agent_from_yaml_file(file_path='config_yamls/agent_config.yaml')
    agent_config = {**ppo_params, **network_params}

    # WILL NOT BE USED BECAUSE WE WANT TO USE THE BRAKET QISKIT PROVIDER SV1 BACKEND
    # device = AwsDevice(os.environ["AMZN_BRAKET_DEVICE_ARN"])

    provider = AWSBraketProvider()
    backend = provider.get_backend('SV1')
    
    q_env = QuantumEnvironment(gate_q_env_config)
    q_env.backend = backend
    
    ppo_agent = make_train_ppo(agent_config, q_env)

    hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
    with open(hp_file, "r") as f:
        hyperparams = json.load(f)
    num_total_updates = hyperparams['num_total_updates']

    # training_results = ppo_agent(total_updates=hyperparams["num_updates"], print_debug=True, num_prints=40)
    training_results = ppo_agent(total_updates=num_total_updates, print_debug=False, num_prints=40)
    # save_job_result({ "measurement_counts": training_results })

    print("Job completed!!!!!")

    training_results['task_summary'] = braket_task_costs.quantum_tasks_statistics()
    training_results['estimated cost'] = float(
            braket_task_costs.qpu_tasks_cost() + braket_task_costs.simulator_tasks_cost()
        )

    return training_results