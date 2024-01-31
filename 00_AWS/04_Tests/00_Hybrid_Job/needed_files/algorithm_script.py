import sys
import os
import json

from gymnasium.wrappers import RescaleAction, ClipAction

from needed_files.quantumenvironment import QuantumEnvironment
from needed_files.helper_functions import load_agent_from_yaml_file
from needed_files.ppo import make_train_ppo
from needed_files.q_env_config import q_env_config as gate_q_env_config

from braket.tracking import Tracker
from braket.jobs import save_job_result
from braket.jobs.metrics import log_metric
from braket.aws import AwsDevice

from braket.aws import AwsSession
from qiskit_braket_provider import AWSBraketProvider



def calibrate_gate():
    print("Job started!!!!!")

    braket_task_costs = Tracker().start()
    agent_config_path = f'{os.environ["AMZN_BRAKET_INPUT_DIR"]}/agent-config/agent_config.yaml'
    ppo_params, network_params  = load_agent_from_yaml_file(file_path=agent_config_path)
    agent_config = {**ppo_params, **network_params}

    # WILL NOT BE USED BECAUSE WE WANT TO USE THE BRAKET QISKIT PROVIDER SV1 BACKEND
    # device = AwsDevice(os.environ["AMZN_BRAKET_DEVICE_ARN"])

    provider = AWSBraketProvider()
    backend = provider.get_backend('Lucy')
    
    q_env = QuantumEnvironment(gate_q_env_config)
    q_env = ClipAction(q_env)
    q_env = RescaleAction(q_env, min_action=-1.0, max_action=1.0)
    print("Backend BEFORE overwriting: ", backend)
    # q_env.unwrapped.backend = backend
    print("Backend AFTER overwriting (q_env.backend): ", q_env.backend)
    print("Backend AFTER overwriting (q_env.unwrapped.backend): ", q_env.unwrapped.backend)
    print('Type of q_env.backend: ', type(q_env.backend))
    print('Type of q_env.unwrapped.backend: ', type(q_env.unwrapped.backend))
    
    ppo_agent = make_train_ppo(agent_config, q_env)

    hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
    with open(hp_file, "r") as f:
        hyperparams = json.load(f)
    num_total_updates = int(hyperparams['num_total_updates'])

    training_results = ppo_agent(total_updates=num_total_updates, 
                                 print_debug=False, 
                                 num_prints=40,
                                 max_cost=33000)

    training_results['task_summary'] = braket_task_costs.quantum_tasks_statistics()
    training_results['estimated cost'] = float(
            braket_task_costs.qpu_tasks_cost() + braket_task_costs.simulator_tasks_cost()
        )
    if not isinstance(training_results['action_vector'], list):
        training_results['action_vector'] = training_results['action_vector'].tolist()
    
    save_job_result(training_results)
    
    print("Job completed!!!!!")

    return training_results