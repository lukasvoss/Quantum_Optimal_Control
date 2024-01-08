import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import json
module_path = os.path.abspath(os.path.join('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control'))
if module_path not in sys.path:
    sys.path.append(module_path)

from quantumenvironment import QuantumEnvironment
from template_configurations import gate_q_env_config
from helper_functions import load_agent_from_yaml_file
from ppo import make_train_ppo

from braket.aws import AwsDevice
from braket.jobs import save_job_result

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s", # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

def start_here():
    print("Test job started!!!!!")

    # #Load the Hybrid Job hyperparameters
    # hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
    # with open(hp_file, "r") as f:
    #     hyperparams = json.load(f)

    device = AwsDevice(os.environ['AMZN_BRAKET_DEVICE_ARN'])
    q_env = QuantumEnvironment(gate_q_env_config)
    q_env.backend = device
    ppo_params, network_params  = load_agent_from_yaml_file(file_path='/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/agent_config.yaml')
    agent_config = {**ppo_params, **network_params}
    ppo_agent = make_train_ppo(agent_config, q_env)

    # training_results = ppo_agent(total_updates=hyperparams["num_updates"], print_debug=True, num_prints=40)
    training_results = ppo_agent(total_updates=10, print_debug=False, num_prints=40)
    save_job_result({ "measurement_counts": training_results })

    print("Test job completed!!!!!")