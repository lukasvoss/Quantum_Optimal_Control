# %% [markdown]
# # Quantum Gate calibration using Model Free Reinforcement Learning in AWS BraketLocalBackend
# 
# We extend the state preparation scheme to a gate calibration scheme by providing multiple input states to the target.

# %%
from qiskit_braket_provider import BraketLocalBackend, AWSBraketProvider, AWSBraketBackend, AWSBraketJob
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
import boto3
client = boto3.client("braket", region_name='us-west-1')
import braket._sdk as braket_sdk
braket_sdk.__version__

# %%
# response = client.search_devices(filters=[{
#     'name': 'deviceArn',
#     'values': ['arn:aws:braket:::device/quantum-simulator/amazon/sv1']
# }], maxResults=10)
# print(f"Found {len(response['devices'])} devices")

# response = client.get_device(deviceArn='arn:aws:braket:::device/quantum-simulator/amazon/sv1')
# print(f"Device {response['deviceName']} is {response['deviceStatus']}")

# %%
import numpy as np
import os
import sys
from typing import Optional
module_path = os.path.abspath(os.path.join('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control'))
if module_path not in sys.path:
    sys.path.append(module_path)
from quantumenvironment import QuantumEnvironment
from helper_functions import select_optimizer, generate_model
from qconfig import QiskitConfig
from template_configurations import gate_q_env_config
from agent import Agent
from helper_functions import load_agent_from_yaml_file
from ppo import make_train_ppo
from qconfig import QEnvConfig

# Qiskit imports for building RL environment (circuit level)
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2
from qiskit.extensions import CXGate, XGate
from qiskit.opflow import Zero, One, Plus, Minus, H, I, X, CX, S, Z
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator

# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag

# Additional imports
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

if __name__ == '__main__':

    backend_name = 'ibm_perth'
    no_backend = None
    backend = no_backend

    # %%
    provider = AWSBraketProvider()

    # TODO: Task Batching currently does not work for SV1 (despite AWS Braket documentation mentioning it as a use case)
    # So it's probably an issue in handling the details in quantuumenvironment.py
    backend = provider.get_backend('SV1')

    # backend = BraketLocalBackend()    
    # backend = LocalSimulator()

    # %%
    type(backend)

    # %%
    from qiskit.providers import BackendV2

    type(backend)
    isinstance(backend, (AWSBraketBackend, LocalSimulator, BraketLocalBackend))

    # %%
    backend

    # %%
    q_env = QuantumEnvironment(gate_q_env_config)
    print("Backend:", q_env.backend)

    ppo_params, network_params  = load_agent_from_yaml_file(file_path='/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/agent_config.yaml')
    agent_config = {**ppo_params, **network_params}

    ppo_agent = make_train_ppo(agent_config, q_env)
    training_results = ppo_agent(total_updates=200, print_debug=True, num_prints=40)