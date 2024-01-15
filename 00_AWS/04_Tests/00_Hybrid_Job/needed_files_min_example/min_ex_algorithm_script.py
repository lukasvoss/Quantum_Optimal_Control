# Import of own modules
from needed_files_min_example.qconfig import QiskitConfig, QEnvConfig
from needed_files_min_example.quantumenvironment import QuantumEnvironment
from needed_files_min_example.helper_functions import load_q_env_from_yaml_file
from needed_files_min_example.q_env_config import get_backend, apply_parametrized_circuit, RuntimeBackend, get_circuit_context

from needed_files_min_example.quantumenvironment import QuantumEnvironment

### INFO ###
# The following modules would also be used in the real implementation but not relevant for this minimal example
# Both load_agent_from_yaml_file and make_train_ppo are functions
# from helper_functions import load_agent_from_yaml_file
# from ppo import make_train_ppo

# Braket import
from braket.tracking import Tracker
from braket.aws import AwsSession
from qiskit_braket_provider import AWSBraketProvider

# Other imports
import os
import json


def calibrate_gate():
    print("Job started!!!!!")
    braket_task_costs = Tracker().start()

    # config_file_address = os.path.join(os.environ["AMZN_BRAKET_INPUT_DIR"], 'q_env_gate_config.yaml')
    config_file_address = f'{os.environ["AMZN_BRAKET_INPUT_DIR"]}/input-data/q_env_gate_config.yaml'

    
    print(config_file_address)
    (params, backend_params, estimator_options, check_on_exp) = load_q_env_from_yaml_file(
    config_file_address
    )
    backend = get_backend(**backend_params)
    backend_config = QiskitConfig(
        apply_parametrized_circuit,
        backend,
        estimator_options=estimator_options
        if isinstance(backend, RuntimeBackend)
        else None,
        parametrized_circuit_kwargs={"target": params["target"], "backend": backend},
    )
    QuantumEnvironment.check_on_exp = check_on_exp
    gate_q_env_config = QEnvConfig(backend_config=backend_config, **params)
    circuit_context = get_circuit_context(backend)




    # XXX: The following step, so using the object gate_q_env_config fails because it is non-serializable.
    q_env = QuantumEnvironment(gate_q_env_config)
    

    ### INFO ###

    # The code below is a dummy/schematic implementation of what happens later in our workflow.
    # The error-causing line is the one with XXX above. Since this error-causing line is the first step in our workflow,
    # we cannot/could not proceed with the rest of the workflow.

    # Since our codebase is based on Qiskit, we want to use Braket's Qiskit Provider (here start with SV1 for testing)
    provider = AWSBraketProvider()
    backend = provider.get_backend('SV1')
    # q_env.backend = backend

    ### Access the hyperparameters from the environment variables
    # hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
    # with open(hp_file, "r") as f:
    #     hyperparams = json.load(f)
    # num_total_updates = hyperparams['num_total_updates']
    
    # Load the RL agent based on the QuantumEnvironment object
    # Run the parametrized quantum circuit that is part of the ``gate_q_env_config`` object
    # and train the RL agent based on the reward
    
    # Return training results (here dummy results)
    dummy_training_results = {
        'avg_return': [0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95],
        'action_vector': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
        'task_summary': braket_task_costs.quantum_tasks_statistics(),
        'estimated cost': float(
            braket_task_costs.qpu_tasks_cost() + braket_task_costs.simulator_tasks_cost()
        )
    }
    return dummy_training_results