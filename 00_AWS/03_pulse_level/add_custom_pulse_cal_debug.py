# %%
from braket.aws import AwsQuantumJob, AwsSession
from braket.jobs.local import LocalQuantumJob
from braket.jobs.image_uris import Framework, retrieve_image
from qiskit_braket_provider.providers import adapter

aws_session = AwsSession(default_bucket="amazon-braket-us-west-1-lukasvoss")

from needed_files.quantumenvironment import QuantumEnvironment
from needed_files.helper_functions import load_agent_from_yaml_file
from needed_files.ppo import make_train_ppo
from needed_files.q_env_config import q_env_config as gate_q_env_config

from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister
from qiskit.circuit.parametervector import ParameterVectorElement
import numpy as np

import time

# %%
q_env = QuantumEnvironment(gate_q_env_config)

# %% [markdown]
# ##### Parametrized qiskit circuit

# %%
qiskit_circuit = q_env.circuit_truncations[0]
# qiskit_circuit.draw(output='mpl')

# %% [markdown]
# #### Convert the qiskit circuit to a braket circuit

# %%
# braket_circuit = adapter.convert_qiskit_to_braket_circuit(qiskit_circuit)

# %%
q_reg = QuantumRegister(2)
params = ParameterVector(name='a', length=7)
my_qc = QuantumCircuit(q_reg, name="custom_cx")
optimal_params = np.pi * np.zeros(7)

my_qc.u(
    optimal_params[0] + params[0],
    optimal_params[1] + params[1],
    optimal_params[2] + params[2],
    q_reg[0],
)
my_qc.u(
    optimal_params[3] + params[3],
    optimal_params[4] + params[4],
    optimal_params[5] + params[5],
    q_reg[1],
)
my_qc.rzx(optimal_params[6] + params[6], q_reg[0], q_reg[1])
my_qc.draw(output='mpl')

# %%
braket_circuit = adapter.convert_qiskit_to_braket_circuit(my_qc)
print(braket_circuit)