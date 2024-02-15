# %%
import numpy as np
import re

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import BackendEstimator, Estimator, Sampler, BackendSampler

from braket.devices import LocalSimulator
from braket.circuits import Circuit, Gate, Instruction, QubitSet, AngledGate, Observable
from braket.circuits import noises
from braket.circuits.gates import X, Rx, Rz, CNot, XY, PulseGate, U
from braket.quantum_information import PauliString
from braket.parametric import FreeParameter, FreeParameterExpression

from qiskit_braket_provider.providers import adapter
from qiskit_braket_provider import AWSBraketProvider

device = LocalSimulator()
# %%
params = ParameterVector("a", 1)
qiskit_circuit = QuantumCircuit(QuantumRegister(1), name="test_param_binding")
qiskit_circuit.rx(params[0], 0)

qiskit_circuit.draw("mpl")
braket_circ = adapter.convert_qiskit_to_braket_circuit(qiskit_circuit)
print(braket_circ)

# run the circuit on the local simulator
task = device.run(braket_circ, shots=1000, inputs={'a0': 0.0})

# visualize the results
result = task.result()
measurement = result.measurement_counts
print('measurement results:', measurement)


# %%
action_vector = np.random.uniform(-np.pi, np.pi, 7)

# %% [markdown]
# ##### Parametrized qiskit circuit

# %%
q_reg = QuantumRegister(2)
baseline = np.random.uniform(-np.pi, np.pi, 7)
params = ParameterVector("a", 7)
qiskit_circuit = QuantumCircuit(q_reg, name="custom_cx")
# optimal_params = np.pi * np.array([0.0, 0.0, 0.5, 0.5, -0.5, 0.5, -0.5])
optimal_params = np.pi * np.zeros(7)

qiskit_circuit.u(
    baseline[0] + params[0],
    baseline[1] + params[1],
    baseline[2] + params[2],
    q_reg[0],
)
qiskit_circuit.u(
    baseline[3] + params[3],
    baseline[4] + params[4],
    baseline[5] + params[5],
    q_reg[1],
)

qiskit_circuit.rzx(baseline[6] + params[6], q_reg[0], q_reg[1])

qiskit_circuit.draw("mpl")

# %% [markdown]
# #### Convert the qiskit circuit to a braket circuit

# %%
braket_circuit = adapter.convert_qiskit_to_braket_circuit(qiskit_circuit)
print(braket_circuit)

# %% [markdown]
# ### Run the circuit

# %%
param_names = [str(param) for param in params]
# Regular expression pattern to match any of the parentheses: {}, [], ()
pattern = r'[\[\]\{\}\(\)]'

# Remove the parentheses from each string in the list
param_names = [re.sub(pattern, '', string) for string in param_names]

# %%
bound_parameters = dict(zip(param_names, action_vector))

# %%
bound_parameters

# %%
braket_circuit.to_ir().instructions

# %%
# run the circuit on the local simulator
task = device.run(braket_circuit, shots=1000, inputs=bound_parameters)

# visualize the results
result = task.result()
measurement = result.measurement_counts
print('measurement results:', measurement)

# %%



