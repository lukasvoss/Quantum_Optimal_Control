# %%
from qiskit import QuantumRegister
import numpy as np
from gymnasium.spaces import Box
from qiskit.quantum_info import Statevector

from rl_qoc import QuantumEnvironment, BenchmarkConfig, StateRewardConfig
from rl_qoc.qibo import QiboEnvironment
from qiskit.circuit import QuantumCircuit, ParameterVector, Gate
from qiskit.circuit.library import CZGate, RXGate, XGate
from rl_qoc import (
    QEnvConfig,
    ExecutionConfig,
    ChannelRewardConfig,
)
from rl_qoc.qibo import QiboConfig
from gymnasium.wrappers import ClipAction, RescaleAction


def param_circuit(
    qc: QuantumCircuit, params: ParameterVector, qreg: QuantumRegister, **kwargs
):
    target = kwargs["target"]
    gate: Gate = target.get("gate", "x")
    if gate == "x":
        gate_name = "x"
    else:
        gate_name = gate.name
    physical_qubits = target["physical_qubits"]
    custom_gate = Gate(f"{gate_name}_cal", len(physical_qubits), params.params)
    qc.append(custom_gate, qreg)

    return qc


def get_backend():
    return "qibolab"


target = {"state": Statevector.from_label("1"), "physical_qubits": [0]}
instruction_durations = {}
action_space_low = np.array([0.001], dtype=np.float32)  # [amp, phase, phase, duration]
action_space_high = np.array([0.1], dtype=np.float32)  # [amp, phase, phase, duration]
action_space = Box(action_space_low, action_space_high)
qibo_config = QiboConfig(
    param_circuit,
    get_backend(),
    platform="dummy",
    physical_qubits=([0]),
    gate_rule="x",
    parametrized_circuit_kwargs={"target": target},
    instruction_durations=None,
)
q_env_config = QEnvConfig(
    target=target,
    backend_config=qibo_config,
    action_space=action_space,
    reward_config=StateRewardConfig(),
    benchmark_config=BenchmarkConfig(1, check_on_exp=True, method="rb"),
    execution_config=ExecutionConfig(
        batch_size=1,
        sampling_paulis=50,
        n_shots=1000,
        n_reps=1,
        c_factor=1,
    ),
)

# env = QuantumEnvironment(q_env_config)
env = QiboEnvironment(
    q_env_config,
)
# env.circuits[0].draw(output="mpl")
# # %%
# env.baseline_circuits[0].draw(output="mpl")
# %%
from rl_qoc import CustomPPO
from rl_qoc.agent import TrainFunctionSettings, TotalUpdates, TrainingConfig
from rl_qoc.helpers import load_from_yaml_file

file_name = "agent_config_personal.yaml"

agent_config = load_from_yaml_file(file_name)
# %%
# ppo = CustomPPO(
#     agent_config,
#     ClipAction(RescaleAction(env, action_space.low, action_space_high)),
#     save_data=True,
# )
total_updates = TotalUpdates(500)
# hardware_runtime = HardwareRuntime(300)
training_config = TrainingConfig(
    training_constraint=total_updates,
    target_fidelities=[0.999, 0.9999],
    lookback_window=10,
    anneal_learning_rate=True,
    std_actions_eps=1e-2,
)

train_function_settings = TrainFunctionSettings(
    plot_real_time=True,
    print_debug=True,
    num_prints=1,
    hpo_mode=False,
    clear_history=True,
)
# %%
# ppo.train(training_config, train_function_settings)
env.step(np.expand_dims(np.array([0.3333]), axis=0))
