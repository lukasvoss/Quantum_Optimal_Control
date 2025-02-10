import os
import sys
import gzip
import pickle

project_path = "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/"
sys.path.append(project_path)
import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.transpiler import PassManager, CouplingMap
from gymnasium.spaces import Box

# Custom imports from project
from gate_level.spillover_noise_use_case.generic_spillover.spillover_effect_on_subsystem import (
    LocalSpilloverNoiseAerPass,
    circuit_context,
    numpy_to_hashable,
    noisy_backend,
)
from rl_qoc import (
    QEnvConfig,
    ExecutionConfig,
    ContextAwareQuantumEnvironment,
    RescaleAndClipAction,
)
from rl_qoc.environment.configuration.backend_config import QiskitConfig
from rl_qoc.helpers.circuit_utils import get_gate
from rl_qoc.agent import (
    PPOConfig,
    CustomPPO,
    TrainingConfig,
    TrainFunctionSettings,
    TotalUpdates,
)


def setup_quantum_circuit(num_qubits, seed):
    """Initialize quantum circuit with parameterized rotations."""
    # np.random.seed(seed)
    rotation_axes = ["rx"] * num_qubits
    rotation_parameters = ParameterVector("θ", num_qubits)
    cm = CouplingMap.from_line(num_qubits, bidirectional=True)
    circuit = circuit_context(
        num_qubits, rotation_axes, rotation_parameters, coupling_map=cm
    )
    return circuit, rotation_parameters, cm


# Add here custom ansatz CX gate
def apply_parametrized_circuit(
    qc: QuantumCircuit, params: ParameterVector, q_reg: QuantumRegister, **kwargs
):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with qiskit_pulse ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :param params: Parameters of the custom Gate
    :param q_reg: Quantum Register formed of target qubits
    :return:
    """
    target = kwargs["target"]
    my_qc = QuantumCircuit(q_reg, name=f"{get_gate(target['gate']).name}_cal")
    optimal_params = np.pi * np.array([0.0, 0.0, 0.5, 0.5, -0.5, 0.5, -0.5])
    # optimal_params = np.pi * np.zeros(len(params))
    new_params = [optimal_params[i] + params[i] for i in range(len(params))]

    my_qc.u(
        *new_params[:3],
        q_reg[0],
    )
    my_qc.u(
        *new_params[3:6],
        q_reg[1],
    )

    my_qc.rzx(new_params[-1], q_reg[0], q_reg[1])

    qc.append(my_qc.to_instruction(label=my_qc.name), q_reg)


def generate_noise_matrix(num_qubits: int, noise_strength_gamma: float) -> np.ndarray:
    """Generate a spillover noise matrix with nearest neighbor interaction of qubits being aligned in a line."""
    gamma_matrix = np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits - 1):
        gamma_matrix[i, i + 1] = noise_strength_gamma
        gamma_matrix[i + 1, i] = noise_strength_gamma
    return gamma_matrix


def apply_noise_pass(circuit, gamma_matrix, param_dict):
    """Apply custom noise pass to the circuit."""
    pm = PassManager(
        [
            LocalSpilloverNoiseAerPass(
                spillover_rate_matrix=numpy_to_hashable(gamma_matrix),
                target_subsystem=(2, 3),
            )
        ]
    )
    return pm.run(circuit.assign_parameters(param_dict))


def define_backend(circuit, gamma_matrix, param_dict):
    """Define the backend with noise model."""
    return noisy_backend(
        circuit.assign_parameters(param_dict), gamma_matrix, target_subsystem=(2, 3)
    )


def train_rl_agent(env, total_updates, training_settings: dict) -> dict:
    """Train RL agent using PPO algorithm."""
    agent_config = PPOConfig.from_yaml(
        "gate_level/spillover_noise_use_case/agent_config.yaml"
    )
    ppo_agent = CustomPPO(agent_config, env, save_data=True)

    ppo_config = TrainingConfig(
        TotalUpdates(total_updates),
        target_fidelities=training_settings["target_fidelities"],
        lookback_window=training_settings["lookback_window"],
        anneal_learning_rate=training_settings["anneal_learning_rate"],
    )
    train_settings = TrainFunctionSettings(
        plot_real_time=training_settings["plot_real_time"],
        print_debug=False,
        num_prints=10,
        hpo_mode=training_settings["hpo_mode"],
        clear_history=training_settings["clear_history"],
    )
    training_results = ppo_agent.train(ppo_config, train_settings)
    return training_results


def plot_learning_curve(env) -> None:
    """Plot RL agent learning curve."""
    reward_history = np.array(env.reward_history)
    mean_rewards = np.mean(reward_history, axis=-1)
    plt.plot(mean_rewards, label="Mean Batch Rewards")
    plt.plot(env.fidelity_history, label="Fidelity")
    plt.legend()
    plt.show()


def save_training_results(all_results: dict, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    # Save combined results as a pickle.gzip file
    results_file_name = params["saving_file_name"] + ".pickle.gz"
    results_file_path = os.path.join(results_dir, results_file_name)
    with gzip.open(results_file_path, "wb") as f:
        pickle.dump(all_results, f)


def create_reward_string(training_settings, use_case_params):
    return f"results_{training_settings['reward_type']}-reward_{use_case_params['target_gate']}-gate_{use_case_params['num_qubits']}-qubits_{use_case_params['gamma_scale']}-gamma"


def main(params: dict) -> None:

    # Setup quantum circuit
    circuit, rotation_parameters, _ = setup_quantum_circuit(
        params["num_qubits"], params["seed"]
    )
    gamma_matrix = generate_noise_matrix(params["num_qubits"], params["gamma_scale"])

    param_dict = {
        theta: val for theta, val in zip(rotation_parameters, params["rotation_angles"])
    }

    noisy_circuit = apply_noise_pass(circuit, gamma_matrix, param_dict)
    # noisy_circuit.draw(output="mpl")
    backend = define_backend(circuit, gamma_matrix, param_dict)

    # Define the RL training environment
    action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    target = {
        "gate": params["target_gate"],
        "physical_qubits": params["physical_qubits"],
    }
    backend_config = QiskitConfig(
        apply_parametrized_circuit,
        backend=backend,
        skip_transpilation=False,
        parametrized_circuit_kwargs={"target": target, "backend": backend},
        pass_manager=None,
    )

    q_env_config = QEnvConfig(
        backend_config=backend_config,
        target=target,
        action_space=action_space,
        execution_config=ExecutionConfig(
            batch_size=params["batch_size"],
            n_reps=params["n_reps"],
            n_shots=params["n_shots"],
            sampling_paulis=params["sampling_paulis"],
            c_factor=params["c_factor"],
        ),
        reward_config=params["reward_type"],
        env_metadata={
            "γ": gamma_matrix,
            "target_subsystem": (2, 3),
            "rotation_axes": ["rx"] * params["num_qubits"],
            "num_qubits": params["num_qubits"],
            "rotation_parameters": rotation_parameters,
            "seed": params["seed"],
        },
    )

    q_env = ContextAwareQuantumEnvironment(q_env_config, circuit_context=noisy_circuit)
    rescaled_env = RescaleAndClipAction(q_env, -1, 1)

    # Train RL agent
    training_results = train_rl_agent(
        rescaled_env, params["total_updates"], training_settings
    )
    training_results["reward_type"] = params["reward_type"]
    training_results["gamma_matrix"] = gamma_matrix

    # Plot learning curve
    # plot_learning_curve(q_env)

    # Combine dictionaries
    combined_results = {
        "training_results": training_results,
        "rl_hyperparams": rl_hyperparams,
        "training_settings": training_settings,
    }
    save_training_results(
        all_results=combined_results, results_dir=params["results_dir"]
    )


if __name__ == "__main__":
    # Pass arguments through main function
    seed = 42
    np.random.seed(seed=seed)

    use_case_params = {
        "num_qubits": 6,
        "target_gate": "cnot",
        "physical_qubits": [0, 1],
        "gamma_scale": 0.05,
        "seed": seed,
    }

    use_case_params["rotation_angles"] = np.random.uniform(
        0, 2 * np.pi, use_case_params["num_qubits"]
    )

    rl_hyperparams = {
        "total_updates": 500,
        "batch_size": 32,
        "n_reps": [4, 7, 9, 12],
        "n_shots": 100,
        "sampling_paulis": 40,
        "c_factor": 1,
        "print_debug": False,
        "num_prints": 10,
        "hpo_mode": False,
        "clear_history": True,
    }
    training_settings = {
        "reward_type": "channel",  # "state", "fidelity", "cafe", "orbit", "channel"
        "target_fidelities": [0.999],
        "lookback_window": 20,
        "anneal_learning_rate": True,
        "plot_real_time": False,
        "hpo_mode": False,
        "clear_history": True,
    }

    saving_results_settings = {
        "results_dir": os.path.join(os.path.dirname(__file__), "training_results"),
        "saving_file_name": create_reward_string(training_settings, use_case_params),
    }

    params = {
        **use_case_params,
        **rl_hyperparams,
        **training_settings,
        **saving_results_settings,
    }

    main(params)
