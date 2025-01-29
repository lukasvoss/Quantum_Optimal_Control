import os
import sys
import gzip
import pickle
project_path = '/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/'
sys.path.append(project_path)
import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.qasm3 import dumps as qasm3_dumps
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
    TotalUpdates
)

def parse_arguments():
    """Parse command-line arguments for script customization."""
    parser = argparse.ArgumentParser(description="Train RL agent for quantum gate calibration.")
    parser.add_argument("--num_qubits", type=int, default=6, help="Number of qubits in the circuit.")
    parser.add_argument("--target_gate", type=str, default="cnot", help="Type of gate to calibrate.")
    parser.add_argument("--gamma_scale", type=float, default=0.05, help="Scaling factor for spillover noise.")
    parser.add_argument("--total_updates", type=int, default=10, help="Total updates for RL training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

def setup_quantum_circuit(num_qubits, seed):
    """Initialize quantum circuit with parameterized rotations."""
    np.random.seed(seed)
    rotation_axes = ["rx"] * num_qubits
    rotation_parameters = ParameterVector("θ", num_qubits)
    cm = CouplingMap.from_line(num_qubits, bidirectional=True)
    circuit = circuit_context(num_qubits, rotation_axes, rotation_parameters, coupling_map=cm)
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

def generate_noise_matrix(num_qubits, scale):
    """Generate a spillover noise matrix."""
    gamma_matrix = scale * np.round(np.random.rand(num_qubits, num_qubits), 3)
    return gamma_matrix

def apply_noise_pass(circuit, gamma_matrix, param_dict):
    """Apply custom noise pass to the circuit."""
    pm = PassManager(
            [LocalSpilloverNoiseAerPass(
                spillover_rate_matrix=numpy_to_hashable(gamma_matrix), target_subsystem=(2, 3)
            )]
        )
    return pm.run(circuit.assign_parameters(param_dict))

def define_backend(circuit, gamma_matrix, param_dict):
    """Define the backend with noise model."""
    return noisy_backend(
        circuit.assign_parameters(param_dict), 
        gamma_matrix, 
        target_subsystem=(2, 3)
    )

def train_rl_agent(env, total_updates, training_settings: dict) -> dict:
    """Train RL agent using PPO algorithm."""
    agent_config = PPOConfig.from_yaml("gate_level/spillover_noise_use_case/agent_config.yaml")
    ppo_agent = CustomPPO(agent_config, env, save_data=False)
    
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

def plot_learning_curve(env):
    """Plot RL agent learning curve."""
    reward_history = np.array(env.reward_history)
    mean_rewards = np.mean(reward_history, axis=-1)
    plt.plot(mean_rewards, label="Mean Batch Rewards")
    plt.plot(env.fidelity_history, label="Fidelity")
    plt.legend()
    plt.show()

def main():
    args = parse_arguments()
    
    # Setup quantum circuit
    circuit, rotation_parameters, cm = setup_quantum_circuit(args.num_qubits, args.seed)
    gamma_matrix = generate_noise_matrix(args.num_qubits, args.gamma_scale)
    
    rotation_angles = np.random.uniform(0, 2 * np.pi, args.num_qubits)
    param_dict = {theta: val for theta, val in zip(rotation_parameters, rotation_angles)}
    
    noisy_circuit = apply_noise_pass(circuit, gamma_matrix, param_dict)
    backend = define_backend(circuit, gamma_matrix, param_dict)
    
    # Define the RL training environment
    action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    target = {"gate": args.target_gate, "physical_qubits": [0, 1]}
    backend_config = QiskitConfig(
        apply_parametrized_circuit,
        backend=backend,
        skip_transpilation=False,
        parametrized_circuit_kwargs={"target": target, "backend": backend},
        pass_manager=None,
    )
    
    ### Hyperparameters
    batch_size = 32
    n_reps = [4, 7, 9, 12]
    n_shots = 100
    sampling_paulis = 40
    c_factor = 1

    training_settings = {
        "target_fidelities": [0.999],
        "lookback_window": 20,
        "anneal_learning_rate": True,
        "plot_real_time": True,
        "hpo_mode": False,
        "clear_history": True,
    }

    q_env_config = QEnvConfig(
        backend_config=backend_config,
        target=target,
        action_space=action_space,
        execution_config=ExecutionConfig(
            batch_size=batch_size, n_reps=n_reps, n_shots=n_shots, sampling_paulis=sampling_paulis, c_factor=c_factor
        ),
        reward_config="cafe",
        env_metadata={
            "γ": gamma_matrix,
            "target_subsystem": (2, 3),
            "rotation_axes": ["rx"] * args.num_qubits,
            "num_qubits": args.num_qubits,
            "rotation_parameters": rotation_parameters,
            "seed": args.seed,
        },
    )
    
    q_env = ContextAwareQuantumEnvironment(
        q_env_config, 
        circuit_context=noisy_circuit
    )
    rescaled_env = RescaleAndClipAction(q_env, -1, 1)

    # Define RL-specific hyperparameters
    rl_hyperparams = {
        "total_updates": args.total_updates,
        "batchsize": batch_size,
        "n_reps": n_reps,
        "n_shots": n_shots,
        "sampling_paulis": sampling_paulis,
        "c_factor": c_factor,
        "plot_real_time": True,
        "print_debug": False,
        "num_prints": 10,
        "hpo_mode": False,
        "clear_history": True,
    }
    
    # Train RL agent
    training_results = train_rl_agent(rescaled_env, args.total_updates, training_settings)
    
    # Plot learning curve
    # plot_learning_curve(q_env)
    
    # Create directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "training_results")
    os.makedirs(results_dir, exist_ok=True)

    # Combine dictionaries
    combined_results = {
        "training_results": training_results,
        "hyperparams": rl_hyperparams,
        "training_settings": training_settings,
    }

    # Save combined results as a pickle.gzip file
    results_file = os.path.join(results_dir, "results.pickle.gz")
    with gzip.open(results_file, "wb") as f:
        pickle.dump(combined_results, f)

    # Output final circuit
    # print(qasm3_dumps(q_env.pubs[6].circuit))

if __name__ == "__main__":
    main()
