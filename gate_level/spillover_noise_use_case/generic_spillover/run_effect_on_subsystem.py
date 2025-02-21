import os
import sys
import gzip
import pickle
import datetime
import smtplib
from email.message import EmailMessage
from typing import Dict, Tuple

# project_path = "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/"
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
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
from rl_qoc.hpo.hpo_config import (
    QUANTUM_ENVIRONMENT,
    DirectoryPaths,
    HPOConfig,
    HardwarePenaltyWeights,
)
from rl_qoc.hpo.hyperparameter_optimization import HyperparameterOptimizer

# Import the Google Cloud Storage client
from google.cloud import storage

#############################################
# New functions for file upload & notification
#############################################

def upload_file_to_gcs(file_path: str, bucket_name: str, destination_blob_name: str) -> str:
    """
    Uploads a file to a GCS bucket and returns a signed URL valid for 1 year (52 weeks).
    """
    bucket = get_or_create_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(file_path)
    
    # Generate a signed URL valid for 1 hour (adjust expiration as needed)
    signed_url = blob.generate_signed_url(expiration=datetime.timedelta(weeks=52))
    return signed_url

def send_email(subject: str, body: str, from_email: str, to_email: str,
               smtp_server: str = 'smtp.gmail.com', smtp_port: int = 587,
               login: str = None, password: str = None):
    """
    Sends an email notification with the given subject and body.
    """
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(login, password)
        server.send_message(msg)

def get_or_create_bucket(bucket_name: str):
    storage_client = storage.Client()
    bucket = storage_client.lookup_bucket(bucket_name)
    if bucket is None:
        print(f"Bucket {bucket_name} not found. Creating bucket...")
        bucket = storage_client.create_bucket(bucket_name)
    return bucket

def notify_completion(results_file_path: str):
    """
    Uploads the results file to GCS, generates a signed URL, and sends an email notification.
    """
    # Replace these placeholder values with your actual configuration
    bucket_name = "rl_training_results"
    from_email = "rlqoc.project@gmail.com"
    to_email = ["lukas_voss@icloud.com", "arthur.strauss@u.nus.edu"]
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    login = "rlqoc.project@gmail.com"
    password = os.getenv("EMAIL_PASSWORD") # fqjy talu uylo buti

    print("results_file_path: ", results_file_path, "type: ", type(results_file_path))
    destination_blob_name = os.path.basename(results_file_path)
    signed_url = upload_file_to_gcs(results_file_path, bucket_name, destination_blob_name)
    
    subject = "Test Email: RL Training Run Finished - Results Available on Google Cloud Bucket (Automated email by LV)"
    body = (f"Your HPO for the {params['reward_type'].upper()} Reward completed: {params['num_hpo_trials']} trials with {params['total_updates']} update steps each.\n\n"
            f"Download your results file here (link valid for 1 year): \n{signed_url}")
    
    send_email(subject, body, from_email, to_email, smtp_server, smtp_port, login, password)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run RL training for quantum control.")
    parser.add_argument(
        "--reward_type",
        type=str,
        choices=["state", "fidelity", "cafe", "orbit", "channel"],
        required=True,
        help="Type of reward to use for training the RL agent.",
    )
    args = parser.parse_args()
    return args

def setup_quantum_circuit(
    num_qubits: int,
) -> Tuple[QuantumCircuit, ParameterVector, CouplingMap]:
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


def apply_noise_pass(
    circuit: QuantumCircuit, gamma_matrix: np.ndarray, param_dict: Dict
):
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


def define_backend(circuit: QuantumCircuit, gamma_matrix: np.ndarray, param_dict: Dict):
    """Define the backend with noise model."""
    return noisy_backend(
        circuit.assign_parameters(param_dict), gamma_matrix, target_subsystem=(2, 3)
    )


def build_rl_environment(params: Dict) -> QUANTUM_ENVIRONMENT:
    # Setup quantum circuit
    circuit, rotation_parameters, _ = setup_quantum_circuit(params["num_qubits"])
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
    return rescaled_env, gamma_matrix


def prepare_hpo_config(
    env: QUANTUM_ENVIRONMENT, ppo_agent: CustomPPO, training_settings: Dict
) -> HPOConfig:
    # Prepare HPO config

    path_agent_config = (
        f"{project_path}/gate_level/spillover_noise_use_case/agent_config.yaml"
    )
    path_hpo_config = (
        f"{project_path}/gate_level/spillover_noise_use_case/noise_hpo_config.yaml"
    )
    save_results_path = f"gate_level/spillover_noise_use_case/hpo_results_feb2025_numpy-seed-{seed}"

    # Hardware penalty weights are currenlty not used as the cost function is the infidelity (see HPO class for reference)
    experimental_penalty_weights = HardwarePenaltyWeights(
        shots_penalty=0.01,
        missed_fidelity_penalty=1e4,
        fidelity_reward=2 * 1e4,
    )

    directory_paths = DirectoryPaths(
        agent_config_path=path_agent_config,
        hpo_config_path=path_hpo_config,
        save_results_path=save_results_path,
    )

    hpo_config = HPOConfig(
        q_env=env,
        agent=ppo_agent,
        num_trials=training_settings["num_hpo_trials"],
        hardware_penalty_weights=experimental_penalty_weights,
        hpo_paths=directory_paths,
    )
    return hpo_config


def run_hpo_training(
    env: QUANTUM_ENVIRONMENT,
    ppo_agent: CustomPPO,
    ppo_config: TrainingConfig,
    train_settings: TrainFunctionSettings,
    training_settings: Dict,
) -> Dict:
    """Run training in HPO mode."""
    hpo_config = prepare_hpo_config(env, ppo_agent, training_settings)
    hpo_engine = HyperparameterOptimizer(
        hpo_config=hpo_config, callback=print_summary_callback
    )
    training_results = hpo_engine.optimize_hyperparameters(
        training_config=ppo_config, train_function_settings=train_settings
    )

    return training_results, hpo_engine.saved_results_path


def run_standard_training(
    ppo_agent: CustomPPO,
    ppo_config: TrainingConfig,
    train_settings: TrainFunctionSettings,
) -> Dict:
    """Run standard training (without HPO)."""
    training_results = ppo_agent.train(ppo_config, train_settings)
    return training_results


def post_process_and_save(
    training_results: Dict, params: Dict, gamma_matrix: np.ndarray
):
    # if isinstance(training_results, list):
    #     for result in training_results:
    #         result["reward_type"] = params["reward_type"]
    #         result["gamma_matrix"] = gamma_matrix
    # elif isinstance(training_results, dict):
    #     training_results["reward_type"] = params["reward_type"]
    #     training_results["gamma_matrix"] = gamma_matrix

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


# Define your callback function for the HPO
def print_summary_callback(trial_data, **kwargs):
    print(f"Trial number: {trial_data['trial_number']}")
    print(f"Max Fidelity: {max(trial_data['training_results']['fidelity_history'])}")
    print(
        f"Hardware Runtime: {round(sum(trial_data['training_results']['hardware_runtime']), 2)} seconds"
    )
    print(f"Custom cost value: {round(trial_data['custom_cost_value'], 2)}")
    print(f"Simulation runtime: {round(trial_data['simulation runtime'], 2)} seconds")
    for key, value in kwargs.items():
        print(f"{key}: {value}")


def train_rl_agent(
    env: QUANTUM_ENVIRONMENT, training_settings: Dict, gamma_matrix: np.ndarray
) -> None:
    """Train RL agent using PPO algorithm."""
    agent_config = PPOConfig.from_yaml(
        "gate_level/spillover_noise_use_case/agent_config.yaml"
    )
    ppo_agent = CustomPPO(
        agent_config, env, save_data=training_settings["log_training_wandb"]
    )

    ppo_config = TrainingConfig(
        TotalUpdates(training_settings["total_updates"]),
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

    if training_settings["hpo_mode"]:
        training_results, results_file_path = run_hpo_training(env, ppo_agent, ppo_config, train_settings, training_settings)
        post_process_and_save(training_results, params, gamma_matrix)
        # Notify via email that the run is complete and include the signed download link.
        notify_completion(results_file_path)
    else:
        training_results = run_standard_training(ppo_agent, ppo_config, train_settings)
        results_file_path = params["saving_file_name"] + ".pickle.gz"
        post_process_and_save(training_results, params, gamma_matrix)


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


def create_file_name(training_settings, use_case_params):
    return f"results_{training_settings['reward_type']}-reward_{use_case_params['target_gate']}-gate_{use_case_params['num_qubits']}-qubits_{use_case_params['gamma_scale']}-gamma"


def main(params: Dict) -> None:
    rescaled_env, gamma_matrix = build_rl_environment(params)
    train_rl_agent(rescaled_env, params, gamma_matrix)


if __name__ == "__main__":
    args = parse_arguments()

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
        "batch_size": 32,
        "n_reps": [4, 7, 9, 12],
        "n_shots": 100,
        "sampling_paulis": 40,
        "c_factor": 1,
    }
    training_settings = {
        "total_updates": 10,
        "reward_type": args.reward_type,  # "state", "fidelity", "cafe", "orbit", "channel"
        "target_fidelities": [0.999, 0.9999],
        "lookback_window": 20,
        "anneal_learning_rate": True,
        "plot_real_time": False,
        "clear_history": True,
        "log_training_wandb": True,  # TODO: Set to True to log training metrics to wandb dashboard (online)
        "print_debug": False,
        "num_prints": 10,
    }
    # TODO: Configure HPO settings
    training_settings["hpo_mode"] = True
    training_settings["num_hpo_trials"] = 2

    saving_results_settings = {
        "results_dir": os.path.join(os.path.dirname(__file__), "training_results"),
        "saving_file_name": create_file_name(training_settings, use_case_params),
    }

    params = {
        **use_case_params,
        **rl_hyperparams,
        **training_settings,
        **saving_results_settings,
    }

    main(params)
