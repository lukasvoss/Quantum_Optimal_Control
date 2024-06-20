import sys
import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
module_path = os.path.abspath(
    os.path.join(
        "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control"
    )
)
if module_path not in sys.path:
    sys.path.append(module_path)
from quantumenvironment import QuantumEnvironmentV2
from context_aware_quantum_environment import ContextAwareQuantumEnvironmentV2
from gate_level_abstraction.spillover_noise_use_case.spillover_noise_quantum_environment import SpilloverNoiseQuantumEnvironment
from gymnasium.wrappers import RescaleAction, ClipAction
from ppoV2 import CustomPPOV2
from helper_functions import load_from_yaml_file
from gate_level_abstraction.spillover_noise_use_case.spillover_noise_q_env_config_function import setup_spillover_noise_qenv_config


from hpo_training_config import (
    TotalUpdates,
    HardwareRuntime,
    TrainingConfig,
    TrainFunctionSettings,
)

def get_environment(config_paths: Dict[str, str], use_context: bool = None, phi_gamma_tuple: Optional[Tuple[float, float]] = None):
    
    print('\nLoading Environment')
    print('----------------------------------------\n')
    print(f'Use context: {use_context}')
    print('Spillover Noise Use Case with phi_gamma_tuple: ', phi_gamma_tuple) if phi_gamma_tuple is not None else None
    time.sleep(4)

    if phi_gamma_tuple is not None and use_context is True:
        gate_q_env_config, circuit_context = setup_spillover_noise_qenv_config(phi_gamma_tuple, config_paths['noise_q_env_config_file'])

        q_env = SpilloverNoiseQuantumEnvironment(
            gate_q_env_config, circuit_context, phi_gamma_tuple
        )
    else:
        from template_configurations.qiskit.gate_level import (
            q_env_config as gate_q_env_config,
            circuit_context,
        )
        circuit_context.draw("mpl")
        
        if use_context:
            q_env = ContextAwareQuantumEnvironmentV2(
                gate_q_env_config, circuit_context, training_steps_per_gate=250
            )
        else:
            q_env = QuantumEnvironmentV2(gate_q_env_config)
    
    return RescaleAction(ClipAction(q_env), -1.0, 1.0)


def main(
    training_config: TrainingConfig,
    train_function_settings: TrainFunctionSettings,
    config_paths: Dict[str, str],
    use_context: bool = False,
    phi_gamma_tuple: Optional[Tuple[float, float]] = None,
):
    rescaled_env = get_environment(config_paths=config_paths, use_context=use_context, phi_gamma_tuple=phi_gamma_tuple)
    agent_config = load_from_yaml_file(config_paths["agent_config_file"])

    ppo_agent = CustomPPOV2(agent_config, rescaled_env)

    training_results = ppo_agent.train(
        training_config=training_config, train_function_settings=train_function_settings
    )

    return training_results


if __name__ == "__main__":

    file_paths = {
        "agent_config_file": "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/agent_config.yaml",
        "noise_q_env_config_file": "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/gate_level_abstraction/spillover_noise_use_case/noise_q_env_gate_config.yml",
    }
    total_updates = TotalUpdates(10)
    # hardware_runtime = HardwareRuntime(300)
    training_config = TrainingConfig(
        training_constraint=total_updates,
        target_fidelities=[0.999, 0.9999],
        lookback_window=10,
        anneal_learning_rate=False,
        std_actions_eps=1e-2,
    )

    train_function_settings = TrainFunctionSettings(
        plot_real_time=False,
        print_debug=True,
        num_prints=1,
        hpo_mode=False,
        clear_history=True,
    )

    use_context = True
    phi_gamma_tuple = (0.1, 0.1)

    training_results = main(training_config, train_function_settings, file_paths, use_context, phi_gamma_tuple)

    plt.plot(training_results["fidelity_history"])
    plt.show()