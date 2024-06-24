import sys
import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
module_path = os.path.abspath(
    os.path.join(
        "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control"
    )
)
if module_path not in sys.path:
    sys.path.append(module_path)
from gymnasium.wrappers import RescaleAction, ClipAction
from gate_level.spillover_noise_use_case.spillover_noise_quantum_environment import SpilloverNoiseQuantumEnvironment
from gymnasium.wrappers import RescaleAction, ClipAction
from rl_qoc.helper_functions import load_from_yaml_file, save_to_pickle
from gate_level_abstraction.spillover_noise_use_case.spillover_noise_q_env_config_function import (
    setup_spillover_noise_qenv_config,
)
from rl_qoc import CustomPPO, HyperparameterOptimizer, QuantumEnvironment, ContextAwareQuantumEnvironment
from rl_qoc.ppo_config import (
    TotalUpdates,
    HardwareRuntime,
    TrainingConfig,
    TrainFunctionSettings,
)
from rl_qoc.ppo_config import (
    TotalUpdates,
    HardwareRuntime,
    TrainingConfig,
    TrainFunctionSettings,
)
from rl_qoc.hpo_config import HardwarePenaltyWeights, HPOConfig, DirectoryPaths

def get_saving_dir(hpo_mode: bool = False, phi_gamma_tuple: Optional[Tuple[float, float]] = None, base_dir: str = 'gate_level') -> str:
    # Determine the folder based on the conditions
    if phi_gamma_tuple is not None and not any(phi_gamma_tuple == 0):
        # Noisy case
        if hpo_mode:
            target_folder = os.path.join(base_dir, 'spillover_noise_use_case', 'hpo_results')
        else:
            target_folder = os.path.join(base_dir, 'spillover_noise_use_case', 'calibration_results')
    else:
        # Noise-free case
        if hpo_mode:
            target_folder = os.path.join(base_dir, 'standard', 'hpo_results')
        else:
            target_folder = os.path.join(base_dir, 'standard', 'calibration_results')

    return target_folder

def get_environment(
    config_paths: Dict[str, str],
    use_context: bool = None,
    phi_gamma_tuple: Optional[Tuple[float, float]] = None,
):

    print("\nLoading Environment")
    print("----------------------------------------\n")
    print(f"Use context: {use_context}")
    (
        print("Spillover Noise Use Case with phi_gamma_tuple: ", phi_gamma_tuple)
        if phi_gamma_tuple is not None
        else None
    )
    time.sleep(4)

    if phi_gamma_tuple is not None and use_context is True:
        gate_q_env_config, circuit_context = setup_spillover_noise_qenv_config(
            phi_gamma_tuple, config_paths["noise_q_env_config_file"]
        )

        q_env = SpilloverNoiseQuantumEnvironment(
            gate_q_env_config, circuit_context, phi_gamma_tuple
        )
    else:
        from q_env_config import q_env_config as gate_q_env_config, circuit_context

        circuit_context.draw("mpl")

        if use_context:
            q_env = ContextAwareQuantumEnvironment(
                gate_q_env_config, circuit_context, training_steps_per_gate=250
            )
        else:
            q_env = QuantumEnvironment(gate_q_env_config)

    return RescaleAction(ClipAction(q_env), -1.0, 1.0)


def do_hpo(
    rescaled_env,
    num_hpo_trials,
    training_config,
    train_function_settings,
    experimental_penalty_weights,
    directory_paths,
):
    hpo_config = HPOConfig(
        q_env=rescaled_env,
        num_trials=num_hpo_trials,
        hardware_penalty_weights=experimental_penalty_weights,
        hpo_paths=directory_paths,
    )
    hpo_engine = HyperparameterOptimizer(
        hpo_config=hpo_config,
    )
    results = hpo_engine.optimize_hyperparameters(
        training_config=training_config, train_function_settings=train_function_settings
    )
    return results


def do_training(
    rescaled_env, training_config, train_function_settings, agent_file_path
):
    agent_config = load_from_yaml_file(agent_file_path)

    ppo_agent = CustomPPO(agent_config, rescaled_env)

    results = ppo_agent.train(
        training_config=training_config, train_function_settings=train_function_settings
    )
    return results


def main(
    training_config: TrainingConfig,
    train_function_settings: TrainFunctionSettings,
    config_paths: Dict[str, str],
    use_context: bool = False,
    phi_gamma_tuple: Optional[Tuple[float, float]] = None,
    **kwargs,
):

    if "hpo_mode" in kwargs and kwargs["hpo_mode"] and "num_hpo_trials" in kwargs:
        num_hpo_trials = kwargs["num_hpo_trials"]
        experimental_penalty_weights = kwargs["experimental_penalty_weights"]
        directory_paths = kwargs["directory_paths"]
        # hpo_agent_path = directory_paths.agent_config_path
        rescaled_env = get_environment(
            config_paths=config_paths,
            use_context=use_context,
            phi_gamma_tuple=phi_gamma_tuple,
        )
        do_hpo(
            rescaled_env,
            num_hpo_trials,
            training_config,
            train_function_settings,
            experimental_penalty_weights,
            directory_paths,
        )
    else:
        rescaled_env = get_environment(
            config_paths=config_paths,
            use_context=use_context,
            phi_gamma_tuple=phi_gamma_tuple,
        )
        training_results = do_training(
            rescaled_env, training_config, train_function_settings, config_paths['agent_config_file']
        )
        saving_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            config_paths["save_results_path"],
            rescaled_env.unwrapped.ident_str
            + f'_timestamp_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.pickle.gz',
        )
        print('Saving to:', saving_path)
        save_to_pickle(training_results, saving_path)


if __name__ == "__main__":

    ################# TO BE SET BY USER #################
    
    """ HPO Settings """
    hpo_mode = True
    num_hpo_trials = 2

    """ Training Settings """
    use_context = False # True
    phi_gamma_tuple = None # (0.1*np.pi, 0.1)

    ######################################################
    
    
    
    file_paths = {
        "agent_config_file": "gate_level/standard/agent_config.yaml",
        "noise_q_env_config_file": "gate_level/spillover_noise_use_case/noise_q_env_gate_config.yml",
        "save_results_path": get_saving_dir(hpo_mode, phi_gamma_tuple, 'gate_level'),
    }

    experimental_penalty_weights = HardwarePenaltyWeights(
        shots_penalty=0.01,
        missed_fidelity_penalty=1e4,
        fidelity_reward=2 * 1e4,
    )
    directory_paths = DirectoryPaths(
        agent_config_path="/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/gate_level/spillover_noise_use_case/agent_config.yaml",
        hpo_config_path="/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/gate_level/spillover_noise_use_case/noise_hpo_config.yaml",
        save_results_path=get_saving_dir(hpo_mode, phi_gamma_tuple, 'gate_level'),
    )

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

    main(
        training_config,
        train_function_settings,
        file_paths,
        use_context,
        phi_gamma_tuple,
        hpo_mode=hpo_mode,
        num_hpo_trials=num_hpo_trials,
        experimental_penalty_weights=experimental_penalty_weights,
        directory_paths=directory_paths,
    )