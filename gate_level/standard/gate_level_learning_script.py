from dataclasses import asdict
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
from rl_qoc.helper_functions import create_custom_file_name, load_from_yaml_file, save_to_pickle
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

def get_saving_dir(hpo_mode: bool = False, phi_gamma_tuple: Optional[Tuple[float, float]] = None, base_dir: str = '/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/gate_level') -> str:
    # Determine the folder based on the conditions
    if phi_gamma_tuple is not None and not any(x == 0 for x in phi_gamma_tuple):
        parent_folder = 'spillover_noise_use_case'
    else:
        parent_folder = 'standard'
    child_folder = 'hpo_results' if hpo_mode else 'calibration_results'
    return os.path.join(base_dir, parent_folder, child_folder)

def print_env_info(q_env):

    print("\nEnvironment Information")
    print("----------------------------------------\n")
    print(f"Reward Settings:  {asdict(q_env.unwrapped.config.reward_config)}\n")
    print(f"Execution Settings:  {asdict(q_env.unwrapped.config.execution_config)}\n")
    time.sleep(4)

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

        if use_context:
            q_env = ContextAwareQuantumEnvironment(
                gate_q_env_config, circuit_context, training_steps_per_gate=250
            )
        else:
            q_env = QuantumEnvironment(gate_q_env_config)

    print_env_info(q_env)

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

        custom_saving_file_name = constraint_str + create_custom_file_name(config_paths["q_env_config_file"])

        saving_path = os.path.join(
            config_paths["save_results_dir"],
            custom_saving_file_name +
            f'_timestamp_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.pickle.gz',
        )
        print('Saving to:', saving_path)
        print('Results:', training_results)
        save_to_pickle(training_results, saving_path)


if __name__ == "__main__":

    ################# TO BE SET BY USER #################
    
    """ HPO Settings """  # Cost Function has been changed to infidelity; n_reps hardcoded to 1 (only for automatic calculation of nreps due to noise)
    hpo_mode = False
    num_hpo_trials = 20

    """ Training Settings """
    use_context = True # False
    phi_gamma_tuple = (np.pi/4, 0.025) # None

    ######################################################
    
    
    
    file_paths = {
        "agent_config_file": "gate_level/standard/agent_config.yaml",
        "q_env_config_file": "gate_level/standard/q_env_gate_config.yml",
        "noise_q_env_config_file": "gate_level/spillover_noise_use_case/noise_q_env_gate_config.yml",
        "save_results_dir": get_saving_dir(hpo_mode, phi_gamma_tuple, 'gate_level'),
    }

    experimental_penalty_weights = HardwarePenaltyWeights(
        shots_penalty=0.01,
        missed_fidelity_penalty=1e4,
        fidelity_reward=2 * 1e4,
    )
    directory_paths = DirectoryPaths(
        agent_config_path="gate_level/spillover_noise_use_case/agent_config.yaml",
        hpo_config_path="gate_level/spillover_noise_use_case/noise_hpo_config.yaml",
        save_results_path=get_saving_dir(hpo_mode, phi_gamma_tuple),
    )

    total_updates = TotalUpdates(250)
    hardware_runtime = HardwareRuntime(300)
    training_config = TrainingConfig(
        training_constraint=total_updates,
        target_fidelities=[0.999, 0.9999, 0.99999],
        lookback_window=10,
        anneal_learning_rate=False,
        std_actions_eps=2.5e-2,
    )
    constraint_str = f'updates_{total_updates.total_updates}_' if isinstance(training_config.training_constraint, TotalUpdates) else f'runtime_{hardware_runtime.hardware_runtime}s_'
    train_function_settings = TrainFunctionSettings(
        plot_real_time=False,
        print_debug=True,
        num_prints=1,
        hpo_mode=hpo_mode,
        clear_history=True,
    )

    main(
        training_config,
        train_function_settings,
        file_paths,
        use_context,
        phi_gamma_tuple,
        constraint_str=constraint_str,
        hpo_mode=hpo_mode,
        num_hpo_trials=num_hpo_trials,
        experimental_penalty_weights=experimental_penalty_weights,
        directory_paths=directory_paths,
    )