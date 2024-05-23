from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Any

from quam.components.channels import Channel as QuamChannel
import torch
from gymnasium.spaces import Box
from qiskit import pulse

from qiskit_ibm_runtime import Options
from qiskit.providers import BackendV2
from qiskit.circuit import (
    QuantumCircuit,
    ParameterVector,
    QuantumRegister,
    Gate,
)
from qiskit_dynamics import Solver
from qualang_tools.config.configuration import QMConfiguration

from QUA_files.qua_backend import QMBackend


@dataclass
class BackendConfig(ABC):
    """
    Abstract base class for backend configurations.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a quantum circuit (Qiskit or QUA)
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        parametrized_circuit_kwargs: Additional arguments to feed the parametrized_circuit function

    """

    parametrized_circuit: Callable
    backend: Optional[BackendV2]
    parametrized_circuit_kwargs: Optional[Dict]


@dataclass
class QiskitConfig(BackendConfig):
    """
    Qiskit configuration elements.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QuantumCircuit instance
        estimator_options: Options to feed the Estimator primitive
        solver: Relevant only if dealing with pulse simulation (typically with DynamicsBackend), gives away solver used
        to run simulations for computing exact fidelity benchmark
        channel_freq: Relevant only if dealing with pulse simulation, Dictionary containing information mapping
        the channels and the qubit frequencies
        calibration_files: load existing gate calibrations from json file for DynamicsBackend
        baseline gate calibrations for running algorithm

    """

    parametrized_circuit: Callable[
        [
            QuantumCircuit,
            ParameterVector,
            QuantumRegister,
            Any,
        ],
        None,
    ]
    estimator_options: Optional[Options] = None
    solver: Optional[Solver] = None
    channel_freq: Optional[Dict] = field(default_factory=dict)
    calibration_files: Optional[str] = None
    do_calibrations: bool = True


@dataclass
class QuaConfig(BackendConfig):
    """
    QUA Configuration

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QUA program
        backend: Quantum Machine backend
        hardware_config: Hardware configuration
        channel_mapping: Dictionary mapping channels to quantum elements
    """

    channel_mapping: Dict[
        pulse.channels.Channel, QuamChannel
    ]  # channel to quantum element mapping (e.g. DriveChannel(0) -> 'd0')


@dataclass
class QEnvConfig:
    """
    Quantum Environment configuration. This is used to define all hyperparameters characterizing the Quantum Environment.
    Those include a description of the backend, the action and observation spaces, the batch size (number of actions per
    policy evaluation), the number of Pauli observables to sample for the fidelity estimation scheme,
    the number of shots per Pauli for the fidelity estimation, the renormalization factor, and the device on which the simulation is run.

    Args:
        target (Dict): Target state or target gate to prepare
        backend_config (BackendConfig): Backend configuration
        action_space (Space): Action space
        batch_size (int, optional): Batch size (iterate over a bunch of actions per policy to estimate expected return). Defaults to 50.
        sampling_Paulis (int, optional): Number of Paulis to sample for the fidelity estimation scheme. Defaults to 100.
        n_shots (int, optional): Number of shots per Pauli for the fidelity estimation. Defaults to 1.
        c_factor (float, optional): Renormalization factor. Defaults to 0.5.
        benchmark_cycle (int, optional): Number of epochs between two fidelity benchmarking. Defaults to 5.
        seed (int, optional): Seed for Observable sampling. Defaults to 1234.
        device (Optional[torch.device], optional): Device on which the simulation is run. Defaults to None.
    """

    target: Dict[str, List | Gate | QuantumRegister | QuantumCircuit]
    backend_config: BackendConfig
    action_space: Box
    batch_size: int = 50
    sampling_Paulis: int = 100
    n_shots: int = 1
    n_reps: int = 1
    c_factor: float = 0.5
    benchmark_cycle: int = 1
    seed: int = 1234
    training_with_cal: bool = True
    check_on_exp: bool = False
    channel_estimator: bool = False
    fidelity_access: bool = False
    device: Optional[torch.device] = None
