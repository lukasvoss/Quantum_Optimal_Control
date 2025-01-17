from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union, Callable, Any, Optional, Dict, Iterable, List

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector

from qiskit.primitives import EstimatorPubLike, SamplerPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.providers import BackendV2
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit_ibm_runtime import OptionsV2
from ..calibration_pubs import (
    CalibrationEstimatorPubLike,
    CalibrationSamplerPubLike,
    CalibrationEstimatorPub,
    CalibrationSamplerPub,
)

PubLike = Union[
    EstimatorPubLike,
    SamplerPubLike,
    CalibrationEstimatorPubLike,
    CalibrationSamplerPubLike,
]
Pub = Union[EstimatorPub, SamplerPub, CalibrationEstimatorPub, CalibrationSamplerPub]


@dataclass
class BackendConfig(ABC):
    """
    Abstract base class for backend configurations.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a quantum circuit (Qiskit or QUA)
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        parametrized_circuit_kwargs: Additional arguments to feed the parametrized_circuit function
        pass_manager: Pass manager to transpile the circuit
        instruction_durations: Dictionary containing the durations of the instructions in the circuit

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
    backend: Optional[BackendV2] = None
    parametrized_circuit_kwargs: Dict = field(default_factory=dict)
    skip_transpilation: bool = False
    pass_manager: Optional[PassManager] = None
    instruction_durations: Optional[InstructionDurations] = None

    @property
    @abstractmethod
    def config_type(self):
        return "backend"

    def process_pubs(self, pubs: Iterable[Pub | PubLike]) -> Iterable[Pub]:
        """
        Process the pub to the correct type for the backend
        """
        return pubs

    def as_dict(self):
        return {
            "parametrized_circuit": self.parametrized_circuit,
            "backend": self.backend,
            "parametrized_circuit_kwargs": self.parametrized_circuit_kwargs,
            "pass_manager": self.pass_manager,
            "instruction_durations": self.instruction_durations,
        }


@dataclass
class QiskitConfig(BackendConfig):
    """
    Qiskit configuration elements.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a quantum circuit (Qiskit or QUA)
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        parametrized_circuit_kwargs: Additional arguments to feed the parametrized_circuit function
        pass_manager
        instruction_durations: Dictionary containing the durations of the instructions in the circuit
    """

    @property
    def config_type(self):
        return "qiskit"

    def process_pubs(self, pubs: Iterable[Pub | PubLike]) -> List[Pub]:
        """
        Process the pub to the correct type for the backend
        """
        new_pubs = []
        for pub in pubs:
            if isinstance(pub, (CalibrationEstimatorPubLike, CalibrationEstimatorPub)):
                new_pubs.extend(CalibrationEstimatorPub.coerce(pub).to_pub_list())
            elif isinstance(pub, (CalibrationSamplerPubLike, CalibrationSamplerPub)):
                new_pubs.extend(CalibrationSamplerPub.coerce(pub).to_pub_list())
            elif isinstance(pub, (EstimatorPubLike, EstimatorPub)):
                new_pubs.append(EstimatorPub.coerce(pub))
            elif isinstance(pub, (SamplerPubLike, SamplerPub)):
                new_pubs.append(SamplerPub.coerce(pub))
            else:
                raise ValueError(f"Pub type {type(pub)} not recognized")
        return new_pubs


@dataclass
class DynamicsConfig(QiskitConfig):
    """
    Qiskit Dynamics configuration elements.

    Args:
        the channels and the qubit frequencies
        calibration_files: load existing gate calibrations from json file for DynamicsBackend
        baseline gate calibrations for running algorithm

    """

    calibration_files: Optional[str] = None
    do_calibrations: bool = True

    @property
    def config_type(self):
        return "dynamics"

    def as_dict(self):
        return super().as_dict() | {
            "calibration_files": self.calibration_files,
            "do_calibrations": self.do_calibrations,
        }


@dataclass
class QiskitRuntimeConfig(QiskitConfig):
    """
    Qiskit Runtime configuration elements.

    Args:
        primitive_options: Options to feed the Qiskit Runtime job
    """

    primitive_options: Optional[OptionsV2] = None

    @property
    def config_type(self):
        return "runtime"

    def as_dict(self):
        return super().as_dict() | {
            "primitive_options": self.primitive_options,
        }
