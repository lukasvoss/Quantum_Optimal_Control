from itertools import permutations
from typing import Optional, Tuple, List, Union, Dict, Sequence

import numpy as np
import tensorflow as tf
import yaml
from gymnasium.spaces import Box
from qiskit import pulse, schedule, transpile
from qiskit.circuit import QuantumCircuit, Gate, Parameter, CircuitInstruction
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.primitives import BackendEstimator, Estimator, Sampler, BackendSampler
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
from qiskit_aer.backends.aerbackend import AerBackend

from qiskit.providers import BackendV1, Backend, BackendV2, Options as AerOptions
from qiskit.providers.fake_provider.fake_backend import FakeBackend, FakeBackendV2
from qiskit.quantum_info import Operator, Statevector, DensityMatrix
from qiskit.transpiler import (
    CouplingMap,
    InstructionDurations,
    InstructionProperties,
    Layout,
)
from qiskit_dynamics import Solver, RotatingFrame
from qiskit_dynamics.array import Array
from qiskit_dynamics.backend.backend_string_parser.hamiltonian_string_parser import (
    parse_backend_hamiltonian_dict,
)
from qiskit_dynamics.backend.dynamics_backend import (
    _get_backend_channel_freqs,
    DynamicsBackend,
)

from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.framework import BatchExperiment, BaseAnalysis, BackendData
from qiskit_experiments.library import (
    StateTomography,
    ProcessTomography,
    RoughXSXAmplitudeCal,
    RoughDragCal,
)
from qiskit_ibm_provider import IBMBackend

from qiskit_ibm_runtime import (
    Session,
    IBMBackend as RuntimeBackend,
    Estimator as RuntimeEstimator,
    Options as RuntimeOptions,
    Sampler as RuntimeSampler,
)

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

from needed_files_min_example.qconfig import QiskitConfig
from qiskit_braket_provider import BraketLocalBackend, AWSBraketBackend

# from needed_files_min_example.basis_gate_library import EchoedCrossResonance, FixedFrequencyTransmon
# from needed_files_min_example.jax_solver import PauliToQuditOperator, JaxSolver
# from needed_files_min_example.qconfig import QiskitConfig
# from needed_files_min_example.dynamicsbackend_estimator import DynamicsBackendEstimator

# from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon, EchoedCrossResonance

Estimator_type = Union[
    AerEstimator,
    RuntimeEstimator,
    Estimator,
    BackendEstimator,
    # DynamicsBackendEstimator,
]
Sampler_type = Union[AerSampler, RuntimeSampler, Sampler, BackendSampler]
Backend_type = Union[BackendV1, BackendV2]


def constrain_mean_value(mu_var):
    return [tf.clip_by_value(m, -1.0, 1.0) for m in mu_var]


def constrain_std_value(std_var):
    return [tf.clip_by_value(std, 1e-3, 3) for std in std_var]


def count_gates(qc: QuantumCircuit):
    """
    Count number of gates in a Quantum Circuit
    """
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def remove_unused_wires(qc: QuantumCircuit):
    """
    Remove unused wires from a Quantum Circuit
    """
    gate_count = count_gates(qc)
    for qubit, count in gate_count.items():
        if count == 0:
            qc.qubits.remove(qubit)
    return qc



def state_fidelity_from_state_tomography(
        qc_list: List[QuantumCircuit],
        backend: Backend,
        physical_qubits: Optional[Sequence[int]],
        analysis: Union[BaseAnalysis, None, str] = "default",
        target_state: Optional[QuantumState] = None,
        session: Optional[Session] = None,
):
    state_tomo = BatchExperiment(
        [
            StateTomography(
                qc,
                physical_qubits=physical_qubits,
                analysis=analysis,
                target=target_state,
            )
            for qc in qc_list
        ],
        backend=backend,
        flatten_results=True,
    )
    if isinstance(backend, RuntimeBackend):
        jobs = run_jobs(session, state_tomo._transpiled_circuits())
        exp_data = state_tomo._initialize_experiment_data()
        exp_data.add_jobs(jobs)
        exp_data = state_tomo.analysis.run(exp_data).block_for_results()
    else:
        exp_data = state_tomo.run().block_for_results()

    fidelities = [
        exp_data.analysis_result("state_fidelity")[i].value for i in range(len(qc_list))
    ]
    avg_fidelity = np.mean(fidelities)
    return avg_fidelity


def run_jobs(session: Session, circuits: List[QuantumCircuit], run_options=None):
    jobs = []
    runtime_inputs = {"circuits": circuits, "skip_transpilation": True, **run_options}
    jobs.append(session.run("circuit_runner", inputs=runtime_inputs))

    return jobs


def gate_fidelity_from_process_tomography(
        qc_list: List[QuantumCircuit],
        backend: Backend,
        target_gate: Gate,
        physical_qubits: Optional[Sequence[int]],
        analysis: Union[BaseAnalysis, None, str] = "default",
        session: Optional[Session] = None,
):
    """
    Extract average gate and process fidelities from batch of Quantum Circuit for target gate
    """
    # Process tomography
    process_tomo = BatchExperiment(
        [
            ProcessTomography(
                qc,
                physical_qubits=physical_qubits,
                analysis=analysis,
                target=Operator(target_gate),
            )
            for qc in qc_list
        ],
        backend=backend,
        flatten_results=True,
    )

    if isinstance(backend, RuntimeBackend):
        circuits = process_tomo._transpiled_circuits()
        jobs = run_jobs(session, circuits)
        exp_data = process_tomo._initialize_experiment_data()
        exp_data.add_jobs(jobs)
        results = process_tomo.analysis.run(exp_data).block_for_results()
    else:
        results = process_tomo.run().block_for_results()

    process_results = [
        results.analysis_results("process_fidelity")[i].value
        for i in range(len(qc_list))
    ]
    dim, _ = Operator(target_gate).dim
    avg_gate_fid = np.mean([(dim * f_pro + 1) / (dim + 1) for f_pro in process_results])
    return avg_gate_fid


def get_control_channel_map(backend: BackendV1, qubit_tgt_register: List[int]):
    """
    Get reduced control_channel_map from Backend configuration (needs to be of type BackendV1)
    :param backend: IBM Backend instance, must carry a configuration method
    :param qubit_tgt_register: Subsystem of interest from which to build control_channel_map
    """
    control_channel_map = {}
    control_channel_map_backend = {
        qubits: backend.configuration().control_channels[qubits][0].index
        for qubits in backend.configuration().control_channels
    }
    for qubits in control_channel_map_backend:
        if qubits[0] in qubit_tgt_register and qubits[1] in qubit_tgt_register:
            control_channel_map[qubits] = control_channel_map_backend[qubits]
    return control_channel_map


def retrieve_primitives(
        backend: Backend_type,
        layout: Layout,
        config: Union[Dict, QiskitConfig],
        abstraction_level: str = "circuit",
        estimator_options: Optional[Union[Dict, AerOptions, RuntimeOptions]] = None,
) -> (Estimator_type, Sampler_type):
    """
    Retrieve appropriate Qiskit primitives (estimator and sampler) from backend and layout
    """
    if isinstance(
            backend, RuntimeBackend
    ):  # Real backend, or Simulation backend from Runtime Service
        estimator: Estimator_type = RuntimeEstimator(
            session=Session(backend.service, backend),
            options=estimator_options,
        )
        sampler: Sampler_type = RuntimeSampler(
            session=estimator.session, options=estimator_options
        )

        if estimator.options.transpilation["initial_layout"] is None:
            estimator.options.transpilation[
                "initial_layout"
            ] = layout.get_physical_bits()
            sampler.options.transpilation["initial_layout"] = layout.get_physical_bits()

    else:
        if isinstance(estimator_options, RuntimeOptions):
            # estimator_options = asdict(estimator_options)
            estimator_options = None
        if isinstance(backend, (AerBackend, FakeBackend, FakeBackendV2)):
            if abstraction_level != "circuit":
                raise ValueError(
                    "AerSimulator only works at circuit level, and a pulse gate calibration is provided"
                )
            # Estimator taking noise model into consideration, have to provide an AerSimulator backend
            estimator = AerEstimator(
                backend_options=backend.options,
                transpile_options={"initial_layout": layout},
                approximation=True,
            )
            sampler = AerSampler(
                backend_options=backend.options,
                transpile_options={"initial_layout": layout},
            )
        elif backend is None:  # No backend specified, ideal state-vector simulation
            if abstraction_level != "circuit":
                raise ValueError("Statevector simulation only works at circuit level")
            estimator = Estimator(options={"initial_layout": layout})
            sampler = Sampler(options={"initial_layout": layout})

    
        elif isinstance(backend, (BraketLocalBackend, AWSBraketBackend)):
            # Can be used for the SV1 simulator of AWS Braket
            if abstraction_level != "circuit":
                raise ValueError("Statevector simulation only works at circuit level")
            estimator = Estimator(options={"initial_layout": layout})
            sampler = Sampler(options={"initial_layout": layout})

        else:
            raise TypeError("Backend not recognized")
    return estimator, sampler


def set_primitives_transpile_options(
        estimator, sampler, layout, skip_transpilation, physical_qubits
):
    if isinstance(estimator, RuntimeEstimator):
        # TODO: Could change resilience level
        estimator.set_options(
            optimization_level=0,
            resilience_level=0,
            skip_transpilation=skip_transpilation,
        )
        estimator.options.transpilation["initial_layout"] = physical_qubits
        sampler.set_options(**estimator.options)

    elif isinstance(estimator, AerEstimator):
        estimator._transpile_options = AerOptions(
            initial_layout=layout, optimization_level=0
        )
        estimator._skip_transpilation = skip_transpilation
        sampler_transpile_options = AerOptions(
            initial_layout=layout, optimization_level=0
        )
        sampler._skip_transpilation = skip_transpilation

    elif isinstance(estimator, BackendEstimator):
        estimator.set_transpile_options(initial_layout=layout, optimization_level=0)
        estimator._skip_transpilation = skip_transpilation
        sampler.set_transpile_options(initial_layout=layout, optimization_level=0)
        sampler._skip_transpilation = skip_transpilation

    else:
        raise TypeError(
            "Estimator primitive not recognized (must be either BackendEstimator, Aer or Runtime)"
        )





def build_qubit_space_projector(initial_subsystem_dims: list):
    """
    Build projector on qubit space from initial subsystem dimensions
    """
    total_dim = np.prod(initial_subsystem_dims)
    projector = Operator(
        np.zeros((total_dim, total_dim), dtype=np.complex128),
        input_dims=tuple(initial_subsystem_dims),
        output_dims=tuple(initial_subsystem_dims),
    )
    for i in range(total_dim):
        s = Statevector.from_int(i, initial_subsystem_dims)
        for key in s.to_dict().keys():
            if all(c in "01" for c in key):
                projector += s.to_operator()
                break
            else:
                continue
    return projector


def qubit_projection(unitary, subsystem_dims):
    """
    Project unitary on qubit space
    """

    proj = build_qubit_space_projector(subsystem_dims)
    new_dim = 2 ** len(subsystem_dims)
    qubitized_unitary = np.zeros((new_dim, new_dim), dtype=np.complex128)
    qubit_count1 = qubit_count2 = 0
    new_unitary = (
            proj
            @ Operator(
        unitary,
        input_dims=subsystem_dims,
        output_dims=subsystem_dims,
    )
            @ proj
    )
    for i in range(np.prod(subsystem_dims)):
        for j in range(np.prod(subsystem_dims)):
            if new_unitary.data[i, j] != 0:
                qubitized_unitary[qubit_count1, qubit_count2] = new_unitary.data[i, j]
                qubit_count2 += 1
                if qubit_count2 == new_dim:
                    qubit_count2 = 0
                    qubit_count1 += 1
                    break
    qubitized_unitary = Operator(
        qubitized_unitary,
        input_dims=(2,) * len(subsystem_dims),
        output_dims=(2,) * len(subsystem_dims),
    )
    return qubitized_unitary


def load_q_env_from_yaml_file(file_path: str):
    """
    Load Qiskit Quantum Environment from yaml file
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    low = np.array(config["ENV"]["ACTION_SPACE"]["LOW"], dtype=np.float32)
    high = np.array(config["ENV"]["ACTION_SPACE"]["HIGH"], dtype=np.float32)
    params = {
        "action_space": Box(
            low=low, high=high, shape=(config["ENV"]["N_ACTIONS"],), dtype=np.float32
        ),
        "observation_space": Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(config["ENV"]["OBSERVATION_SPACE"],),
            dtype=np.float32,
        ),
        "batch_size": config["ENV"]["BATCH_SIZE"],
        "sampling_Paulis": config["ENV"]["SAMPLING_PAULIS"],
        "n_shots": config["ENV"]["N_SHOTS"],
        "c_factor": config["ENV"]["C_FACTOR"],
        "seed": config["ENV"]["SEED"],
        "benchmark_cycle": config["ENV"]["BENCHMARK_CYCLE"],
        "target": {
            "register": config["TARGET"]["PHYSICAL_QUBITS"],
            "training_with_cal": config["ENV"]["TRAINING_WITH_CAL"],
        },
    }
    if "GATE" in config["TARGET"]:
        params["target"]["gate"] = get_standard_gate_name_mapping()[
            config["TARGET"]["GATE"].lower()
        ]
    else:
        params["target"]["dm"] = DensityMatrix.from_label(config["TARGET"]["STATE"])

    backend_params = {
        "real_backend": config["BACKEND"]["REAL_BACKEND"],
        "backend_name": config["BACKEND"]["NAME"],
        "use_dynamics": config["BACKEND"]["DYNAMICS"]["USE_DYNAMICS"],
        "physical_qubits": config["BACKEND"]["DYNAMICS"]["PHYSICAL_QUBITS"],
        "channel": config["SERVICE"]["CHANNEL"],
        "instance": config["SERVICE"]["INSTANCE"],
    }
    runtime_options = config["RUNTIME_OPTIONS"]
    check_on_exp = config["ENV"]["CHECK_ON_EXP"]
    return params, backend_params, RuntimeOptions(**runtime_options), check_on_exp


def load_agent_from_yaml_file(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    ppo_params = {
        # "n_steps": config["AGENT"]["NUM_UPDATES"],
        "RUN_NAME": config["AGENT"]["RUN_NAME"],
        "NUM_UPDATES": config["AGENT"]["NUM_UPDATES"],
        "N_EPOCHS": config["AGENT"]["N_EPOCHS"],
        "MINIBATCH_SIZE": config["AGENT"]["MINIBATCH_SIZE"],
        "LR": config["AGENT"]["LR_ACTOR"],
        # "lr_critic": config["AGENT"]["LR_CRITIC"],
        "GAMMA": config["AGENT"]["GAMMA"],
        "GAE_LAMBDA": config["AGENT"]["GAE_LAMBDA"],
        "ENT_COEF": config["AGENT"]["ENT_COEF"],
        "V_COEF": config["AGENT"]["V_COEF"],
        "GRADIENT_CLIP": config["AGENT"]["GRADIENT_CLIP"],
        "CLIP_VALUE_LOSS": config["AGENT"]["CLIP_VALUE_LOSS"],
        "CLIP_VALUE_COEF": config["AGENT"]["CLIP_VALUE_COEF"],
        "CLIP_RATIO": config["AGENT"]["CLIP_RATIO"],
    }
    network_params = {
        "OPTIMIZER": config["NETWORK"]["OPTIMIZER"],
        "N_UNITS": config["NETWORK"]["N_UNITS"],
        "ACTIVATION": config["NETWORK"]["ACTIVATION"],
        "INCLUDE_CRITIC": config["NETWORK"]["INCLUDE_CRITIC"],
        "NORMALIZE_ADVANTAGE": config["NETWORK"]["NORMALIZE_ADVANTAGE"],
        "CHKPT_DIR": config["NETWORK"]["CHKPT_DIR"],
    }
    return ppo_params, network_params




def retrieve_tgt_instruction_count(qc: QuantumCircuit, target: Dict):
    tgt_instruction = CircuitInstruction(
        target["gate"], [qc.qubits[i] for i in target["register"]]
    )
    return qc.data.count(tgt_instruction)


def select_optimizer(
        lr: float,
        optimizer: str = "Adam",
        grad_clip: Optional[float] = None,
        concurrent_optimization: bool = True,
        lr2: Optional[float] = None,
):
    if concurrent_optimization:
        if optimizer == "Adam":
            return tf.optimizers.Adam(learning_rate=lr, clipvalue=grad_clip)
        elif optimizer == "SGD":
            return tf.optimizers.SGD(learning_rate=lr, clipvalue=grad_clip)
    else:
        if optimizer == "Adam":
            return tf.optimizers.Adam(learning_rate=lr), tf.optimizers.Adam(
                learning_rate=lr2, clipvalue=grad_clip
            )
        elif optimizer == "SGD":
            return tf.optimizers.SGD(learning_rate=lr), tf.optimizers.SGD(
                learning_rate=lr2, clipvalue=grad_clip
            )


def generate_model(
        input_shape: Tuple,
        hidden_units: Union[List, Tuple],
        n_actions: int,
        actor_critic_together: bool = True,
        hidden_units_critic: Optional[Union[List, Tuple]] = None,
):
    """
    Helper function to generate fully connected NN
    :param input_shape: Input shape of the NN
    :param hidden_units: List containing number of neurons per hidden layer
    :param n_actions: Output shape of the NN on the actor part, i.e. dimension of action space
    :param actor_critic_together: Decide if actor and critic network should be distinct or should be sharing layers
    :param hidden_units_critic: If actor_critic_together set to False, List containing number of neurons per hidden
           layer for critic network
    :return: Model or Tuple of two Models for actor critic network
    """
    input_layer = Input(shape=input_shape)
    Net = Dense(
        hidden_units[0],
        activation="relu",
        input_shape=input_shape,
        kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
        bias_initializer=tf.initializers.RandomNormal(stddev=0.5),
        name=f"hidden_{0}",
    )(input_layer)
    for i in range(1, len(hidden_units)):
        Net = Dense(
            hidden_units[i],
            activation="relu",
            kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
            bias_initializer=tf.initializers.RandomNormal(stddev=0.5),
            name=f"hidden_{i}",
        )(Net)

    mean_param = Dense(n_actions, activation="tanh", name="mean_vec")(
        Net
    )  # Mean vector output
    sigma_param = Dense(n_actions, activation="softplus", name="sigma_vec")(
        Net
    )  # Diagonal elements of cov matrix
    # output

    if actor_critic_together:
        critic_output = Dense(1, activation="linear", name="critic_output")(Net)
        return Model(
            inputs=input_layer, outputs=[mean_param, sigma_param, critic_output]
        )
    else:
        assert (
                hidden_units_critic is not None
        ), "Network structure for critic network not provided"
        input_critic = Input(shape=input_shape)
        Critic_Net = Dense(
            hidden_units_critic[0],
            activation="relu",
            input_shape=input_shape,
            kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
            bias_initializer=tf.initializers.RandomNormal(stddev=0.5),
            name=f"hidden_{0}",
        )(input_critic)
        for i in range(1, len(hidden_units)):
            Critic_Net = Dense(
                hidden_units[i],
                activation="relu",
                kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.initializers.RandomNormal(stddev=0.5),
                name=f"hidden_{i}",
            )(Critic_Net)
            critic_output = Dense(1, activation="linear", name="critic_output")(
                Critic_Net
            )
            return Model(inputs=input_layer, outputs=[mean_param, sigma_param]), Model(
                inputs=input_critic, outputs=critic_output
            )