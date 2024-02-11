"""
Class to generate a RL environment suitable for usage with Gym and PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 26/06/2023
"""

from typing import Dict, Optional, List, Any, Tuple, TypeVar, SupportsFloat, Union

import numpy as np

# Qiskit imports
from qiskit import transpile
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ParameterVector,
)
from qiskit.circuit.library import ECRGate
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.states import partial_trace
from qiskit.quantum_info.operators.measures import state_fidelity

from qiskit.quantum_info.states import Statevector, DensityMatrix
from qiskit.transpiler import Layout, InstructionProperties
from qiskit_aer.backends import AerSimulator
from qiskit_aer.backends.aerbackend import AerBackend
from qiskit_dynamics import DynamicsBackend
from qiskit_experiments.calibration_management import (
    Calibrations,
    FixedFrequencyTransmon,
    EchoedCrossResonance,
)

# Qiskit Primitive: for computing Pauli expectation value sampling easily
from qiskit_ibm_runtime import Estimator as RuntimeEstimator

from helper_functions import (
    get_ecr_params,
    set_primitives_transpile_options,
    handle_session,
    projected_statevector,
)
from qconfig import QEnvConfig
from quantumenvironment import QuantumEnvironment, _calculate_chi_target_state
from custom_jax_sim import JaxSolver

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def create_array(circ_trunc, batchsize, n_actions):
    arr = np.empty((circ_trunc,), dtype=object)
    for i in range(circ_trunc):
        arr[i] = np.zeros((i + 1, batchsize, n_actions))
    return arr


class ContextAwareQuantumEnvironment(QuantumEnvironment):
    def __init__(
        self,
        training_config: QEnvConfig,
        circuit_context: QuantumCircuit,
        training_steps_per_gate: Union[List[int], int] = 1500,
        intermediate_rewards: bool = False,
    ):
        """
        Class for wrapping a quantum environment in a Gym environment able to tackle
        context-aware gate calibration. Target gate should be pre-defined through the declaration of a
        QuantumEnvironment, and a circuit in which this gate is used should be provided.
        This class will look for the locations of the target gates in the circuit context and will enable the
        writing of new parametrized circuits, where each gate instance is replaced by a custom gate defined by the
        Callable parametrized_circuit_func in QuantumEnvironment.

        :param training_config: Training configuration containing all hyperparameters characterizing the environment
            for the RL training
        :param circuit_context: QuantumCircuit where target gate is located to perform context aware gate
            calibration.
        :param training_steps_per_gate: Number of calibration steps per gate instance in circuit context
           (for sequential calibration mode)
        :param intermediate_rewards: Specify if calibration within circuit should be sequential (False)
         or across circuit context (True). For now, sequential mode is preferred.
        """

        self._training_steps_per_gate = training_steps_per_gate
        self._inside_trunc_tracker = 0
        self._trunc_index = 0

        super().__init__(training_config)

        assert (
            self.target_type == "gate"
        ), "This class is made for gate calibration only"

        # Retrieve information on the backend for building circuit context workflow

        if self.abstraction_level == "pulse":
            if "cx" in self.backend_info.basis_gates:
                print(
                    "Basis gate Library for CX gate not yet available, will be transpiled over ECR basis gate"
                )
            self._add_ecr_gate(
                self.backend_info.basis_gates, self.backend_info.coupling_map
            )
        # Transpile circuit context to match backend and retreve instruction timings
        self.circuit_context = transpile(
            circuit_context.remove_final_measurements(inplace=False),
            backend=self.backend,
            scheduling_method="asap",
            basis_gates=self.backend_info.basis_gates,
            coupling_map=self.backend_info.coupling_map
            if self.backend_info.coupling_map.size() != 0
            else None,
            instruction_durations=self.backend_info.instruction_durations,
            optimization_level=0,
            dt=self.backend_info.dt,
        )
        # Define target register and nearest neighbor register for truncated circuits
        self.tgt_register = QuantumRegister(
            bits=[self.circuit_context.qubits[i] for i in self.physical_target_qubits],
            name="tgt",
        )
        self.nn_register = QuantumRegister(
            bits=[
                self.circuit_context.qubits[i] for i in self._physical_neighbor_qubits
            ],
            name="nn",
        )
        # Define layout for transpilation
        self.layout = Layout(
            {
                self.tgt_register[i]: self.physical_target_qubits[i]
                for i in range(len(self.tgt_register))
            }
            | {
                self.nn_register[j]: self._physical_neighbor_qubits[j]
                for j in range(len(self.nn_register))
            }
        )

        skip_transpilation = False

        set_primitives_transpile_options(
            self.estimator,
            self.fidelity_checker,
            self.layout,
            skip_transpilation,
            self.physical_target_qubits + self.physical_neighbor_qubits,
        )

        self._d = 2**self.tgt_register.size

        # Adjust target register to match it with circuit context
        self.target_instruction.qubits = tuple(self.tgt_register)
        self._tgt_instruction_counts = self.circuit_context.data.count(
            self.target_instruction
        )

        self._parameters = [
            ParameterVector(f"a_{j}", self.action_space.shape[-1])
            for j in range(self.tgt_instruction_counts)
        ]
        self._param_values = create_array(
            self.tgt_instruction_counts, self.batch_size, self.action_space.shape[-1]
        )

        # Store time and instruction indices where target gate is played in circuit
        self._target_instruction_timings = []
        if self.backend_info.instruction_durations is not None:
            for i, instruction in enumerate(self.circuit_context.data):
                if instruction == self.target_instruction:
                    self._target_instruction_timings.append(
                        self.circuit_context.op_start_times[i]
                    )

        (
            self.circuit_truncations,
            self.baseline_truncations,
        ) = self._generate_circuit_truncations()

        # Storing data
        self._punctual_avg_fidelities = np.zeros(self.batch_size)
        self._punctual_circuit_fidelities = np.zeros(self.batch_size)
        self._punctual_observable: SparsePauliOp = SparsePauliOp("X")
        self.circuit_fidelity_history = []
        self._index_input_state = np.random.randint(len(self.target["input_states"]))
        self._intermediate_rewards = intermediate_rewards
        self._max_return = 0

    @property
    def done(self):
        return self._episode_ended

    @property
    def training_steps_per_gate(self):
        return self._training_steps_per_gate

    @training_steps_per_gate.setter
    def training_steps_per_gate(self, nb_of_steps: int):
        try:
            assert nb_of_steps > 0 and isinstance(nb_of_steps, int)
            self._training_steps_per_gate = nb_of_steps
        except AssertionError:
            raise ValueError("Training steps number should be positive integer.")

    def _retrieve_target_state(self, input_state_index: int, iteration: int) -> Dict:
        """
        Calculate anticipated target state for fidelity estimation based on input state preparation
        and truncation of the circuit
        :param input_state_index: Integer index indicating the PauliPreparationBasis vector
        :param iteration: Truncation index of the transpiled contextual circuit
        """
        # input_circuit: QuantumCircuit = self.target["input_states"][input_state_index]["circuit"]
        input_circuit = self._input_circuits[input_state_index]
        ref_circuit, custom_circuit = (
            self.baseline_truncations[iteration],
            self.circuit_truncations[iteration],
        )
        target_circuit = ref_circuit.compose(input_circuit, inplace=False, front=True)

        # custom_target_circuit.append(input_circuit.to_instruction(), self.tgt_register)
        # target_circuit.append(input_circuit.to_instruction(), self.tgt_register)

        # for gate in ref_circuit.data:  # Append only gates that are applied on the target register
        #     if all([qubit in self.tgt_register for qubit in gate.qubits]):
        #         target_circuit.append(gate)
        return _calculate_chi_target_state(
            {
                "dm": DensityMatrix(target_circuit)
                if len(self.physical_neighbor_qubits) == 0
                else partial_trace(
                    Statevector(target_circuit),
                    list(range(self.tgt_register.size, target_circuit.num_qubits)),
                ),
                "circuit": custom_circuit,
                "target_type": "state",
                "input_state_circ": input_circuit,
            },
            n_qubits=len(self.tgt_register),
        )

    def _generate_circuit_truncations(
        self,
    ) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
        custom_circuit_truncations = [
            QuantumCircuit(
                self.tgt_register, self.nn_register, name=f"c_circ_trunc_{i}"
            )
            for i in range(self.tgt_instruction_counts)
        ]
        baseline_circuit_truncations = [
            QuantumCircuit(
                self.tgt_register, self.nn_register, name=f"b_circ_trunc_{i}"
            )
            for i in range(self.tgt_instruction_counts)
        ]
        custom_gates, custom_gate_circ = [], []
        # Build sub-circuit contexts: each circuit goes until target gate and preserves nearest neighbor operations
        for i in range(self.tgt_instruction_counts):
            counts = 0
            for start_time, instruction in zip(
                self.circuit_context.op_start_times, self.circuit_context.data
            ):
                qubits_in_vicinity = np.array(
                    [
                        (qubit, qubit in self.tgt_register or qubit in self.nn_register)
                        for qubit in instruction.qubits
                    ]
                )
                other_qubits = [
                    qubit[0]
                    for qubit in qubits_in_vicinity
                    if not qubit[1]
                    and qubit[0] not in custom_circuit_truncations[i].qubits
                ]

                # if all([qubit in self.tgt_register or qubit in self.nn_register for qubit in instruction.qubits]):
                if any(qubits_in_vicinity[:, 1]):
                    if (
                        counts <= i or start_time <= self._target_instruction_timings[i]
                    ):  # Append until reaching tgt i
                        if other_qubits:
                            last_reg_name = (
                                baseline_circuit_truncations[i].qregs[-1].name
                            )
                            if last_reg_name != "nn":
                                new_reg = QuantumRegister(
                                    name=last_reg_name
                                    + str(int(last_reg_name[-1]) + 1),
                                    bits=other_qubits,
                                )
                            else:
                                new_reg = QuantumRegister(
                                    name="anc_0", bits=other_qubits
                                )
                            baseline_circuit_truncations[i].add_register(new_reg)
                            custom_circuit_truncations[i].add_register(new_reg)
                            # baseline_circuit_truncations[i].add_bits(other_qubits)
                            # custom_circuit_truncations[i].add_bits(other_qubits)
                        baseline_circuit_truncations[i].append(instruction)

                        if instruction != self.target_instruction:
                            custom_circuit_truncations[i].append(instruction)

                        else:  # Add custom instruction in place of target gate
                            if self.parametrized_circuit_func is not None:
                                self.parametrized_circuit_func(
                                    custom_circuit_truncations[i],
                                    self._parameters[counts],
                                    self.tgt_register,
                                    **self._func_args,
                                )
                            else:
                                custom_circuit_truncations[i].append(instruction)
                            op = custom_circuit_truncations[i].data[-1]
                            if op not in custom_gates:
                                custom_gates.append(op)
                                gate_circ = QuantumCircuit(self.tgt_register)
                                if self.parametrized_circuit_func is not None:
                                    self.parametrized_circuit_func(
                                        gate_circ,
                                        self._parameters[counts],
                                        self.tgt_register,
                                        **self._func_args,
                                    )
                                else:
                                    gate_circ.append(self.target_instruction)
                                custom_gate_circ.append(gate_circ)
                            counts += 1

        return (
            custom_circuit_truncations,
            baseline_circuit_truncations,
            # custom_gate_circ,
        )

    def store_benchmarks(self, params: np.array):
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        :param params: Batch of actions
        """
        n_actions = self.action_space.shape[-1]

        n_custom_instructions = (
            self._trunc_index + 1
        )  # Count custom instructions present in the current truncation
        benchmark_circ = self.circuit_truncations[self._trunc_index].copy(name="b_circ")
        baseline_circ = self.baseline_truncations[self._trunc_index]

        if (
            self.check_on_exp
        ):  # Perform real experiments to retrieve from measurement data fidelities
            # Assess circuit fidelity with ComputeUncompute algo
            try:
                job = self.fidelity_checker.run(
                    [benchmark_circ] * len(params),
                    [baseline_circ] * len(params),
                    values_1=params,
                )
                circuit_fidelities = job.result().fidelities
            except Exception as exc:
                self.close()
                raise exc

            self._punctual_circuit_fidelities = circuit_fidelities
            self.circuit_fidelity_history.append(np.mean(circuit_fidelities))

        else:  # Perform ideal simulation at circuit or pulse level
            if self.abstraction_level == "circuit":
                # Calculate circuit fidelity with statevector simulation
                if isinstance(self.backend, AerBackend):
                    backend = self.backend
                else:
                    backend = AerSimulator.from_backend(
                        self.backend, method="density_matrix"
                    )

                benchmark_circ.save_density_matrix()
                states_result = backend.run(
                    benchmark_circ.decompose(),
                    parameter_binds=[
                        {
                            self._parameters[i][j]: params[:, i * n_actions + j]
                            for i in range(n_custom_instructions)
                            for j in range(n_actions)
                        }
                    ],
                ).result()
                output_states = [
                    states_result.data(i)["density_matrix"]
                    for i in range(self.batch_size)
                ]
                self.update_circuit_fidelity_history(output_states, baseline_circ)

            else:  # Pulse simulation
                # Calculate circuit fidelity with pulse simulation
                if isinstance(self.backend, DynamicsBackend) and isinstance(
                    self.backend.options.solver, JaxSolver
                ):
                    # Jax compatible pulse simulation

                    output_states = np.array(self.backend.options.solver.batched_sims)[
                        :, 1, :
                    ]

                    output_states = [
                        projected_statevector(s, self.backend.options.subsystem_dims)
                        for s in output_states
                    ]

                    self.update_circuit_fidelity_history(output_states, baseline_circ)

                else:
                    raise NotImplementedError(
                        "Pulse simulation not yet implemented for this backend"
                    )

    def _select_trunc_index(self):
        if self._intermediate_rewards:
            return self.step_tracker % self.tgt_instruction_counts
        else:
            return np.min(
                [
                    self._step_tracker // self.training_steps_per_gate,
                    self.tgt_instruction_counts - 1,
                ]
            )

    def episode_length(self, global_step: int):
        """
        Method to retrieve the length of the current episode, i.e. the gate instance that should be calibrated
        """
        assert (
            global_step == self.step_tracker
        ), "Given step not synchronized with internal environment step counter"
        return self._select_trunc_index() + 1

    def _get_info(self) -> Any:
        step = self._episode_tracker
        if self._episode_ended:
            if self.do_benchmark():
                info = {
                    "reset_stage": False,
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "circuit fidelity": self.circuit_fidelity_history[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "max circuit fidelity": np.max(self.circuit_fidelity_history),
                    "arg max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "arg max circuit fidelity": np.argmax(
                        self.circuit_fidelity_history
                    ),
                    "optimal action": self._optimal_action,
                    "truncation_index": self._trunc_index,
                    "input_state": self._input_circuits[self._index_input_state].name,
                    "observable": self._punctual_observable,
                }
            else:
                info = {
                    "reset_stage": False,
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "arg_max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "optimal action": self._optimal_action,
                    "truncation_index": self._trunc_index,
                    "input_state": self._input_circuits[self._index_input_state].name,
                    "observable": self._punctual_observable,
                }
        else:
            info = {
                "reset_stage": self._inside_trunc_tracker == 0,
                "step": step,
                "gate_index": self._inside_trunc_tracker,
                "input_state": self._input_circuits[self._index_input_state].name,
                "truncation_index": self._trunc_index,
                "observable": self._punctual_observable,
            }
        return info

    def _get_obs(self):
        obs = np.array(
            [
                self._index_input_state / len(self.target["input_states"]),
                self._inside_trunc_tracker,
            ]
        )
        # obs = np.array([0., 0])
        return obs

    def perform_action(self, actions):
        """
        Perform action on the environment
        """
        return self.compute_reward()

    def compute_reward(self, fidelity_access=False) -> np.ndarray:
        self._index_input_state = np.random.randint(len(self._input_circuits))
        trunc_index = self._inside_trunc_tracker
        target_state = self._retrieve_target_state(self._index_input_state, trunc_index)
        qc: QuantumCircuit = self.circuit_truncations[trunc_index]

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        self._observables, self._pauli_shots = self.retrieve_observables(
            target_state, qc
        )

        self._punctual_observable = self._observables
        # self.parametrized_circuit_func(training_circ, self._parameters[:iteration])

        reshaped_params = np.reshape(
            np.vstack([param_set for param_set in self._param_values[trunc_index]]),
            (self.batch_size, (trunc_index + 1) * self.action_space.shape[-1]),
        )

        if self.do_benchmark():
            print("Starting benchmarking...")
            self.store_benchmarks(reshaped_params)
            print("Finished benchmarking")
        if fidelity_access:
            reward_table = self._punctual_circuit_fidelities
        else:
            print("Sending job...")
            try:
                self.estimator = handle_session(
                    self.estimator,
                    self.backend,
                    self._session_counts
                    if isinstance(self.estimator, RuntimeEstimator)
                    else trunc_index,
                    qc,
                    target_state["input_state_circ"],
                )

                # Append input state prep circuit to the custom circuit with front composition
                full_circ = qc.compose(
                    target_state["input_state_circ"], inplace=False, front=True
                )
                print(self._observables)

                job = self.estimator.run(
                    circuits=[full_circ] * self.batch_size,
                    observables=[self._observables] * self.batch_size,
                    parameter_values=reshaped_params,
                    shots=int(np.max(self._pauli_shots) * self.n_shots),
                )

                reward_table = job.result().values
            except Exception as exc:
                self.close()
                raise exc
            # scaling_reward_factor = len(self._observables) / 4 ** len(self.tgt_register)
            # reward_table *= scaling_reward_factor
            print("Job done")

        if np.mean(reward_table) > self._max_return:
            self._max_return = np.mean(reward_table)
            self._optimal_action = np.mean(reshaped_params, axis=0)
        self.reward_history.append(reward_table)
        assert (
            len(reward_table) == self.batch_size
        ), f"Reward table size mismatch {len(reward_table)} != {self.batch_size} "

        return reward_table

    def clear_history(self) -> None:
        """Reset all counters related to training"""
        self.step_tracker = 0
        self._episode_tracker = 0
        self.qc_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.avg_fidelity_history.clear()
        self.process_fidelity_history.clear()
        self.built_unitaries.clear()
        self.circuit_fidelity_history.clear()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the Environment, chooses a new input state"""
        super().reset(seed=seed)

        self._param_values = create_array(
            self.tgt_instruction_counts, self.batch_size, self.action_space.shape[0]
        )
        self._inside_trunc_tracker = 0
        self._episode_tracker += 1
        self._episode_ended = False

        if isinstance(self.estimator, RuntimeEstimator):
            self.estimator.options.environment["job_tags"] = [
                f"rl_qoc_step{self._step_tracker}"
            ]

        return self._get_obs(), self._get_info()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # trunc_index tells us which circuit truncation should be trained
        # Dependent on global_step and method select_trunc_index
        trunc_index = self._trunc_index = self._select_trunc_index()

        # Figure out if in middle of param loading or should compute the final reward (step_status < trunc_index or ==)
        step_status = self._inside_trunc_tracker
        self._step_tracker += 1

        if self._episode_ended:
            terminated = True
            return (
                self.reset()[0],
                np.zeros(self.batch_size),
                terminated,
                False,
                self._get_info(),
            )

        if trunc_index >= self.tgt_instruction_counts:
            # raise IndexError(f"Circuit does contain only {self.tgt_instruction_counts} target gates and step"
            #                  f" function tries to access gate nb {trunc_index} ")
            truncated = True
            return (
                self.reset()[0],
                np.zeros(self.batch_size),
                False,
                truncated,
                self._get_info(),
            )

        params, batch_size = np.array(action), len(np.array(action))
        if batch_size != self.batch_size:
            raise ValueError(
                f"Action batch size {batch_size} does not match environment batch size {self.batch_size}"
            )
        self._param_values[trunc_index][step_status] = params

        if step_status < trunc_index:  # Intermediate step within the circuit truncation
            self._inside_trunc_tracker += 1
            terminated = False

            if self._intermediate_rewards:
                reward_table = self.compute_reward()
                obs = reward_table  # Set observation to obtained reward (might not be the smartest choice here)
                return obs, reward_table, terminated, False, self._get_info()
            else:
                return (
                    self._get_obs(),
                    np.zeros(batch_size),
                    terminated,
                    False,
                    self._get_info(),
                )

        else:
            terminated = True
            self._episode_ended = terminated
            reward_table = self.compute_reward()
            if self._intermediate_rewards:
                obs = reward_table
            else:
                obs = self._get_obs()

            # Using Negative Log Error as the Reward
            optimal_error_precision = 1e-6
            max_fidelity = 1.0 - optimal_error_precision
            reward = np.clip(reward_table, a_min=0.0, a_max=max_fidelity)
            reward = -np.log(1.0 - reward)

            return obs, reward, terminated, False, self._get_info()

    def __repr__(self):
        string = QuantumEnvironment.__repr__(self)
        string += f"Number of target gates in circuit context: {self.tgt_instruction_counts}\n"
        return string

    def update_circuit_fidelity_history(self, output_states, baseline_circ):
        batched_dm = DensityMatrix(
            np.mean(
                [state.to_operator().to_matrix() for state in output_states],
                axis=0,
            )
        )
        batched_circuit_fidelity = state_fidelity(
            batched_dm, Statevector(baseline_circ)
        )
        self.circuit_fidelity_history.append(batched_circuit_fidelity)
        circuit_fidelities = [
            state_fidelity(state, Statevector(baseline_circ)) for state in output_states
        ]
        self._punctual_circuit_fidelities = np.array(circuit_fidelities)

    def _add_ecr_gate(self, basis_gates: Optional[List[str]] = None, coupling_map=None):
        """
        Add ECR gate to basis gates if not present
        :param basis_gates: Basis gates of the backend
        :param coupling_map: Coupling map of the backend
        """
        if "ecr" not in basis_gates and self.backend.num_qubits > 1:
            target = self.backend.target
            target.add_instruction(
                ECRGate(),
                properties={qubits: None for qubits in coupling_map.get_edges()},
            )
            cals = Calibrations.from_backend(
                self.backend,
                [
                    FixedFrequencyTransmon(["x", "sx"]),
                    EchoedCrossResonance(["cr45p", "cr45m", "ecr"]),
                ],
                add_parameter_defaults=True,
            )

            for qubit_pair in coupling_map.get_edges():
                if target.has_calibration("cx", qubit_pair):
                    default_params, _, _ = get_ecr_params(self.backend, qubit_pair)
                    error = self.backend.target["cx"][qubit_pair].error
                    target.update_instruction_properties(
                        "ecr",
                        qubit_pair,
                        InstructionProperties(
                            error=error,
                            calibration=cals.get_schedule(
                                "ecr", qubit_pair, default_params
                            ),
                        ),
                    )
            basis_gates.append("ecr")
            for i, gate in enumerate(basis_gates):
                if gate == "cx":
                    basis_gates.pop(i)
            # raise ValueError("Backend must carry 'ecr' as basis_gate for transpilation, will change in the
            # future")
