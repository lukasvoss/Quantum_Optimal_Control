# %%
from braket.circuits import (
    Circuit,
    FreeParameter,
    Instruction,
    gates,
    result_types,
    observables,
)
import warnings
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Measure, Parameter, ParameterVector, ParameterExpression
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

qiskit_gate_names_to_braket_gates: Dict[str, Callable] = {
    "u1": lambda lam: [gates.PhaseShift(lam)],
    "u2": lambda phi, lam: [
        gates.PhaseShift(lam),
        gates.Ry(np.pi / 2),
        gates.PhaseShift(phi),
    ],
    "u3": lambda theta, phi, lam: [
        gates.PhaseShift(lam),
        gates.Ry(theta),
        gates.PhaseShift(phi),
    ],
    "u": lambda theta, phi, lam: [
        gates.PhaseShift(lam),
        gates.Ry(theta),
        gates.PhaseShift(phi),
    ],
    "p": lambda angle: [gates.PhaseShift(angle)],
    "cp": lambda angle: [gates.CPhaseShift(angle)],
    "cx": lambda: [gates.CNot()],
    "x": lambda: [gates.X()],
    "y": lambda: [gates.Y()],
    "z": lambda: [gates.Z()],
    "t": lambda: [gates.T()],
    "tdg": lambda: [gates.Ti()],
    "s": lambda: [gates.S()],
    "sdg": lambda: [gates.Si()],
    "sx": lambda: [gates.V()],
    "sxdg": lambda: [gates.Vi()],
    "swap": lambda: [gates.Swap()],
    "rx": lambda angle: [gates.Rx(angle)],
    "ry": lambda angle: [gates.Ry(angle)],
    "rz": lambda angle: [gates.Rz(angle)],
    "rzz": lambda angle: [gates.ZZ(angle)],
    "id": lambda: [gates.I()],
    "h": lambda: [gates.H()],
    "cy": lambda: [gates.CY()],
    "cz": lambda: [gates.CZ()],
    "ccx": lambda: [gates.CCNot()],
    "cswap": lambda: [gates.CSwap()],
    "rxx": lambda angle: [gates.XX(angle)],
    "ryy": lambda angle: [gates.YY(angle)],
    "ecr": lambda: [gates.ECR()],
}


translatable_qiskit_gates = set(qiskit_gate_names_to_braket_gates.keys()).union(
    {"measure", "barrier", "reset"}
)

def convert_qiskit_to_braket_circuit(circuit: QuantumCircuit) -> Circuit:
    """Return a Braket quantum circuit from a Qiskit quantum circuit.
     Args:
            circuit (QuantumCircuit): Qiskit Quantum Cricuit

    Returns:
        Circuit: Braket circuit
    """
    quantum_circuit = Circuit()
    if not (
        {gate.name for gate, _, _ in circuit.data}.issubset(translatable_qiskit_gates)
    ):
        circuit = transpile(circuit, basis_gates=translatable_qiskit_gates)
    if circuit.global_phase > 1e-10:
        warnings.warn("Circuit transpilation resulted in global phase shift")
    # handle qiskit to braket conversion
    for qiskit_gates in circuit.data:
        name = qiskit_gates[0].name
        if name == "measure":
            # TODO: change Probability result type for Sample for proper functioning # pylint:disable=fixme
            # Getting the index from the bit mapping
            quantum_circuit.add_result_type(
                # pylint:disable=fixme
                result_types.Sample(
                    observable=observables.Z(),
                    target=[
                        circuit.find_bit(qiskit_gates[1][0]).index,
                        circuit.find_bit(qiskit_gates[2][0]).index,
                    ],
                )
            )
        elif name == "barrier":
            # This does not exist
            pass
        elif name == "reset":
            raise NotImplementedError(
                "reset operation not supported by qiskit to braket adapter"
            )
        else:
            params = []
            if hasattr(qiskit_gates[0], "params"):
                params = qiskit_gates[0].params

            ### TODO: Find a way how to extract the individual parameters from the qiskit gate
            for i, param in enumerate(params):
                if isinstance(param, Parameter):
                    params[i] = FreeParameter(param.name)
                elif isinstance(param, ParameterExpression):
                    param_name, _ = list(param._names.items())[0]
                    params[i] = FreeParameter(param_name)

            for gate in qiskit_gate_names_to_braket_gates[name](*params):
                instruction = Instruction(
                    # Getting the index from the bit mapping
                    operator=gate,
                    target=[circuit.find_bit(i).index for i in qiskit_gates[1]],
                )
                quantum_circuit += instruction
    return quantum_circuit

# %%
# def apply_parametrized_circuit(qc: QuantumCircuit):
#     """
#     Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
#     :param qc: Quantum Circuit instance to add the gates on
#     :return:
#     """
#     # qc.num_qubits
#     global n_actions
#     params = ParameterVector('theta', n_actions)
#     qc.u(2 * np.pi * params[0], 2 * np.pi * params[1], 2 * np.pi * params[2], 0)
#     qc.u(2 * np.pi * params[3], 2 * np.pi * params[4], 2 * np.pi * params[5], 1)
#     qc.rzx(2 * np.pi * params[6], 0, 1)

# %%
# n_actions = 7
# qiskit_circuit = QuantumCircuit(2)
# apply_parametrized_circuit(qiskit_circuit)

# braket_circuit = convert_qiskit_to_braket_circuit(qiskit_circuit)

# print(braket_circuit)
