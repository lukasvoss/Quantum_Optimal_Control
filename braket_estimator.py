from typing import Union, Optional
import numpy as np

from braket.circuits import Circuit, Observable
from braket.circuits.observables import TensorProduct
from braket.quantum_information import PauliString


class BraketEstimator:
    def __init__(self, backend):
        self.backend = backend

    def _validate_inputs(self, circuit, bound_parameters):
        """
        Validate the inputs to ensure the number of circuits matches the number of bound_parameters.

        :param circuit: A single QuantumCircuit or a list of QuantumCircuits.
        :param bound_parameters: A single dictionary of parameters or a list of dictionaries.
        :raises ValueError: If the number of circuits and parameter dictionaries do not match.
        """
        # Check if inputs are lists; if not, wrap them in a list
        if not isinstance(circuit, list):
            circuit = [circuit]
        if not isinstance(bound_parameters, list):
            bound_parameters = [bound_parameters]

        # Check if the number of circuits matches the number of parameter dictionaries
        if len(circuit) != len(bound_parameters):
            raise ValueError("The number of circuits must match the number of dictionaries of bound parameters.")
        
    def _validate_observable_string(self, obs_str: str):
        """
        Validate the input string for the observable.

        :param obs_str: The input string for the observable.
        :raises ValueError: If the input string is not a valid Pauli string.
        """
        if not all(char in ["X", "Y", "Z", "I"] for char in obs_str):
            raise ValueError("The observable string must be a valid Pauli string. Use only 'X', 'Y', 'Z', and 'I' characters.")

        
    def _post_process(self, result, observables, coefficients=None):
        """
        Process the result from the quantum computation to extract and optionally scale the expectation values.

        :param result: The result object returned by the quantum computation.
        :param observables: A list of observables for which the expectation values were measured.
        :param coefficients: Optional list of coefficients to scale the expectation values of the observables.
                            This should be in the same order as the observables list.
        :return: A list of processed expectation values.
        """
        if not isinstance(result, list):
            expectation_values = []

            # If coefficients are provided, they must match the number of observables
            if coefficients and len([observables]) != len([coefficients]):
                raise ValueError("The number of coefficients must match the number of observables.")

            for i, observable in enumerate([observables]):
                # Extract the raw expectation value from the result object
                raw_expectation_value = result.values[i]  # Assuming result.values is a list of expectation values
            
                # Apply the coefficient if provided, else use the raw value
                if coefficients:
                    coefficients = [coefficients] if not isinstance(coefficients, list) else coefficients   
                    processed_value = coefficients[i] * raw_expectation_value
                else:
                    processed_value = raw_expectation_value

                expectation_values.append(processed_value)

            return expectation_values
        
        elif isinstance(result, list) and len(result) > 1:
            expectation_values = []
            for res in result:
                expectation_values.append(self._post_process(res, observables, coefficients))
            return expectation_values
        
    def post_process(self, result, coefficients=None):
        """
        Calculate the weighted sum of the expectation values.
        """
        if coefficients is None:
            coefficients = [1.0] * len(result.values)

        # Calculate the weighted sum of the expectation values
        total_expectation = sum(coeff * result.values[0] for coeff, result in zip(coefficients, result))
        return total_expectation
    
    @staticmethod
    def _commutator(op1, op2):
        return op1@op2 == op2@op1
    
    def _group_observables(self, observables):
        groups = []
        for obs in observables:
            if isinstance(obs, str):
                    self._validate_observable_string(obs)
                    obs = PauliString(obs).to_unsigned_observable(include_trivial=True)
            placed = False
            for group in groups:
                if all(self._commutator(obs, member) for member in group):
                    group.append(obs)
                    placed = True
                    break
            if not placed:
                groups.append([obs])
        return groups


    def run(
        self, 
        circuit: Union[Circuit, list[Circuit]], 
        observables: Union[Observable, str, list[Observable], list[str]] = None,
        op_coefficients: Union[complex, list[complex]] = [1.0],
        target_register: Union[int, list[int]] = [0],
        bound_parameters: Optional[dict] = None,
        shots: int = 0, 
    ):

        self._validate_inputs(circuit, bound_parameters)

        if isinstance(observables, list):
            assert len(observables) == len(op_coefficients), "The number of observables must match the number of coefficients."
            op_coefficients = [1.0] * len(observables) if op_coefficients is None else op_coefficients
            # Group observables to minimize the number of circuits
            observable_groups = self._group_observables(observables=observables)
                        
            hamiltonian_results = []
            for group in observable_groups:
                # Make a copy of the original circuit for each group
                circ = circuit.copy() if isinstance(circuit, Circuit) else [c.copy() for c in circuit]

                for obs in group:
                    # Convert string observables to the Observable type if necessary
                    if not isinstance(obs, str) and not isinstance(obs, Observable) and not isinstance(obs, TensorProduct):
                        raise ValueError("Passed Observables can only be of type str, Observable or TensorProduct.")
                    if isinstance(obs, str):
                        self._validate_observable_string(obs)
                        obs = PauliString(obs).to_unsigned_observable(include_trivial=True)

                    # Use a dummy variable to avoid modifying the original circuit
                    # Add the expectation instruction to the circuit for each observable
                    circ = circuit.copy() if isinstance(circuit, Circuit) else [c.copy() for c in circuit]
                    circ.expectation(observable=obs, target=target_register)

                    if circuit.parameters: # Run with bound parameters if they exist
                        job = self.backend.run(circ, inputs=bound_parameters, shots=shots)
                    else:
                        job = self.backend.run(circ, shots=shots)

                    hamiltonian_results.append(job.result())

            hamiltonian_expval = self.post_process(hamiltonian_results, op_coefficients)
            return hamiltonian_results, np.array(hamiltonian_expval)

        elif isinstance(circuit, list) and len(circuit) > 1: # Batch execution
            if isinstance(observables, str):
                self._validate_observable_string(obs)
                observables = PauliString(observables).to_unsigned_observable(include_trivial=True)
            for circ in circuit:
                circ.expectation(observable=observables, target=target_register)
            
            print('Running batch...')
            job = self.backend.run_batch(circuit, inputs=bound_parameters, shots=shots)
            print('Finished running batch.')
            
            return list(job.results()), self._post_process(list(job.results()), observables, op_coefficients)
        
        else:
            if isinstance(observables, str):
                self._validate_observable_string(obs)
                observables = PauliString(observables).to_unsigned_observable(include_trivial=True)
            circuit.expectation(observable=observables, target=target_register)
            
            if circuit.parameters:    
                job = self.backend.run(circuit, inputs=bound_parameters, shots=shots)
            else:
                job = self.backend.run(circuit, shots=shots)

            return job.result(), self._post_process(job.result(), observables, op_coefficients)