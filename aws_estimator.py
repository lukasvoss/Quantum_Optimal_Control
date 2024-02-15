class AWSEstimator:
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

        
    def post_process(self, result, observables, coefficients=None):
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
                    processed_value = coefficients[i] * raw_expectation_value
                else:
                    processed_value = raw_expectation_value

                expectation_values.append(processed_value)

            return expectation_values
        
        elif isinstance(result, list) and len(result) > 1:
            expectation_values = []
            for res in result:
                expectation_values.append(self.post_process(res, observables, coefficients))
            return expectation_values

    def run(
        self, 
        circuit, 
        observables,
        op_coefficients: complex | list[complex] = [1.0],
        target_register: int | list[int] = [0],
        bound_parameters: dict = None,
        shots: int = 1000, 
        ):

        self._validate_inputs(circuit, bound_parameters)

        if isinstance(circuit, list) and len(circuit) > 1: # Batch execution
            for circ in circuit:
                circ.expectation(observable=observables, target=target_register)
            
            print('Running batch...')
            job = self.backend.run_batch(circuit, inputs=bound_parameters, shots=shots)
            print('Finished running batch.')
            
            return list(job.results()), self.post_process(list(job.results()), observables, op_coefficients)
        
        else:
            circuit.expectation(observable=observables, target=target_register)
            
            if circuit.parameters:    
                job = self.backend.run(circuit, inputs=bound_parameters, shots=shots)
            else:
                job = self.backend.run(circuit, shots=shots)

            return job.result(), self.post_process(job.result(), observables, op_coefficients)
