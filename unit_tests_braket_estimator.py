import unittest
import numpy as np

from braket.devices import LocalSimulator
from braket.circuits import Circuit

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from braket.parametric import FreeParameter
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

from braket_estimator import BraketEstimator


class TestBraketEstimator(unittest.TestCase):
    def setUp(self):
        # Initialize BraketEstimator with a mock backend for testing
        self.backend = LocalSimulator()
        self.estimator = BraketEstimator(self.backend)
        self.qiskit_estimator = Estimator() # For comparison

    def test_single_circuit_single_observable(self):
        # Define a single circuit and observable
        circuit = [Circuit().h(0)]
        observable = [('X', 1.0)]
        result = self.estimator.run(circuit, observables=observable, target_register=[0])

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.h(0)
        qiskit_observables = SparsePauliOp.from_list(observable)
        qiskit_job = self.qiskit_estimator.run(qiskit_circuit, observables=qiskit_observables)

        assert np.allclose(result, qiskit_job.result().values, atol=1e-6)

    def test_single_circuit_multiple_observables(self):
        # Define a single circuit and multiple observables with coefficients
        circuit = [Circuit().h(0)]
        observables = [('X', 1.0), ('Y', 0.5)]
        result = self.estimator.run(circuit, observables=observables, target_register=[0])
        
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.h(0)
        qiskit_observables = SparsePauliOp.from_list(observables)
        qiskit_job = self.qiskit_estimator.run(qiskit_circuit, observables=qiskit_observables)

        assert np.allclose(result, qiskit_job.result().values, atol=1e-6)

    def test_multiple_circuits_single_observable(self):
        # Define multiple circuits and a single observable with coefficient
        circuits = [Circuit().h(0), Circuit().x(0)]
        observable = [('Z', 1.0)]
        result = self.estimator.run(circuits, observables=observable, target_register=[[0], [0]])
        
        qiskit_circuit_01 = QuantumCircuit(1)
        qiskit_circuit_01.h(0)
        qiskit_circuit_02 = QuantumCircuit(1)
        qiskit_circuit_02.x(0)
        qiskit_circuits= [qiskit_circuit_01, qiskit_circuit_02]
        qiskit_observables = SparsePauliOp.from_list(observable)
        qiskit_job = self.qiskit_estimator.run(qiskit_circuits, observables=[qiskit_observables]*len(qiskit_circuits))

        assert np.allclose(result, qiskit_job.result().values, atol=1e-6)

    def test_multiple_circuits_multiple_observables(self):
        # Define multiple circuits and multiple observables with coefficients
        circuits = [Circuit().h(0), Circuit().x(0)]
        observables = [[('X', 1.0), ('Y', 0.5)], [('Z', 0.8)]]
        result = self.estimator.run(circuits, observables=observables, target_register=[[0], [0]])
        
        qiskit_circuit_01 = QuantumCircuit(1)
        qiskit_circuit_01.h(0)
        qiskit_circuit_02 = QuantumCircuit(1)
        qiskit_circuit_02.x(0)
        qiskit_circuits= [qiskit_circuit_01, qiskit_circuit_02]
        qiskit_observable_0 = SparsePauliOp.from_list(observables[0])
        qiskit_observable_1 = SparsePauliOp.from_list(observables[1])
        qiskit_observables = [qiskit_observable_0, qiskit_observable_1]
        qiskit_job = self.qiskit_estimator.run(qiskit_circuits, observables=qiskit_observables)

        assert np.allclose(result, qiskit_job.result().values, atol=1e-6)
    
    def test_two_qubit_circuit(self):
        # Define a single circuit and observable
        circuit = [Circuit().h(0).cnot(0, 1)]
        observable = [('XX', 1.0)]
        result = self.estimator.run(circuit, observables=observable, target_register=[0, 1])
        
        qiskit_circuit = QuantumCircuit(2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_observables = SparsePauliOp.from_list(observable)
        qiskit_job = self.qiskit_estimator.run(qiskit_circuit, observables=qiskit_observables)

        assert np.allclose(result, qiskit_job.result().values, atol=1e-6)

    def test_two_qubit_circuit_multiple_observables(self):
        # Define a single circuit and multiple observables with coefficients
        circuit = [Circuit().h(0).cnot(0, 1)]
        observables = [('XX', 1.0), ('YY', 0.5)]
        result = self.estimator.run(circuit, observables=observables, target_register=[0, 1])

        qiskit_circuit = QuantumCircuit(2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_observables = SparsePauliOp.from_list(observables)
        qiskit_job = self.qiskit_estimator.run(qiskit_circuit, observables=qiskit_observables)

        assert np.allclose(result, qiskit_job.result().values, atol=1e-6)

    def test_parametric_circuit(self):
        # Define a parametric circuit and observable
        circuit = [Circuit().rx(0, FreeParameter("alpha"))]
        observable = [('X', 1.0), ('Y', 0.5)]
        result = self.estimator.run(circuit, observables=observable, target_register=[0], bound_parameters={'alpha': 0.5})
        
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.rx(Parameter('alpha'), 0)
        qiskit_observables = SparsePauliOp.from_list(observable)
        qiskit_job = self.qiskit_estimator.run(qiskit_circuit, observables=qiskit_observables, parameter_values=[0.5])

        assert np.allclose(result, qiskit_job.result().values, atol=1e-6)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)