"""
Helper file that allows a user to specify the parameters for the gate calibration simulation.

Author: Lukas Voss
Created on 04/11/2023
"""
from qiskit import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Options
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import XGate, SXGate, YGate, ZGate, HGate, CXGate, SGate, ECRGate
from qiskit.providers.fake_provider import FakeJakarta, FakeMelbourne, FakeRome, FakeSydney, FakeValencia, FakeVigo

from qconfig import SimulationConfig

"""
-----------------------------------------------------------------------------------------------------
    User Input: Simulation parameters
-----------------------------------------------------------------------------------------------------
"""
def get_backend():
    ### 
    # TODO: Set Backend
    ###
    # backend = FakeJakarta()
    backend = None

    ### IBM Runtime
    # QiskitRuntimeService.save_account(channel='ibm_quantum', token='MY_IBM_QUANTUM_TOKEN') # Save an IBM Quantum account
    # QiskitRuntimeService.save_account(channel='ibm_cloud', token='MY_IBM_CLOUD_API_KEY', instance='MY_IBM_CLOUD_CRN') # Save an IBM Cloud account
    # service = QiskitRuntimeService()

    # get a real backend from a real provider
    # provider = IBMQ.load_account()
    # backend = provider.get_backend('ibmq_manila')
    # generate a simulator that mimics the real quantum system with the latest calibration results
    # backend_sim = AerSimulator.from_backend(backend)

    return backend

########################################

def set_target():
    ### 
    # TODO: Set Target Gate and Register
    ###
    abstraction_level = 'circuit'
    gate_str = 'x'
    target_gate = CXGate()
    register = [0, 1]

    target = {
        'abstraction_level': abstraction_level,
        'gate_str': gate_str,
        'gate': target_gate, 
        'register': register
    }

    return target

########################################

def get_sim_details():
    ### 
    # TODO: Set Simulation Details
    ###
    n_actions = 7
    sampling_Paulis = 100
    n_shots = 1
    c_factor = 0.25
    device = None

    return {
        'n_actions': n_actions,
        'sampling_Paulis': sampling_Paulis,
        'n_shots': n_shots,
        'c_factor': c_factor,
        'device': device
    }

########################################

# Create a configutration object for the simulation
sim_config = SimulationConfig(
    target=set_target(),
    backend=get_backend(),
    n_actions=get_sim_details()['n_actions'],
    sampling_Paulis=get_sim_details()['sampling_Paulis'],
    n_shots=get_sim_details()['n_shots'],
    c_factor=get_sim_details()['c_factor'],
    device=get_sim_details()['device'],
)