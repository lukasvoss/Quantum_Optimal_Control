# %% [markdown]
# # Quantum Gate calibration using Model Free Reinforcement Learning in AWS BraketLocalBackend
# 
# We extend the state preparation scheme to a gate calibration scheme by providing multiple input states to the target.

# %%
from qiskit_braket_provider import BraketLocalBackend, AWSBraketProvider, AWSBraketBackend, AWSBraketJob
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
import boto3
client = boto3.client("braket", region_name='us-west-1')
import braket._sdk as braket_sdk
braket_sdk.__version__

# %%
# response = client.search_devices(filters=[{
#     'name': 'deviceArn',
#     'values': ['arn:aws:braket:::device/quantum-simulator/amazon/sv1']
# }], maxResults=10)
# print(f"Found {len(response['devices'])} devices")

# response = client.get_device(deviceArn='arn:aws:braket:::device/quantum-simulator/amazon/sv1')
# print(f"Device {response['deviceName']} is {response['deviceStatus']}")

# %%
import numpy as np
import os
import sys
from typing import Optional
module_path = os.path.abspath(os.path.join('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control'))
if module_path not in sys.path:
    sys.path.append(module_path)
from quantumenvironment import QuantumEnvironment
from helper_functions import select_optimizer, generate_model
from qconfig import QiskitConfig, SimulationConfig
# Qiskit imports for building RL environment (circuit level)
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2
from qiskit.extensions import CXGate, XGate
from qiskit.opflow import Zero, One, Plus, Minus, H, I, X, CX, S, Z
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator

# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag

# Additional imports
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

# %%
# Ansatz function, could be at pulse level or circuit level
def apply_parametrized_circuit(qc: QuantumCircuit):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
    :param qc: Quantum Circuit instance to add the gates on
    :return:
    """
    # qc.num_qubits
    global n_actions
    params = ParameterVector('theta', n_actions)
    qc.u(2 * np.pi * params[0], 2 * np.pi * params[1], 2 * np.pi * params[2], 0)
    qc.u(2 * np.pi * params[3], 2 * np.pi * params[4], 2 * np.pi * params[5], 1)
    qc.rzx(2 * np.pi * params[6], 0, 1)

# %% [markdown]
# # Defining the QuantumEnvironment
# 
# Below, we set the RL environment parameters, that is how we describe our quantum system. Below, we can choose to go through the use of Qiskit Runtime, or to speed things up by using the local CPU and a state-vector simulator to get measurement outcomes based on the ansatz circuit defined above. The Environment is defined as a class object called QuantumEnvironment.

# %% [markdown]
# ## Generic information characterizing the quantum system
# 
# The algorithm is built upon Qiskit modules. To specify how to address our quantum system of interest, we therefore adopt the IBM approach to define a quantum backend, on which qubits are defined and can be accessed via control actions and measurements.
# 
# The cell below specifies:
# - ```qubit_tgt_register```: List of qubit indices which are specifically addressed by controls , namely the ones for which we intend to calibrate a gate upon or steer them in a specific quantum state. Note that this list could include less qubits than the total number of qubits, which can be useful when one wants to take into account crosstalk effects emerging from nearest-neigbor coupling.
# - ```sampling_Paulis```: number of Pauli observables  to be sampled from the system: the algorithm relies on the ability to process measurement outcomes to estimate the expectation value of different Pauli operators. The more observables we provide for sampling, the more properties we are able to deduce with accuracy about the actual state that was created when applying our custom controls. For a single qubit, the possible Pauli operators are $\sigma_0=I$, $\sigma_x=X$, $\sigma_y=Y$, $\sigma_z=Z$. For a general multiqubit system, the Pauli observables are tensor products of those single qubit Pauli operators. The algorithm will automatically estimate which observables are the most relevant to sample based on the provided target. The probability distribution from which those observables are sampled is derived from the Direct Fidelity Estimation (equation 3, https://link.aps.org/doi/10.1103/PhysRevLett.106.230501) algorithm. 
# - ```N_shots```: Indicates how many measurements shall be done for each provided circuit (that is a specific combination of an action vector and a Pauli observable to be sampled)
# - The dimension of the action vector: Indicates the number of pulse/circuit parameters that characterize our parametrized quantum circuit.
# - ```estimator_options```: Options of the Qiskit Estimator primitive. The Estimator is the Qiskit module enabling an easy computation of Pauli expectation values. One can set options to make this process more reliable (typically by doing some error mitigation techniques in post-processing). Works only with Runtime Backend at the moment
# - ```abstraction_level``` chosen to encode our quantum circuit. One can choose here to stick to the usual circuit model of quantum computing, by using the ```QuantumCircuit``` objects from Qiskit and therefore set the ```abstraction_level``` to ```"circuit"```. However, depending on the task at hand, one can also prefer to use a pulse description of all the operations in our circuit. This is possible by using resources of another module of Qiskit called Qiskit Dynamics. In this case, one should define the ansatz circuit above in a pulse level fashion, and the simulation done at the Hamiltonian level, and not only via statevector calculations. In this notebook we set the ```abstraction_level``` to ```"circuit"```. Another notebook at the pulse level is available in the repo.
if __name__ == '__main__':

    # %%
    qubit_tgt_register = [0, 1]  # Choose which qubits of the QPU you want to address 
    sampling_Paulis = 100
    N_shots = 1  # Number of shots for sampling the quantum computer for each action vector
    n_actions = 7  # Choose how many control parameters in pulse/circuit parametrization
    seed = 4000
    estimator_options = {'seed_simulator': seed, 'resilience_level': 0}


    # %% [markdown]
    # Choose below which IBM Backend to use. As we are dealing with circuit level implementation, we can look for a backend supporting Qiskit Runtime (could be a cloud simulator, or real backend) or simply set backend to None and rely on the Estimator primitive based on statevector simulation. In either case, we need access to one Estimator primitive to run the algorithm, as the feedback from the measurement outcomes is done by calculating Pauli expectation values.

    # %% [markdown]
    # ## 1. Setting up a Quantum Backend

    # %% [markdown]
    # ### Real backend initialization
    # 
    # Uncomment the cell below to declare a Qiskit Runtime backend. You need an internet connection and an IBM Id account to access this.

    # %%
    """
    Real backend initialization:
    Run this cell only if intending to use a real backend,
    where Qiskit Runtime is enabled
    """
    backend_name = 'ibm_perth'

    #service = QiskitRuntimeService(channel='ibm_quantum')
    #runtime_backend = service.get_backend(backend_name)
    #estimator_options = {'resilience_level': 0}


    # %% [markdown]
    # ### Simulation backend initialization
    # If you want to run the algorithm over a simulation, you can use Qiskit BaseEstimator, which does not need any real backend and relies on statevector simulation.
    # 
    # Note that you could also define a custom Aer noise model and use an Aer version of the Estimator primitive. This feature will become available soon.
    # 

    # %%
    """
    If using Qiskit native Estimator primitive
    (statevector simulation)
    """
    no_backend = None
    backend = no_backend

    # %% [markdown]
    # ### Choose backend and define Qiskit config dictionary
    # Below, set the Backend that you would like to run among the above defined backend.
    # Then define the config gathering all the components enabling the definition of the ```QuantumEnvironment```.
    # 
    # 

    # %%
    provider = AWSBraketProvider()
    # TODO: Task Batching currently does not work for SV1 (despite AWS Braket documentation mentining it as a use case)
    # So it's probably an issue in handling the details in quantuumenvironment.py
    backend = provider.get_backend('SV1')

    # backend = BraketLocalBackend()    
    # backend = LocalSimulator()

    # %%
    type(backend)

    # %%
    from qiskit.providers import BackendV2

    type(backend)
    isinstance(backend, (AWSBraketBackend, LocalSimulator, BraketLocalBackend))

    # %%
    backend

    # %%
    # backend = FakeJakartaV2()

    # %% [markdown]
    # ## 2. Define quantum target: State preparation or Gate calibration
    # 
    # The target of our optimal control task can be of two different types:
    # 1.  An arbitrary quantum state to prepare with high accuracy
    # 2. A Quantum Gate to be calibrated in a noise-robust manner
    # 
    # Both targets are dictionaries that are identified with a key stating their ```target_type```, which can be either ```"state"``` or ```"gate"```.
    # 
    # For a gate target $G$, one can add the target quantum gate with a ```"gate"``` argument specifying a specific instance of a Qiskit ```Gate``` object. Here, we settle for calibrating a ```CXGate()```.
    # Moreover, a gate calibration requires a set of input states $\{|s_i\rangle \}$ to be provided, such that the agent can try to set the actions such that the fidelity between the anticipated ideal target state (calculated as  $G|s_i\rangle$) and the output state are simultaneously maximized. To ensure a correlation between the average reward computed from the measurement outcomes and the average gate fidelity, the provided set of input states must be tomographically complete.
    # 
    # For a state target, one can provide, similarly to an input state, an ideal circuit to prepare it (```"circuit": QuantumCircuit```, or a density matrix (key ```"dm": DensityMatrix```).
    # 
    # Another important key that should figure in the dictionary is the ```"register"``` indicating the qubits indices that should be addressed by this target, i.e. upon which qubits should the target be engineered.
    # 

    # %%
    # Target gate: CNOT gate

    circuit_Plus_i = S @ H
    circuit_Minus_i = S @ H @ X
    cnot_target = {
        "abstraction_level": "circuit",
        'gate_string': 'cx',
        "gate": CXGate("CNOT"),
        "register": qubit_tgt_register
    }

    target = cnot_target

    # %%
    target = cnot_target

    # %% [markdown]
    # ## 3. Declare QuantumEnvironment object
    # Running the box below declares the QuantumEnvironment instance.
    # 

    # %%
    # Create a configutration object for the simulation
    sim_config = SimulationConfig(
        parametrized_circuit=apply_parametrized_circuit,
        target=target,
        backend=backend,
        n_actions=n_actions,
        sampling_Paulis=sampling_Paulis,
        n_shots=N_shots,
        c_factor=0.25,
        device=None,
    )

    q_env = QuantumEnvironment(simulation_config=sim_config)

    # %% [markdown]
    # # Defining the RL agent: PPO

    # %%
    """
    -----------------------------------------------------------------------------------------------------
    Hyperparameters for RL agent
    -----------------------------------------------------------------------------------------------------
    """

    n_epochs = 1000  # Number of epochs
    batchsize = 300  # Batch size (iterate over a bunch of actions per policy to estimate expected return)
    opti = "Adam"  # Optimizer choice
    eta = 0.0018  # Learning rate for policy update step
    eta_2 = None  # Learning rate for critic (value function) update step

    use_PPO = True
    epsilon = 0.1  # Parameter for clipping value (PPO)
    grad_clip = 0.02
    critic_loss_coeff = 0.5
    optimizer = select_optimizer(lr=eta, optimizer=opti, grad_clip=grad_clip, concurrent_optimization=True, lr2=eta_2)
    sigma_eps = 1e-3  # for numerical stability

    # %%
    """
    -----------------------------------------------------------------------------------------------------
    Policy parameters
    -----------------------------------------------------------------------------------------------------
    """
    n_qubits = 2  # Number of qubits in the system
    N_in = n_qubits + 1  # One input for each measured qubit state (0 or 1 input for each neuron)
    hidden_units = [20, 20, 30]  # List containing number of units in each hidden layer

    network = generate_model((N_in,), hidden_units, n_actions, actor_critic_together=True)
    network.summary()
    init_msmt = np.zeros((1, N_in))  # Here no feedback involved, so measurement sequence is always the same

    # %%
    # Plotting tools
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    avg_return = np.zeros(n_epochs)
    fidelities = np.zeros(n_epochs)
    visualization_steps = 20

    # %% [markdown]
    # ## Run algorithm

    # %%
    """
    -----------------------------------------------------------------------------------------------------
    Training loop
    -----------------------------------------------------------------------------------------------------
    """
    # TODO: Use TF-Agents PPO Agent
    mu_old = tf.Variable(initial_value=network(init_msmt)[0][0], trainable=False)
    sigma_old = tf.Variable(initial_value=network(init_msmt)[1][0], trainable=False)

    for i in tqdm(range(n_epochs)):

        Old_distrib = MultivariateNormalDiag(loc=mu_old, scale_diag=sigma_old,
                                            validate_args=True, allow_nan_stats=False)

        with tf.GradientTape(persistent=True) as tape:

            mu, sigma, b = network(init_msmt, training=True)
            mu = tf.squeeze(mu, axis=0)
            sigma = tf.squeeze(sigma, axis=0)
            b = tf.squeeze(b, axis=0)

            Policy_distrib = MultivariateNormalDiag(loc=mu, scale_diag=sigma,
                                                    validate_args=True, allow_nan_stats=False)

            action_vector = tf.stop_gradient(tf.clip_by_value(Policy_distrib.sample(batchsize), -1., 1.))

            reward = q_env.perform_action(action_vector)
            advantage = reward - b
            if use_PPO:
                ratio = Policy_distrib.prob(action_vector) / (tf.stop_gradient(Old_distrib.prob(action_vector)) + 1e-6)
                actor_loss = - tf.reduce_mean(tf.minimum(advantage * ratio,
                                                        advantage * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)))
            else:  # REINFORCE algorithm
                actor_loss = - tf.reduce_mean(advantage * Policy_distrib.log_prob(action_vector))

            critic_loss = tf.reduce_mean(advantage ** 2)
            combined_loss = actor_loss + critic_loss_coeff * critic_loss

        grads = tape.gradient(combined_loss, network.trainable_variables)

        # For PPO, update old parameters to have access to "old" policy
        if use_PPO:
            mu_old.assign(mu)
            sigma_old.assign(sigma)

        avg_return[i] = np.mean(q_env.reward_history, axis =1)[i]
        fidelities[i] = q_env.avg_fidelity_history[i]
        print("Gate Fidelity", fidelities[i])
        if i%visualization_steps == 0:
            clear_output(wait=True) # for animation
            fig, ax = plt.subplots()
            ax.plot(np.arange(1, n_epochs, 20),avg_return[0:-1:visualization_steps], '-.', label='Average return')
            ax.plot(np.arange(1, n_epochs, 20),fidelities[0:-1:visualization_steps], label='Average Gate Fidelity')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("State Fidelity")
            ax.legend()
            plt.show()
            print("Maximum fidelity reached so far:", np.max(fidelities), "at Epoch", np.argmax(fidelities))

        # Apply gradients
        optimizer.apply_gradients(zip(grads, network.trainable_variables))
    if isinstance(q_env.estimator, Estimator):
        q_env.estimator.session.close()