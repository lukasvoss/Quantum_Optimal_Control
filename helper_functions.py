import tensorflow as tf
from typing import Optional, Tuple, List, Union, Dict
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import Model

from qiskit.circuit import ParameterVector, QuantumRegister
from qiskit import pulse
from qiskit.pulse import ScheduleBlock
from qiskit_ibm_provider import IBMBackend
from qiskit_dynamics import DynamicsBackend


def constrain_mean_value(mu_var):
    return [tf.clip_by_value(m, -1., 1.) for m in mu_var]


def constrain_std_value(std_var):
    return [tf.clip_by_value(std, 1e-3, 3) for std in std_var]


def custom_pulse_schedule(backend: Union[IBMBackend, DynamicsBackend], target: Dict,
                          qubit_tgt_register: Union[List[int],
                          QuantumRegister], params: ParameterVector,
                          default_schedule: Optional[ScheduleBlock]=None):
    """
    Define parametrization of the pulse schedule characterizing the target gate
    :param qubit_tgt_register: Qubit register on which
    :param target: Dictionary containing information about the target (gate or state)
    :param backend: IBM Backend on which schedule shall be added
    :param params: Parameters of the Schedule
    :param default_schedule: Schedule baseline from which one can customize the pulse parameters
    (QOC)
    :return: Parametrized Schedule
    """

    if default_schedule is None:  # No baseline pulse, full waveform builder
        pass
    else:

        # Look here for the pulse features to specifically optimize upon, for the x gate here, simply retrieve relevant
        # parameters for the Drag pulse
        pulse_ref = default_schedule.instructions[0][1].pulse

        with pulse.build(backend=backend, name='parametrized_schedule') as parametrized_schedule:

            pulse.play(pulse.Drag(duration=pulse_ref.duration, amp=params[0], sigma=pulse_ref.sigma,
                                  beta=pulse_ref.beta, angle=pulse_ref.angle),
                       channel=pulse.DriveChannel(qubit_tgt_register[0]))

            # if dynamics_backend:  # Has to go in class, and add expectation value computation
            #     pulse.acquire(duration=1, qubit_or_channel=pulse.AcquireChannel(qubit_tgt_register[0]),
            #                   register=pulse.MemorySlot(qubit_tgt_register[0]))
        return parametrized_schedule


def select_optimizer(lr: float, optimizer: str = "Adam", grad_clip: Optional[float] = None,
                     concurrent_optimization: bool = True, lr2: Optional[float] = None):
    if concurrent_optimization:
        if optimizer == 'Adam':
            return tf.optimizers.Adam(learning_rate=lr, clipvalue=grad_clip)
        elif optimizer == 'SGD':
            return tf.optimizers.SGD(learning_rate=lr, clipvalue=grad_clip)
    else:
        if optimizer == 'Adam':
            return tf.optimizers.Adam(learning_rate=lr), tf.optimizers.Adam(learning_rate=lr2, clipvalue=grad_clip)
        elif optimizer == 'SGD':
            return tf.optimizers.SGD(learning_rate=lr), tf.optimizers.SGD(learning_rate=lr2, clipvalue=grad_clip)


def generate_model(input_shape: Tuple, hidden_units: Union[List, Tuple], n_actions: int,
                   actor_critic_together: bool = True,
                   hidden_units_critic: Optional[Union[List, Tuple]] = None):
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
    Net = Dense(hidden_units[0], activation='relu', input_shape=input_shape,
                kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{0}")(input_layer)
    for i in range(1, len(hidden_units)):
        Net = Dense(hidden_units[i], activation='relu', kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                    bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{i}")(Net)

    mean_param = Dense(n_actions, activation='tanh', name='mean_vec')(Net)  # Mean vector output
    sigma_param = Dense(n_actions, activation="softplus", name="sigma_vec")(Net)  # Diagonal elements of cov matrix
    # output

    if actor_critic_together:
        critic_output = Dense(1, activation='linear', name="critic_output")(Net)
        return Model(inputs=input_layer, outputs=[mean_param, sigma_param, critic_output])
    else:
        assert hidden_units_critic is not None, "Network structure for critic network not provided"
        input_critic = Input(shape=input_shape)
        Critic_Net = Dense(hidden_units_critic[0], activation='relu', input_shape=input_shape,
                           kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{0}")(input_critic)
        for i in range(1, len(hidden_units)):
            Critic_Net = Dense(hidden_units[i], activation='relu',
                               kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                               bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{i}")(
                Critic_Net)
            critic_output = Dense(1, activation='linear', name="critic_output")(Critic_Net)
            return Model(inputs=input_layer, outputs=[mean_param, sigma_param]), \
                Model(inputs=input_critic, outputs=critic_output)
