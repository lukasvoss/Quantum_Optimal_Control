# %%
from qiskit_braket_provider import AWSBraketProvider
from braket.aws import AwsSession

aws_session = AwsSession(default_bucket="amazon-braket-us-west-1-lukasvoss")

# %%
import numpy as np
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
module_path = os.path.abspath(os.path.join('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control'))
if module_path not in sys.path:
    sys.path.append(module_path)

from template_configurations import gate_q_env_config
from quantumenvironment import QuantumEnvironment
from gymnasium.wrappers import RescaleAction, ClipAction

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s", # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

# %%
# Define the original action space
print('Initial lower bounds:', gate_q_env_config.action_space.low)
print('Initial upper bounds:', gate_q_env_config.action_space.high)

q_env = QuantumEnvironment(gate_q_env_config)

# Apply the RescaleAction wrapper
q_env = ClipAction(q_env)
q_env = RescaleAction(q_env, min_action=-1.0, max_action=1.0)

# Confirm the rescale box dimensions
print('Rescaled lower bounds:', q_env.action_space.low)
print('Rescaled upper bounds:', q_env.action_space.high)

# %%
print(q_env.backend)

# %%
from helper_functions import load_agent_from_yaml_file
agent_config  = load_agent_from_yaml_file(file_path='/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/agent_config.yaml')

# %%
from ppo import make_train_ppo

ppo_agent = make_train_ppo(agent_config, q_env)

# %%
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## ToDo:
# 
# - Use debugger to walk through the ``perform_action()`` method of the QuantumEnvironment and transform the relevant input metrics to the shape that the ``BraketEstimator`` expects

# %%
training_results = ppo_agent(total_updates=3, print_debug=True, num_prints=40, max_cost=10)


