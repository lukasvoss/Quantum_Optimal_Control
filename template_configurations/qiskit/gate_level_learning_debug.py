# %%
from qiskit_braket_provider import AWSBraketProvider
from braket.aws import AwsSession

aws_session = AwsSession(default_bucket="amazon-braket-us-west-1-lukasvoss")
AWSBraketProvider().backends()

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
print('Initial loower bounds:', gate_q_env_config.action_space.low)
print('Initial upper bounds:', gate_q_env_config.action_space.high)

q_env = QuantumEnvironment(gate_q_env_config)

# Apply the RescaleAction wrapper
q_env = ClipAction(q_env)
q_env = RescaleAction(q_env, min_action=-1.0, max_action=1.0)

# Confirm the rescale box dimensions
print('Rescaled lower bounds:', q_env.action_space.low)
print('Rescaled upper bounds:', q_env.action_space.high)

# %%
gate_name = q_env.target['gate'].name
gate_name

# %%
print(q_env.backend)

# %%
from helper_functions import load_agent_from_yaml_file
ppo_params, network_params  = load_agent_from_yaml_file(file_path='/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/agent_config.yaml')
agent_config = {**ppo_params, **network_params}

# %%
agent_config

# %%
from ppo import make_train_ppo

ppo_agent = make_train_ppo(agent_config, q_env)

# %%
num_updates = 400

# %%
training_results = ppo_agent(total_updates=num_updates, print_debug=True, num_prints=40)

# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

plt.style.use('ggplot')

# %%
avg_reward = training_results['avg_reward']
std_actions = training_results['std_actions']
fidelities = training_results['fidelities']

# %%
job_name = f'{gate_name}-gate-calibration-{int(time.time())}-max-fidelity-{round(max(fidelities), 5)}.pickle'

# %%
job_name = f'{gate_name}-gate-calibration-{int(time.time())}-max-fidelity-{max(fidelities):4%}.pickle'

with open(job_name, 'wb') as handle:
    pickle.dump(training_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
print(f'Final Gate Fidelity: {fidelities[-1]:.4%}')
print(f'\nMax Gate Fidelity: {max(fidelities):.4%}')

# %%
reward_history = np.array(q_env.reward_history)
mean_rewards = np.mean(reward_history, axis=-1)
max_mean = int(np.max(mean_rewards) * 1e4) / 1e4

fidelity = np.array(q_env.avg_fidelity_history)
mean_fidelity = np.mean(fidelity, axis=-1)
max_fidelity = int(np.max(mean_rewards) * 1e4) / 1e4

plt.plot(mean_rewards, label=f'Mean Batch Rewards, max: {max_mean}')
plt.plot(q_env.avg_fidelity_history, label=f'Mean Batch Gate Fidelity, max: {max_fidelity}')
plt.xlabel('Updates')
plt.title('CX Learning Curve')
plt.legend()
plt.show()

# %%
std_actions_componentwise = list(zip(*std_actions))

for ind, param in enumerate(std_actions_componentwise):
    plt.plot(np.arange(1, num_updates+1), param, label=r'std $\theta_{}$'.format(ind+1))

plt.xlabel('update step')
plt.legend()

# %%
print('Final action vector:\n', training_results['action_vector'])


