import numpy as np

from env_vars import *

# expert_actions, input_vectors = load_training_data(npz_file_path)
#
def load_training_data(npz_file_path):
    data = np.load(npz_file_path)
    assert('expert_actions' in data.files)
    assert('input_vectors' in data.files)
    expert_actions = data['expert_actions']
    input_vectors = data['input_vectors']
    data.close()
    if (not USING_LSTM_MODEL):
        input_vectors = input_vectors.reshape(-1, input_vectors.shape[-1])
        expert_actions = expert_actions.flatten()
    return (expert_actions, input_vectors)

