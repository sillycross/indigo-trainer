import numpy as np

from env_vars import *

# expert_actions, input_vectors = load_training_data(npz_file_path)
#
def load_training_data(npz_file_path):
    data = np.load(npz_file_path)
    if 'empty' in data.files:
        data.close()
        return (None, None)
        
    assert('expert_actions' in data.files)
    assert('input_vectors' in data.files)
    expert_actions = data['expert_actions']
    input_vectors = data['input_vectors']
    data.close()
    assert(len(input_vectors.shape) == 3)
    assert(len(expert_actions.shape) == 2)
    assert(input_vectors.shape[0] == expert_actions.shape[0])
    assert(input_vectors.shape[1] == EPISODE_LEN)
    assert(expert_actions.shape[1] == EPISODE_LEN)
    if (not USING_LSTM_MODEL):
        input_vectors = input_vectors.reshape(-1, input_vectors.shape[-1])
        expert_actions = expert_actions.flatten()
    return (expert_actions, input_vectors)

