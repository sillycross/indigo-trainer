import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn
import argparse
import os

from trainer import Trainer
from env_vars import *

model_root = PROJECT_ROOT + '/' + MODEL_REPO_NAME

os.chdir(PROJECT_ROOT)

t = Trainer(model_path = model_root + '/model')

param = np.zeros([t.aug_state_dim], np.float32)
for i in range(0, t.aug_state_dim):
    param[i] = 0.05 * i + 0.01

lstm_state = t.model.zero_init_state(1)
if USING_LSTM_MODEL:
    input_v = np.array([[param]])
    action_probs, lstm_state = t.sess.run([t.model.action_probs, t.model.state_out], {
        t.model.input: input_v,
        t.model.state_in: lstm_state,
    })
    action_probs = action_probs[0]
else:
    input_v = np.array([param])
    action_probs, lstm_state = t.sess.run([t.model.action_probs, t.model.state_out], {
        t.model.input: input_v,
        t.model.state_in: lstm_state,
    })

assert(len(action_probs.shape) == 2)
assert(action_probs.shape[0] == 1)
action_probs = action_probs[0]

exit_code = os.system('%s/sanity_checker > expected_output.txt' % LKM_REPO_NAME)
assert(exit_code == 0)

with open('expected_output.txt') as f:
    all_lines = f.read().splitlines()
    assert(len(all_lines) == 1)
    all_values = all_lines[0].split()
    assert(len(all_values) == t.action_cnt)

for i in range(0, t.action_cnt):
    all_values[i] = float(all_values[i])
    
print('NN sanity checker: Expecting following output')
print(all_values)
print('NN sanity checker: Got following:')
print(action_probs)

assert(len(all_values) == action_probs.shape[0])
for i in range(0, len(all_values)):
    if abs(all_values[i] - action_probs[i]) > 1e-5 * min(abs(all_values[i]), abs(action_probs[i])):
        assert(False)
        

