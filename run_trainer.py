import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn
import argparse
import os
import json

from trainer import Trainer
from env_vars import *
from load_training_data import *
from logger import logger

model_root = PROJECT_ROOT + '/' + MODEL_REPO_NAME

os.chdir(PROJECT_ROOT)

with open('%s/version' % MODEL_REPO_NAME) as f:
    values = f.read().splitlines()
    assert(len(values) == 1)
    VERSION = int(values[0])
    assert(VERSION >= 0)

with open('workloads/config.json') as json_file:
    tasks_json = json.load(json_file) 

# read all training data points
#
num_collected = 0
expert_actions_list = []
input_vectors_list = []
for it in range(0, VERSION + 1):
    assert(os.path.isdir("training_data/%d" % it))
    for j in range(0, len(tasks_json)):
        for k in range(0, tasks_json[j]['repeats']):
            fpath = "training_data/%d/%d_%d.npz" % (it, j, k)
            if os.path.exists(fpath):
                expert_actions, input_vectors = load_training_data(fpath)
                if expert_actions is not None:
                    if it == VERSION:
                        num_collected += expert_actions.shape[0]
                    expert_actions_list.append(expert_actions)
                    input_vectors_list.append(input_vectors)
            else:
                assert(False)

logger.info('Finished reading training data')

assert(len(expert_actions_list) > 0)
aggregated_expert_actions = np.concatenate(expert_actions_list)
aggregated_input_vectors = np.concatenate(input_vectors_list)

if not USING_LSTM_MODEL:
    assert(num_collected % EPISODE_LEN == 0)
    num_collected = num_collected // EPISODE_LEN
    
expect_collected = 0
for task in tasks_json:
    expect_collected += task['repeats'] * task['num-clients']

logger.info('Expecting %d data points in last iteration, actually got %d data points.' % (expect_collected, num_collected))

t = Trainer(model_path = model_root + '/model')

correct, total = t.test_accuracy(aggregated_input_vectors, aggregated_expert_actions)
logger.info('Before training: %d / %d correct' % (correct, total))

t.train(aggregated_input_vectors, aggregated_expert_actions)

correct, total = t.test_accuracy(aggregated_input_vectors, aggregated_expert_actions)
logger.info('After training: %d / %d correct' % (correct, total))

t.save_model(model_path = model_root + '/model')

logger.info('Updated model has been saved.')

