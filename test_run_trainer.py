import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn
import argparse
import os

from trainer import Trainer
from env_vars import *
from load_training_data import *

model_root = PROJECT_ROOT + '/' + MODEL_REPO_NAME

t = Trainer(model_path = '')
expert_actions, input_vectors = load_training_data('out.npz')

correct, total = t.test_accuracy(input_vectors, expert_actions)
print('Before training: %d / %d correct' % (correct, total))

t.train(input_vectors, expert_actions)

correct, total = t.test_accuracy(input_vectors, expert_actions)
print('After training: %d / %d correct' % (correct, total))

