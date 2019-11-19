import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn
import argparse
import os

from trainer import Trainer
from env_vars import *

# save an initial model with random weights for later training
#

model_root = PROJECT_ROOT + '/' + MODEL_REPO_NAME

t = Trainer(model_path = '') 
t.save_model(model_path = model_root + '/model')

