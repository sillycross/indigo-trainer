import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn
import argparse
import os

from trainer import Trainer
from env_vars import *

# load the model and save the pb file
#

model_root = PROJECT_ROOT + '/' + MODEL_REPO_NAME

t = Trainer(model_path = model_root + '/model')

output_graph_def = tf.graph_util.convert_variables_to_constants(
  t.sess, 
  tf.get_default_graph().as_graph_def(), 
  ["action_probs", "lstm_state_out"]) 
  
with tf.gfile.GFile(model_root + "/graph.pb", "wb") as f:
    f.write(output_graph_def.SerializeToString())


