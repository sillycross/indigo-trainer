import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn

class DaggerNetwork(object):
    def __init__(self, state_dim, action_cnt):
        self.input = tf.placeholder(tf.float32, [None, state_dim], name="inference_inputs")

        actor_h1 = layers.relu(self.input, 8)
        actor_h2 = layers.relu(actor_h1, 8)
        self.action_scores = layers.linear(actor_h2, action_cnt)
        self.action_probs = tf.nn.softmax(self.action_scores,
                                          name='action_probs')
		
        # these doesn't really matter, 
        # just to provide a consistent interface so things are easier
        #
        self.num_layers = 1
        self.lstm_dim = 32
        self.state_in = tf.placeholder(tf.float32, [None, self.lstm_dim * self.num_layers * 2], name="lstm_state_in")
        self.state_out = tf.identity(self.state_in, name="lstm_state_out")
        
        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            
    def zero_init_state(self, batch_size):
        init_state = np.zeros([batch_size, self.lstm_dim * self.num_layers * 2], np.float32)
        return init_state 

class DaggerLSTM(object):
    def __init__(self, state_dim, action_cnt):
        # self.input: [batch_size, max_time, state_dim]
        self.input = tf.placeholder(tf.float32, [None, None, state_dim], name="inference_inputs")

        self.num_layers = 1
        self.lstm_dim = 32
        stacked_lstm = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=False)
            for _ in range(self.num_layers)], state_is_tuple=False)

        self.state_in = tf.placeholder(tf.float32, [None, self.lstm_dim * self.num_layers * 2], name="lstm_state_in")

        # self.output: [batch_size, max_time, lstm_dim]
        output, temp_state_out = tf.nn.dynamic_rnn(
            stacked_lstm, self.input, initial_state=self.state_in)

        self.state_out = tf.identity(temp_state_out, name="lstm_state_out")

        # map output to scores
        self.action_scores = layers.linear(output, action_cnt)
        temp_action_probs = tf.nn.softmax(self.action_scores)
        self.action_probs = tf.identity(temp_action_probs, name="action_probs")
		
        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def zero_init_state(self, batch_size):
        init_state = np.zeros([batch_size, self.lstm_dim * self.num_layers * 2], np.float32)
        return init_state 

