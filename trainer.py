import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn
import time

from model import DaggerLSTM, DaggerNetwork
from env_vars import *
from logger import logger

class Trainer(object):
    def __init__(self, model_path):
        self.state_dim = 4
        self.action_cnt = 5
        # augmented state space: state and previous action (one-hot vector)
        self.aug_state_dim = self.state_dim + self.action_cnt

        self.learn_rate = 0.01
        self.regularization_lambda = 1e-4

        self.default_batch_size = 256
        self.max_eps = 1000
		
        # LSTM model
        if USING_LSTM_MODEL:
            self.model = DaggerLSTM(state_dim=self.aug_state_dim,
                                    action_cnt=self.action_cnt)
            with tf.variable_scope('best_model'):
                self.best_model = DaggerLSTM(state_dim=self.aug_state_dim,
                                             action_cnt=self.action_cnt)
        else:
            self.model = DaggerNetwork(state_dim=self.aug_state_dim,
                                       action_cnt=self.action_cnt)
            with tf.variable_scope('best_model'):
                self.best_model = DaggerNetwork(state_dim=self.aug_state_dim,
                                                action_cnt=self.action_cnt)

        self.save_to_best_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(
            self.best_model.trainable_vars, self.model.trainable_vars)])

        self.load_from_best_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(
            self.model.trainable_vars, self.best_model.trainable_vars)])
            
        # default initial state of LSTM
        self.default_init_state = self.model.zero_init_state(self.default_batch_size)

        # regularization loss
        reg_loss = 0.0
        for var in self.model.trainable_vars:
            reg_loss += tf.nn.l2_loss(var)
        reg_loss *= self.regularization_lambda

        # cross entropy loss relative to correct actions
        if USING_LSTM_MODEL:
        	self.actions = tf.placeholder(tf.int32, [None, None])
        else:
        	self.actions = tf.placeholder(tf.int32, [None])
        	
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.actions,
                logits=self.model.action_scores))

        # total loss = cross entropy loss + reg_lambda * regularization loss
        self.total_loss = cross_entropy_loss + reg_loss

        # op to train / optimize
        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        self.train_op = optimizer.minimize(self.total_loss)

        # Tensorflow session
        self.sess = tf.Session()

        if model_path != '':
		    # restore saved variables
            saver = tf.train.Saver(self.model.trainable_vars)
            saver.restore(self.sess, model_path)

            # init the remaining vars, especially those created by optimizer
            uninit_vars = set(tf.global_variables())
            uninit_vars -= set(self.model.trainable_vars)
            self.sess.run(tf.variables_initializer(uninit_vars))
        else:
            # initial model with random weights
            self.sess.run(tf.global_variables_initializer())
            
    def save_model(self, model_path):
        saver = tf.train.Saver(self.model.trainable_vars)
        saver.save(self.sess, model_path)
        
    def run_one_train_step(self, batch_states, batch_actions):
        ops_to_run = [self.train_op, self.total_loss]
        ret = self.sess.run(ops_to_run, feed_dict={
            self.model.input: batch_states,
            self.actions: batch_actions,
            self.model.state_in: self.init_state}) 
        return ret[1]

	# all_inputs is in the shape of [n, T, input_vec_len] 
	# all_expert_actions is in the shape of [n, T]
	# where n is the # of samples, T is the length of time series in each sample
	# 
    def train(self, all_inputs, all_expert_actions):
        """ Runs the training operator until the loss converges.
        """
        curr_iter = 0

        min_loss = float('inf')
        iters_since_min_loss = 0

        if USING_LSTM_MODEL:
            assert(len(all_expert_actions.shape) == 2)
            assert(len(all_inputs.shape) == 3)
            assert(all_expert_actions.shape[0] == all_inputs.shape[0])
            assert(all_expert_actions.shape[1] == all_inputs.shape[1])
        else:
            assert(len(all_expert_actions.shape) == 1)
            assert(len(all_inputs.shape) == 2)
            assert(all_expert_actions.shape[0] == all_inputs.shape[0])
       
        n = all_inputs.shape[0]
        
        # TODO: the last batch is discarded if n is not a multiple of batch_size..
        batch_size = min(n, self.default_batch_size)
        num_batches = n // batch_size

        if batch_size != self.default_batch_size:
            self.init_state = self.model.zero_init_state(batch_size)
        else:
            self.init_state = self.default_init_state

        self.sess.run(self.save_to_best_op)
        
        start_time = time.time()
        
        while True:
            # shuffle the training data
            permutation = np.random.permutation(n)
            all_inputs = all_inputs[permutation]
            all_expert_actions = all_expert_actions[permutation]
            
            curr_iter += 1

            mean_loss = 0.0
            max_loss = 0.0

            for batch_num in range(0, num_batches):

                start = batch_num * batch_size
                end = start + batch_size

                batch_states = all_inputs[start:end]
                batch_actions = all_expert_actions[start:end]

                loss = self.run_one_train_step(batch_states, batch_actions)

                mean_loss += loss
                max_loss = max(loss, max_loss)

            mean_loss /= num_batches

            logger.info('--- iter %d: max loss %.4f, mean loss %.4f\n' %
                             (curr_iter, max_loss, mean_loss))

            time_elapsed = float(time.time() - start_time)
            
            if max_loss < min_loss - 0.001:
                min_loss = max_loss
                self.sess.run(self.save_to_best_op)
                iters_since_min_loss = 0
            else:
                iters_since_min_loss += 1

            if curr_iter > 1000 and time_elapsed > 600:
                break
            
            if max_loss < 0.005:
                break
                
            if iters_since_min_loss >= max(0.2 * curr_iter, 50) and time_elapsed > 600:
                break
		    
		    if time_elapsed > 900:
		        break
		        
        self.sess.run(self.load_from_best_op)
		
    def test_accuracy(self, all_inputs, all_expert_actions):
        """ Returns the current accuracy on the given inputs
        """
        correct = 0
        total = 0
        
        if USING_LSTM_MODEL:
        
            assert(len(all_expert_actions.shape) == 2)
            assert(len(all_inputs.shape) == 3)
            assert(all_expert_actions.shape[0] == all_inputs.shape[0])
            assert(all_expert_actions.shape[1] == all_inputs.shape[1])
            
            for k in range(0, all_inputs.shape[0]):
                lstm_state = self.model.zero_init_state(1)
                for i in range(0, all_inputs.shape[1]):
                    input_v = np.array([[all_inputs[k][i]]])
                    action_probs, lstm_state = self.sess.run([self.model.action_probs, self.model.state_out], {
                        self.model.input: input_v,
                        self.model.state_in: lstm_state,
                    })
                    action = np.argmax(action_probs[0])
                    if action == all_expert_actions[k][i]:
                        correct += 1
                    total += 1
        else:
        
            assert(len(all_expert_actions.shape) == 1)
            assert(len(all_inputs.shape) == 2)
            assert(all_expert_actions.shape[0] == all_inputs.shape[0])
            
            lstm_state = self.model.zero_init_state(1)
            
            action_probs, lstm_state = self.sess.run([self.model.action_probs, self.model.state_out], {
                self.model.input: all_inputs,
                self.model.state_in: lstm_state,
            })
            
            for i in range(0, all_inputs.shape[0]):
                action = np.argmax(action_probs[i])
                if action == all_expert_actions[i]:
                    correct += 1
                total += 1
        
        return (correct, total)
        
