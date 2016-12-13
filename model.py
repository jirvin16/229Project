from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn
import numpy as np

from data import data_iterator
import datetime
import os
import time
import sys
import h5py
from pprint import pprint

def leaky_relu(x, alpha=0.1):
	return tf.maximum(alpha*x, x)


class AttentionNN(object):
	def __init__(self, config, sess):

		self.sess = sess

		# Training details
		self.batch_size            = config.batch_size
		# self.max_size              = 10
		self.epochs                = config.epochs
		self.current_learning_rate = config.init_learning_rate
		self.grad_max_norm 		   = config.grad_max_norm
		self.MLP 				   = config.network_type == 2
		self.use_lstm			   = config.network_type == 0
		self.use_attention 		   = config.use_attention and self.use_lstm
		self.dropout 			   = config.dropout
		self.random_seed  		   = config.random_seed
		self.optimizer 		   	   = config.optimizer
		self.bidirectional 		   = config.bidirectional
		self.hidden_nonlinearity   = config.hidden_nonlinearity.lower()
		assert self.hidden_nonlinearity in ["leaky_relu", "relu", "sigmoid", "tanh", "elu", "none"]
		self.optim 				   = None
		self.loss                  = None

		self.data_directory 	   = "/deep/group/dlbootcamp/jirvin16/229/data_vectors/"
		# self.data_directory 	   = "/deep/group/dlbootcamp/jirvin16/229/unique_data_vectors/"
		self.is_test 			   = config.mode == 1
		self.validate 			   = config.validate
		
		self.save_every 		   = config.save_every
		self.model_name 		   = config.model_name
		self.model_directory 	   = self.model_name
		self.checkpoint_directory  = os.path.join(self.model_directory, "checkpoints")
		
		# Dimensions and initialization parameters
		self.init_std 			   = 0.1
		self.hidden_dim 	       = config.hidden_dim
		self.embedding_dim         = 100		
		self.num_layers 	       = config.num_layers
		self.num_genres			   = 10

		if self.is_test:
			self.dropout = 0

		if not os.path.isdir(self.data_directory):
			raise Exception(" [!] Data directory %s not found" % self.data_directory)

		if not os.path.isdir(self.model_directory):
			os.makedirs(self.model_directory)

		if not os.path.isdir(self.checkpoint_directory):
			if self.is_test:
				raise Exception(" [!] Checkpoints directory %s not found" % self.checkpoint_directory)
			else:
				os.makedirs(self.checkpoint_directory)

		if self.is_test:
			self.outfile = os.path.join(self.model_directory, "test.out")
		else:
			self.outfile = os.path.join(self.model_directory, "train.out")

		with open(self.outfile, 'w') as outfile:
			pprint(config.__dict__['__flags'], stream=outfile)
			outfile.flush()

		with h5py.File(os.path.join(self.data_directory, 'data.h5')) as hf:
			X    		 	   = np.transpose(hf["X"][:], [0, 2, 1])
			genres 	 		   = hf["y"][:]
			self.genre_names   = hf["genres"][:]
			self.mean   	   = hf["mean"]
			self.std    	   = hf["std"]

		if self.MLP:
			X 				   = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
			self.embedding_dim = X.shape[1]
		else:
			self.max_size      = X.shape[1]
			self.embedding_dim = X.shape[2]

		np.random.seed(self.random_seed)
		train_size 	      = int(0.7 * X.shape[0])
		validation_size   = int(0.1 * X.shape[0])
		permutation 	  = np.random.permutation(X.shape[0])
		shuffled_X 	      = X[permutation]
		shuffled_genres   = genres[permutation]
		self.X_train 	  = shuffled_X[:train_size]
		self.y_train 	  = shuffled_genres[:train_size]
		if self.validate:
			self.X_test   = shuffled_X[train_size:train_size+validation_size]
			self.y_test   = shuffled_genres[train_size:train_size+validation_size]
		else:
			self.X_test   = shuffled_X[train_size+validation_size:]
			self.y_test	  = shuffled_genres[train_size+validation_size:]

		# Model placeholders
		if self.MLP:
			self.audio_batch   = tf.placeholder(tf.float32, shape=[None, self.embedding_dim], name="audio_batch")
		else:
			self.audio_batch   = tf.placeholder(tf.float32, shape=[None, self.max_size, self.embedding_dim], name="audio_batch")
		self.genres 		   = tf.placeholder(tf.int32, shape=[None])
		self.dropout_var 	   = tf.placeholder(tf.float32, name="dropout_var")

	def build_model(self):
		
		W_initializer = tf.truncated_normal_initializer(stddev=self.init_std)
		b_initializer = tf.constant_initializer(0.1, dtype=tf.float32)

		with tf.variable_scope("network"):
			
			self.W_input  = tf.get_variable("W_input", shape=[self.embedding_dim, self.hidden_dim], 
											initializer=W_initializer)
			self.b_input  = tf.get_variable("b_input", shape=[self.hidden_dim], 
											initializer=b_initializer)
			self.W_output = tf.get_variable("W_output", shape=[self.hidden_dim, self.num_genres], 
											 initializer=b_initializer)
			self.b_output = tf.get_variable("b_output", shape=[self.num_genres], 
											 initializer=b_initializer)

			if self.MLP:
				
				hidden_output = tf.matmul(self.audio_batch, self.W_input) + self.b_input

				if self.hidden_nonlinearity == "leaky_relu":
					hidden_output = leaky_relu(hidden_output)
				elif self.hidden_nonlinearity == "relu":
					hidden_output = tf.nn.relu(hidden_output)
				elif self.hidden_nonlinearity == "sigmoid":
					hidden_output = tf.sigmoid(hidden_output)
				elif self.hidden_nonlinearity == "tanh":
					hidden_output = tf.tanh(hidden_output)
				elif self.hidden_nonlinearity == "elu":
					hidden_output = tf.nn.elu(hidden_output)
				
				self.logits = tf.matmul(hidden_output, self.W_output) + self.b_output

			else:

				if self.use_attention:
					
					self.Wc = tf.get_variable("W_context", shape=[2 * self.hidden_dim, self.hidden_dim], 
											  initializer=W_initializer)
					self.bc = tf.get_variable("b_context", shape=[self.hidden_dim], 
											  initializer=b_initializer)
					self.Wa = tf.get_variable("W_attention", shape=[self.hidden_dim, self.hidden_dim],
											  initializer=W_initializer)
					self.ba = tf.get_variable("b_attention", shape=[self.hidden_dim],
											  initializer=b_initializer)

				if self.bidirectional:

					encode_lstm_fw        = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, 
																 		 state_is_tuple=True)
					encode_lstm_fw 		  = tf.nn.rnn_cell.DropoutWrapper(encode_lstm_fw, 
																	      output_keep_prob=1-self.dropout_var)
					encode_lstm_bw        = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, 
																		 state_is_tuple=True)
					encode_lstm_bw 		  = tf.nn.rnn_cell.DropoutWrapper(encode_lstm_bw, 
																	      output_keep_prob=1-self.dropout_var)
					self.bidir_proj       = tf.get_variable("bidir_proj", shape=[2*self.hidden_dim, self.hidden_dim],
															initializer=initializer)
					self.bidir_proj_bias  = tf.get_variable("bidir_proj_bias", shape=[self.hidden_dim],
															initializer=initializer)
				
				if self.use_lstm:
					cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

				else:
					cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim)

				if self.dropout > 0:
					cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-self.dropout)

				split_embeddings = tf.unpack(self.audio_batch, axis=1)

				if self.bidirectional:

					outputs, output_state_fw, output_state_bw = rnn.bidirectional_rnn(encode_lstm_fw, encode_lstm_bw, split_embeddings, dtype=tf.float32)

					projections = tf.matmul(tf.reshape(tf.pack(outputs), [self.max_size * self.batch_size, 2 * self.hidden_dim]), \
											self.bidir_proj) + self.bidir_proj_bias
					hidden_states = tf.unpack(tf.reshape(projections, [self.max_size, self.batch_size, self.hidden_dim]), axis=0)

				else:

					hidden_states, state = rnn.rnn(cell, split_embeddings, dtype=tf.float32)

				if self.num_layers == 2:

					cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

					if self.dropout > 0:
						cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=1-self.dropout)

					# state = (c, h), the last memory and hidden states
					hidden_states, state = rnn.rnn(cell2, hidden_states, dtype=tf.float32, scope="layer2")

				if self.use_attention:

					packed_hidden_states = tf.pack(hidden_states)

					a 					 = tf.matmul(state[1], self.Wa) + self.ba

					attention_scores  	 = tf.reduce_sum(tf.mul(a, packed_hidden_states), 2) # (M, B)

					alpha       	  	 = tf.nn.softmax(tf.transpose(attention_scores))

					c       			 = tf.batch_matmul(tf.transpose(packed_hidden_states, perm=[1, 2, 0]), tf.expand_dims(alpha, 2))

					h_tilde 			 = tf.tanh(tf.matmul(tf.concat(1, [tf.squeeze(c, [2]), a]), self.Wc) + self.bc)

				else:

					h_tilde 			 = state[1]

				self.logits = tf.matmul(h_tilde, self.W_output) + self.b_output
				# (B, H) x (H, G) -> (B, G)

		batch_loss  = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.genres)
		self.loss   = tf.reduce_mean(batch_loss)

		if not self.is_test:
			self.optim = tf.contrib.layers.optimize_loss(self.loss, None, self.current_learning_rate, self.optimizer, clip_gradients=self.grad_max_norm,
									  				 	 summaries=["learning_rate", "gradient_norm", "loss", "gradients"])
		
		self.sess.run(tf.initialize_all_variables())

		with open(self.outfile, 'a') as outfile:
			for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
				print(var.name, file=outfile)
				print(var.get_shape(), file=outfile)	
				outfile.flush()

		self.saver = tf.train.Saver()

	def train(self):

		total_loss = 0.0

		merged_sum = tf.merge_all_summaries()
		t 	   	   = datetime.datetime.now()
		writer     = tf.train.SummaryWriter(os.path.join(self.model_directory, \
										    "logs", "{}-{}-{}-{}-{}".format(t.year, t.month, t.day, t.hour % 12, t.minute)), \
										    self.sess.graph)

		i 					= 0
		# best_valid_loss       = float("inf")
		best_valid_accuracy     = 0

		for epoch in xrange(self.epochs):

			train_loss  = 0.0
			num_batches = 0.0
			
			for audio_batch, genres in data_iterator(self.X_train, self.y_train, self.batch_size):
				
				feed = {self.audio_batch: audio_batch, self.genres: genres, 
						self.dropout_var: self.dropout}

				_, batch_loss, summary = self.sess.run([self.optim, self.loss, merged_sum], feed)

				train_loss += batch_loss
				
				if i % 50 == 0:
					writer.add_summary(summary, i)
					with open(self.outfile, 'a') as outfile:
						print(batch_loss, file=outfile)
						outfile.flush()
				
				i += 1
				num_batches += 1.0

			state = {
				"train_loss" : train_loss / num_batches,
				"epoch" : epoch,
				"learning_rate" : self.current_learning_rate,
			}

			with open(self.outfile, 'a') as outfile:
				print(state, file=outfile)
				outfile.flush()
			
			if self.validate:
				valid_loss, valid_accuracy = self.test()

				# if validation loss increases, halt training
				# model in previous epoch will be saved in checkpoint
				# if valid_loss > best_valid_loss:
				if valid_accuracy < best_valid_accuracy:
					if tolerance >= 20:
						break
					else:
						tolerance += 1
				# save model after validation check
				else:
					tolerance = 0
					self.saver.save(self.sess,
									os.path.join(self.checkpoint_directory, "MemN2N.model")
									)
					# best_valid_loss = valid_loss
					best_valid_accuracy = valid_accuracy
			
			else:
				if epoch % self.save_every == 0:
					self.saver.save(self.sess,
									os.path.join(self.checkpoint_directory, "MemN2N.model")
									)

	def test(self):

		# only load if in test mode (rather than cv)
		if self.is_test:
			self.load()

		test_loss    = 0
		num_batches  = 0.0
		num_correct  = 0.0
		num_examples = 0.0
		
		for audio_batch, genres in data_iterator(self.X_test, self.y_test, self.batch_size):

			feed          = {self.audio_batch: audio_batch, self.genres: genres,\
							 self.dropout_var: 0.0}

			loss, logits  = self.sess.run([self.loss, self.logits], feed)

			test_loss    += loss

			predictions   = np.argmax(logits, 1)
			num_correct  += np.sum(predictions == genres)
			num_examples += predictions.shape[0]

			num_batches  += 1.0

		state = {
			"test_loss" : test_loss / num_batches,
			"accuracy" : num_correct / num_examples
		}

		with open(self.outfile, 'a') as outfile:
			print(state, file=outfile)
			outfile.flush()

		return test_loss / num_batches, num_correct / num_examples

	def run(self):
		if self.is_test:
			self.test()
		else:
			self.train()

	def load(self):
		with open(self.outfile, 'a') as outfile:
			print(" [*] Reading checkpoints...", file=outfile)
			outfile.flush()
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_directory)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			raise Exception(" [!] Test mode but no checkpoint found")



