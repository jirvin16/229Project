from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from data import data_iterator
import datetime
import os
import time
import sys
import h5py
from pprint import pprint


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
		self.optim 				   = None
		self.loss                  = None

		self.data_directory 	   = "data_vectors/"
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
			X    		 	 = hf["X"][:]
			genres 	 		 = hf["y"][:]
			self.genre_names = hf["genres"][:]
			self.mean   	 = hf["mean"]
			self.std    	 = hf["std"]

		if self.MLP:
			timestep_X  	  = X
		else:
			assert X.shape[1] % self.embedding_dim == 0
			timesteps 		  = int(X.shape[1] / self.embedding_dim)
			self.max_size     = timesteps
			timestep_X		  = X.reshape(X.shape[0], timesteps, self.embedding_dim)

		np.random.seed(self.random_seed)
		train_size 	      = int(0.7 * X.shape[0])
		validation_size   = int(0.15 * X.shape[0])
		permutation 	  = np.random.permutation(X.shape[0])
		shuffled_X 	      = timestep_X[permutation]
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
		# Going to need to pad inputs so that they are all the same length
		if self.MLP:
			self.embedding_dim = X.shape[1]
			self.audio_batch   = tf.placeholder(tf.float32, shape=[None, self.embedding_dim], name="audio_batch")
		else:
			self.audio_batch   = tf.placeholder(tf.float32, shape=[None, self.max_size, self.embedding_dim], name="audio_batch")
		self.genres 		   = tf.placeholder(tf.int32, shape=[None])
		self.dropout_var 	   = tf.placeholder(tf.float32, name="dropout_var")
		self.batch_size_var    = tf.placeholder(tf.int32, name="batch_size")

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
				
				first_layer = tf.nn.relu(tf.matmul(self.audio_batch, self.W_input) + self.b_input)
				
				self.logits = tf.matmul(first_layer, self.W_output) + self.b_output

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
				
				lstm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
				if self.dropout > 0:
					lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=1-self.dropout)
				self.multilayer_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * self.num_layers)

				split_embeddings 	 = tf.split(1, self.max_size, self.audio_batch)
				
				initial_hidden_state = self.multilayer_lstm.zero_state(self.batch_size_var, dtype=tf.float32)
				hidden_state 		 = initial_hidden_state
				
				# used for attention
				hidden_states 		 = []
				
				for t in xrange(self.max_size):
					if t >= 1:
						tf.get_variable_scope().reuse_variables()

					input_projection 	 = tf.matmul(tf.squeeze(split_embeddings[t], [1]), self.W_input) + self.b_input
					
					output, hidden_state = self.multilayer_lstm(input_projection, hidden_state)

					# used for attention
					hidden_states.append(output)

				if self.use_attention:

					packed_hidden_states = tf.pack(hidden_states)

					a 					 = tf.matmul(output, self.Wa) + self.ba

					attention_scores  	 = tf.reduce_sum(tf.mul(a, packed_hidden_states), 2) # (M, B)

					alpha       	  	 = tf.nn.softmax(tf.transpose(attention_scores))

					c       			 = tf.batch_matmul(tf.transpose(packed_hidden_states, perm=[1, 2, 0]), tf.expand_dims(alpha, 2))

					h_tilde 			 = tf.tanh(tf.matmul(tf.concat(1, [tf.squeeze(c, [2]), a]), self.Wc) + self.bc)

				else:

					h_tilde 			 = output

				self.logits = tf.matmul(h_tilde, self.W_output) + self.b_output
				# (B, H) x (H, G) -> (B, G)

		batch_loss  = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.genres)
		self.loss   = tf.reduce_mean(batch_loss)

		if not self.is_test:
			self.optim = tf.contrib.layers.optimize_loss(self.loss, None, self.current_learning_rate, self.optimizer, clip_gradients=self.grad_max_norm,
									  				 	 summaries=["learning_rate", "gradient_norm", "loss", "gradients"])
		
		self.sess.run(tf.initialize_all_variables())

		for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
			print(var.name)
			print(var.get_shape())	
			sys.stdout.flush()

		self.saver = tf.train.Saver()

	def train(self):

		total_loss = 0.0

		merged_sum = tf.merge_all_summaries()
		t 	   	   = datetime.datetime.now()
		writer     = tf.train.SummaryWriter(os.path.join(self.model_directory, \
										    "logs", "{}-{}-{}-{}-{}".format(t.year, t.month, t.day, t.hour % 12, t.minute)), \
										    self.sess.graph)

		i 					= 0
		previous_train_loss = float("inf")
		valid_loss 			= float("inf")
		best_valid_loss     = float("inf")
		
		for epoch in xrange(self.epochs):

			train_loss  = 0.0
			num_batches = 0.0
			
			for audio_batch, genres in data_iterator(self.X_train, self.y_train, self.batch_size):
				
				feed = {self.audio_batch: audio_batch, self.genres: genres, 
						self.dropout_var: self.dropout, self.batch_size_var: audio_batch.shape[0]}

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
				previous_valid_loss = valid_loss
				valid_loss = self.test()

				# if validation loss increases, halt training
				# model in previous epoch will be saved in checkpoint
				# if valid_loss > previous_valid_loss:
				# 	break

			# Adaptive learning rate
			if previous_train_loss <= train_loss + 1e-1:
				self.current_learning_rate /= 2.

			# save model after validation check
			if (epoch % self.save_every == 0 or epoch == self.epochs - 1) and valid_loss <= best_valid_loss:
				self.saver.save(self.sess,
								os.path.join(self.checkpoint_directory, "MemN2N.model")
								)
				best_valid_loss = valid_loss

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
							 self.dropout_var: 0.0, self.batch_size_var: audio_batch.shape[0]}

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

		return test_loss / num_batches

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



