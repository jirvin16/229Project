from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# import data
import datetime
import os
import time
import sys


class AttentionNN(object):
	def __init__(self, config, sess):
		self.sess = sess

		# Training details
		self.batch_size = config.batch_size
		self.max_size = config.max_size
		self.epochs = config.epochs
		self.current_learning_rate = config.init_learning_rate
		# self.lr = None
		self.loss = None
		self.log_loss = []
		self.log_perp = []
		self.opt = None
		self.optim = None
		self.grad_max_norm = config.grad_max_norm
		self.use_attention = config.use_attention
		self.dropout = config.dropout

		self.show = config.show
		self.is_test = config.is_test
		if self.is_test:
			config.dropout = 0
		self.saver = None

		self.checkpoint_dir = config.checkpoint_dir
		if not os.path.isdir(self.checkpoint_dir):
			if self.is_test:
				raise Exception(" [!] Checkpoints directory %s not found" % self.checkpoint_dir)
			else:
				os.makedirs(self.checkpoint_dir)

		self.data_directory = config.data_directory
		if not os.path.isdir(self.data_directory):
			raise Exception(" [!] Data directory %s not found" % self.data_directory)

		sample_prefix = ""
		if config.sample:
			sample_prefix = "sample."

		self.train_data_path = os.path.join(self.data_directory, sample_prefix + "train")
		self.test_data_path = os.path.join(self.data_directory, sample_prefix + "test")

		# Dimensions and initialization parameters
		self.init_min = -0.1
		self.init_max = 0.1
		self.hidden_dim = config.hidden_dim		
		self.embedding_dim = config.embedding_dim
		self.num_genres = config.num_genres
		self.num_layers = config.num_layers

		# Model placeholders
		self.input_batch = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_size, self.embedding_dim], name="source_batch")
		self.genres = tf.placeholder(tf.int32, shape=[self.batch_size])


	def build_model(self):

		with tf.variable_scope("network"):
			
			self.W_input = tf.get_variable("W_input", shape=[self.embedding_dim, self.hidden_dim], 
											initializer=tf.random_uniform_initializer(minval=self.init_min, maxval=self.init_max), dtype=tf.float32)
			self.b_input = tf.get_variable("b_input", shape=[self.hidden_dim], 
											initializer=tf.random_uniform_initializer(minval=self.init_min, maxval=self.init_max), dtype=tf.float32)
			self.W_output = tf.get_variable("W_output", shape=[self.hidden_dim, self.num_genres], 
											 initializer=tf.random_uniform_initializer(minval=self.init_min, maxval=self.init_max), dtype=tf.float32)
			self.b_output = tf.get_variable("b_output", shape=[self.num_genres], 
											 initializer=tf.random_uniform_initializer(minval=self.init_min, maxval=self.init_max), dtype=tf.float32)
			
			lstm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
			if self.dropout > 0:
				lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=1-self.dropout)
			self.multilayer_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * self.num_layers)
			
			initial_hidden_state = self.multilayer_lstm.zero_state(self.batch_size, dtype=tf.float32)
			hidden_state = initial_hidden_state
			# used for attention
			hidden_states = []
			
			for t in xrange(self.max_size):
				if t >= 1:
					tf.get_variable_scope().reuse_variables()

				print(self.input_batch, self.W_input.get_shape())
				print(tf.matmul(self.input_batch, self.W_input))
				input_projection = tf.squeeze(tf.batch_matmul(self.input_batch, self.W_input)) + self.b_input
				# (B, M, E) x (E, H) -> (B, M, H)
				
				output, hidden_state = self.multilayer_lstm(projection, hidden_state)
				# (B, M, H) -> ... -> (B, H)

				# used for attention
				hidden_states.append(output)

			output_injection = tf.matmul(output, self.W_output) + self.b_output
			# (B, H) x (H, G) -> (B, G)

			output_softmax = tf.softmax(output_injection)
			# (B, G) -> (B, G)

			# input (B, G, 1), (B, ), [(B, )] -> scalar
			# may have to reshape output_softmax and genres (expand_dims) to use seq2seq loss!!
			self.loss = tf.nn.seq2seq.sequence_loss(output_softmax, genres, [tf.ones(self.batch_size)])

		self.optim = tf.contrib.layers.optimize_loss(self.loss, None, self.current_learning_rate, "SGD", clip_gradients=self.grad_max_norm,
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
		t = datetime.datetime.now()
		writer = tf.train.SummaryWriter(os.path.join(self.language, "logs", "{}-{}-{}-{}-{}".format(t.year, t.month, t.day, t.hour % 12, t.minute)), self.sess.graph)

		with open(self.train_source_data_path) as train_data:
			train_size = sum(1 for line in train_data)

		num_batches = int(np.ceil(train_size / self.batch_size))

		if self.show:
			from utils import ProgressBar
			bar = ProgressBar('Train', max=self.epochs)

		i = 0
		for epoch in range(self.epochs):
			if self.show: 
				bar.next()
				sys.stdout.flush()

			train_loss = 0
			for source_batch, target_batch in data.data_iterator(self.train_source_data_path, self.train_target_data_path, self.source_vocab_index, \
																 self.target_vocab_index, self.max_size, self.batch_size):

				feed = {self.source_batch: source_batch, self.target_batch: target_batch}

				_, batch_loss, summary = self.sess.run([self.optim, self.loss, merged_sum], feed)

				train_loss += batch_loss
				
				if i % 50 == 0:
					writer.add_summary(summary, i)
					print(batch_loss)
					sys.stdout.flush()
				
				i += 1

			self.log_loss.append(train_loss / num_batches)

			perplexity = np.exp(train_loss / num_batches)
			self.log_perp.append(perplexity)

			state = {
				"loss" : train_loss / num_batches,
				"perplexity" : perplexity,
				"epoch" : epoch,
				"learning_rate" : self.current_learning_rate,
			}

			print(state)
			sys.stdout.flush()

			# Learning rate annealing 
			if epoch >= 9 and epoch % 2 == 0:
				self.current_learning_rate = self.current_learning_rate / 2.
				# self.lr.assign(self.current_learning_rate).eval()

			self.saver.save(self.sess,
							os.path.join(self.checkpoint_dir, "MemN2N.model")
							)
		if self.show: 
			bar.finish()
			sys.stdout.flush()

	def test(self):
		
		self.load()

		with open(self.test_source_data_path) as test_data:
			test_size = len(test_data.readlines())

		num_batches = int(np.ceil(test_size / self.batch_size))

		if self.show:
			from utils import ProgressBar
			bar = ProgressBar('Train', max=num_batches)
			sys.stdout.flush()

		test_loss = 0

		for source_batch, target_batch in data.data_iterator(self.test_source_data_path, self.test_target_data_path, self.source_vocab_index, \
															self.target_vocab_index, self.max_size, self.batch_size):

			if self.show: 
				bar.next()
				sys.stdout.flush()

			feed = {self.source_batch: source_batch, self.target_batch: target_batch}

			loss, = self.sess.run([self.loss], feed)

			test_loss += loss

		if self.show: 
			bar.finish()
			sys.stdout.flush()

		perplexity = np.exp(test_loss / num_batches)

		state = {
			"loss" : test_loss / num_batches,
			"perplexity" : perplexity,
		}

		print(state)
		sys.stdout.flush()


	def run(self):
		if not self.is_test:
			self.train()
		else:
			self.test()

	def load(self):
		print(" [*] Reading checkpoints...")
		sys.stdout.flush()
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			raise Exception(" [!] Test mode but no checkpoint found")



