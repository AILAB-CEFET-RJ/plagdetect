import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time, memory_profiler as mem_profile
import h5py as h5
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from random import random
from preprocess import MyVocabularyProcessor
import sys, os, shutil

sys.path.append('../')
import datagen.generate_dataset as datagen

reload(sys)
sys.setdefaultencoding("utf-8")


class InputHelper(object):
	pre_emb = dict()
	vocab_processor = None

	def cleanText(self, s):
		s = re.sub(r"[^\x00-\x7F]+", " ", s)
		s = re.sub(r'[\~\!\`\^\*\{\}\[\]\#\<\>\?\+\=\-\_\(\)]+', "", s)
		s = re.sub(r'( [0-9,\.]+)', r"\1 ", s)
		s = re.sub(r'\$', " $ ", s)
		s = re.sub('[ ]+', ' ', s)
		return s.lower()

	def getVocab(self, vocab_path, max_document_length, filter_h_pad):
		if self.vocab_processor == None:
			print('locading vocab')
			vocab_processor = MyVocabularyProcessor(max_document_length - filter_h_pad, min_frequency=0)
			self.vocab_processor = vocab_processor.restore(vocab_path)
		return self.vocab_processor

	def loadW2V(self, emb_path, type="bin"):
		print("Loading W2V data...")
		num_keys = 0
		if type == "textgz":
			# this seems faster than gensim non-binary load
			for line in gzip.open(emb_path):
				l = line.strip().split()
				st = l[0].lower()
				self.pre_emb[st] = np.asarray(l[1:])
			num_keys = len(self.pre_emb)
		if type == "text":
			# this seems faster than gensim non-binary load
			for line in open(emb_path):
				l = line.strip().split()
				st = l[0].lower()
				self.pre_emb[st] = np.asarray(l[1:])
			num_keys = len(self.pre_emb)
		else:
			self.pre_emb = Word2Vec.load_word2vec_format(emb_path, binary=True)
			self.pre_emb.init_sims(replace=True)
			num_keys = len(self.pre_emb.vocab)
		print("loaded word2vec len ", num_keys)
		gc.collect()

	def deletePreEmb(self):
		self.pre_emb = dict()
		gc.collect()

	def getTsvData(self, filepath):
		print("Loading training data from " + filepath)
		x1 = []
		x2 = []
		y = []
		# positive samples from file
		for line in open(filepath):
			l = line.strip().split("\t")
			if len(l) < 2:
				continue
			if random() > 0.5:
				x1.append(l[0].lower())
				x2.append(l[1].lower())
			else:
				x1.append(l[1].lower())
				x2.append(l[0].lower())
			y.append(int(l[2]))
		return np.asarray(x1), np.asarray(x2), np.asarray(y)

	def getTsvDataCharBased(self, filepath):
		print("Loading training data from " + filepath)
		x1 = []
		x2 = []
		y = []
		# positive samples from file
		for line in open(filepath):
			l = line.strip().split("\t")
			if len(l) < 2:
				continue
			if random() > 0.5:
				x1.append(l[0].lower())
				x2.append(l[1].lower())
			else:
				x1.append(l[1].lower())
				x2.append(l[0].lower())
			y.append(1)  # np.array([0,1]))
		# generate random negative samples
		combined = np.asarray(x1 + x2)
		shuffle_indices = np.random.permutation(np.arange(len(combined)))
		combined_shuff = combined[shuffle_indices]
		for i in xrange(len(combined)):
			x1.append(combined[i])
			x2.append(combined_shuff[i])
			y.append(0)  # np.array([1,0]))
		return np.asarray(x1), np.asarray(x2), np.asarray(y)

	def getTsvTestData(self, filepath):
		print("Loading testing/labelled data from " + filepath)
		x1 = []
		x2 = []
		y = []
		# positive samples from file
		for line in open(filepath):
			l = line.strip().split("\t")
			if len(l) < 3:
				continue
			x1.append(l[1].lower())
			x2.append(l[2].lower())
			y.append(int(l[0]))  # np.array([0,1]))
		return np.asarray(x1), np.asarray(x2), np.asarray(y)

	def batch_batch_iter(self, cursor, number_of_samples, batch_size, num_epochs):
		cursor.execute('select * from dataset_sentence')
		print cursor.rowcount
		tuples = cursor.fetchmany(number_of_samples)
		for epoch in range(num_epochs):
			while len(tuples) > 0:
				data = self.my_batch_iter(tuples, batch_size)
				yield data

	def my_batch_iter(self, data, batch_size, shuffle=True):
		"""
		Generates a batch iterator for a dataset.
		"""
		data = np.asarray(data)
		print(data)
		print(data.shape)
		data_size = len(data)
		num_batches_per_epoch = int(len(data) / batch_size) + 1
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

	def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
		"""
		Generates a batch iterator for a dataset.
		"""
		data = np.asarray(data)
		print(data)
		print(data.shape)
		data_size = len(data)
		num_batches_per_epoch = int(len(data) / batch_size) + 1
		for epoch in range(num_epochs):
			# Shuffle the data at each epoch
			if shuffle:
				shuffle_indices = np.random.permutation(np.arange(data_size))
				shuffled_data = data[shuffle_indices]
			else:
				shuffled_data = data
			for batch_num in range(num_batches_per_epoch):
				start_index = batch_num * batch_size
				end_index = min((batch_num + 1) * batch_size, data_size)
				yield shuffled_data[start_index:end_index]

	def dumpValidation(self, x1_text, x2_text, y, shuffled_index, dev_idx, i):
		print("dumping validation " + str(i))
		x1_shuffled = x1_text[shuffled_index]
		x2_shuffled = x2_text[shuffled_index]
		y_shuffled = y[shuffled_index]
		x1_dev = x1_shuffled[dev_idx:]
		x2_dev = x2_shuffled[dev_idx:]
		y_dev = y_shuffled[dev_idx:]
		del x1_shuffled
		del y_shuffled
		with open('validation.txt' + str(i), 'w') as f:
			for text1, text2, label in zip(x1_dev, x2_dev, y_dev):
				f.write(str(label) + "\t" + text1 + "\t" + text2 + "\n")
			f.close()
		del x1_dev
		del y_dev

	# Data Preparatoin
	# ==================================================

	def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size, is_char_based):
		if is_char_based:
			x1_text, x2_text, y = self.getTsvDataCharBased(training_paths)
		else:
			x1_text, x2_text, y = self.getTsvData(training_paths)
		# Build vocabulary
		print("Building vocabulary")
		vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0, is_char_based=is_char_based)
		vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
		print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))
		i1 = 0
		train_set = []
		dev_set = []
		sum_no_of_batches = 0
		x1 = np.asarray(list(vocab_processor.transform(x1_text)))
		x2 = np.asarray(list(vocab_processor.transform(x2_text)))
		# Randomly shuffle data
		np.random.seed(131)
		shuffle_indices = np.random.permutation(np.arange(len(y)))
		x1_shuffled = x1[shuffle_indices]
		x2_shuffled = x2[shuffle_indices]
		y_shuffled = y[shuffle_indices]
		dev_idx = -1 * len(y_shuffled) * percent_dev // 100
		del x1
		del x2
		# Split train/test set
		self.dumpValidation(x1_text, x2_text, y, shuffle_indices, dev_idx, 0)
		# TODO: This is very crude, should use cross-validation
		x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
		x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
		y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
		print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
		sum_no_of_batches = sum_no_of_batches + (len(y_train) // batch_size)
		train_set = (x1_train, x2_train, y_train)
		dev_set = (x1_dev, x2_dev, y_dev)
		gc.collect()
		return train_set, dev_set, vocab_processor, sum_no_of_batches

	def myGetDataSets(self, cursor, max_document_length, percent_dev, batch_size, is_char_based, number_of_samples):
		# edited
		start_time = time.time()
		cursor.execute('select * from dataset_sentence')
		end_time = time.time()
		print('Time elapsed on running select all: {} seconds.'.format(round(end_time-start_time, 2)))


		start_time = time.time()
		tuples = cursor.fetchmany(number_of_samples)
		end_time = time.time()
		print('Time elapsed on fetching {} lines: {} seconds.'.format(number_of_samples, round(end_time-start_time, 2)))


		x1_text = np.asarray([i[0] for i in tuples])
		x2_text = np.asarray([i[1] for i in tuples])
		y = np.asarray([i[2] for i in tuples])

		# Build vocabulary
		print("Building vocabulary")
		vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0, is_char_based=is_char_based)
		vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
		print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))
		i1 = 0
		train_set = []
		dev_set = []
		sum_no_of_batches = 0
		x1 = np.asarray(list(vocab_processor.transform(x1_text)))
		x2 = np.asarray(list(vocab_processor.transform(x2_text)))
		# Randomly shuffle data
		np.random.seed(131)
		shuffle_indices = np.random.permutation(np.arange(len(y)))
		x1_shuffled = x1[shuffle_indices]
		x2_shuffled = x2[shuffle_indices]
		y_shuffled = y[shuffle_indices]
		dev_idx = -1 * len(y_shuffled) * percent_dev // 100
		del x1
		del x2
		# Split train/test set
		self.dumpValidation(x1_text, x2_text, y, shuffle_indices, dev_idx, 0)
		# TODO: This is very crude, should use cross-validation
		x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
		x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
		y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
		print("Train/Dev split for {}: {:d}/{:d}".format('dataset_id', len(y_train), len(y_dev)))
		sum_no_of_batches = sum_no_of_batches + (len(y_train) // batch_size)
		train_set = (x1_train, x2_train, y_train)
		dev_set = (x1_dev, x2_dev, y_dev)
		gc.collect()
		return train_set, dev_set, vocab_processor, sum_no_of_batches

	def getTestDataSet(self, data_path, vocab_path, max_document_length):
		x1_temp, x2_temp, y = self.getTsvTestData(data_path)

		# Build vocabulary
		vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
		vocab_processor = vocab_processor.restore(vocab_path)
		print len(vocab_processor.vocabulary_)

		x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
		x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
		# Randomly shuffle data
		del vocab_processor
		gc.collect()
		return x1, x2, y

	def getEmbeddingsMap(self, cursor, max_document_length):
		print('Loading sentences')
		print('Memory (before): {}Mb'.format(mem_profile.memory_usage()))
		ids, sentences = map(list, zip(*datagen.get_sentences_list(cursor)))
		print('Memory (after): {}Mb\n'.format(mem_profile.memory_usage()))

		# Build vocabulary
		print("Building vocabulary")
		vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0, is_char_based=False)
		#sentences_array = np.asarray(sentences) # line in which memory error occurs with full list of datasets (size = 6620242)

		print('Memory (before): {}Mb'.format(mem_profile.memory_usage()))
		start_time = time.time()
		vocab_processor.fit_transform(sentences)
		end_time = time.time()
		print('Time elapsed on vocabulary fitting (fit_transform): {} seconds.'.format(round(end_time-start_time, 2)))
		print('Memory (after): {}Mb'.format(mem_profile.memory_usage()))

		print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))
		print('Vocabulary created!\n')

		print('Memory (before): {}Mb'.format(mem_profile.memory_usage()))
		start_time = time.time()
		embeddings = np.asarray(list(vocab_processor.transform(sentences)))
		end_time = time.time()
		print('Time elapsed on sentences to word ids (transform): {} seconds.'.format(round(end_time-start_time, 2)))
		print('Memory (after): {}Mb\n'.format(mem_profile.memory_usage()))

		print('Embeddings generated in memory!')

		gc.collect()
		return dict(zip(ids, embeddings)), vocab_processor

	def build_datasets(self, cursor, total_size, batch_size, percent_dev):
		print('Building dataset files...')
		dev_batch_size = batch_size * percent_dev // 100
		train_batch_size = (batch_size * (100 - percent_dev) // 100) + 1
		
		cursor.execute('select * from dataset_id')

		if os.path.exists('ds'):
			shutil.rmtree('ds', ignore_errors=True)
		os.mkdir('ds')
		with h5.File('ds/train.hdf5', 'w') as ft, h5.File('ds/dev.hdf5', 'w') as fd:
			tset = ft.create_dataset("chunked_train", shape=(0, 3), dtype='int', maxshape=(None, 3), chunks=(batch_size, 3),
															 compression="gzip", compression_opts=9)
			dset = fd.create_dataset("chunked_dev", shape=(0, 3), dtype='int', maxshape=(None, 3), chunks=(batch_size, 3),
															 compression="gzip", compression_opts=9)
			train_count, dev_count = 0, 0
			for i in range(total_size / batch_size + 1):
				batch = cursor.fetchmany(batch_size)
				l_size = len(batch)
				dev_idx = l_size * percent_dev // 100
				dev, train = batch[:dev_idx], batch[dev_idx:]

				tset.resize(tuple(map(sum, zip(tset.shape, (len(train), 0)))))
				dset.resize(tuple(map(sum, zip(dset.shape, (len(dev), 0)))))
				tset[i*train_batch_size : i*train_batch_size + len(train)] = train
				dset[i*dev_batch_size : i*dev_batch_size + len(dev)] = dev

				train_count = train_count + len(train)
				dev_count = dev_count + len(dev)
				percent_complete = ((i+1) * batch_size) / float(total_size)  * 100
				if percent_complete > 100:
					percent_complete = 100
				print('Adding batch to dataset. %.2f%% complete.' % percent_complete)

			print('Dataset files built!')
			return train_count, dev_count
			
			
			

	def my_train_batch(self, embeddings_map, total_size, batch_size, num_epochs, shuffle=True):
		num_batches_per_epoch = int(total_size/batch_size) + 1

		for epoch in range(num_epochs):
			##sets cursor
			#cursor.execute('select * from dataset_train')
			
			with h5.File('ds/train.hdf5', 'r') as f:
				dset = f['chunked_train']
				
				for batch_num in range(num_batches_per_epoch):
					# fetches batch_size rows from dataset, replacing ids for embeddings
					beg, end = (batch_num*batch_size), ((batch_num+1)*batch_size-1)
					if batch_num == num_batches_per_epoch-1:
						ids = dset[beg:, :]
					else:
						ids = dset[beg:end, :]
					data = self.ids_to_embeddings(embeddings_map, ids)
					data = np.asarray(data)
	
					# Shuffle the data at each epoch
					if shuffle:
						shuffle_indices = np.random.permutation(np.arange(data.shape[0]))
						shuffled_data = data[shuffle_indices]
					else:
						shuffled_data = data
	
					yield shuffled_data

	def my_dev_batch(self, embeddings_map, total_size, batch_size, num_epochs, shuffle=True):
		num_batches_per_epoch = int(total_size/batch_size) + 1

		for epoch in range(num_epochs):
			##sets cursor
			#cursor.execute('select * from dataset_train')
			
			with h5.File('ds/dev.hdf5', 'r') as f:
				dset = f['chunked_dev']
	
				for batch_num in range(num_batches_per_epoch):
					# fetches batch_size rows from dataset, replacing ids for embeddings
					beg, end = (batch_num*batch_size), ((batch_num+1)*batch_size-1)
					if batch_num == num_batches_per_epoch-1:
						ids = dset[beg:, :]
					else:
						ids = dset[beg:end, :]
					data = self.ids_to_embeddings(embeddings_map, ids)
					data = np.asarray(data)
	
					# Shuffle the data at each epoch
					if shuffle:
						shuffle_indices = np.random.permutation(np.arange(data.shape[0]))
						shuffled_data = data[shuffle_indices]
					else:
						shuffled_data = data
	
					yield shuffled_data

	def ids_to_embeddings(self, emb_map, rows):
		x1, x2, y = zip(*rows)
		x1 = list(x1)
		x2 = list(x2)
		for i in range(len(y)):
			x1[i] = emb_map[x1[i]]
			x2[i] = emb_map[x2[i]]
		x1 = np.asarray(x1)
		x2 = np.asarray(x2)
		y = np.asarray(y)
		return zip(x1, x2, y)

	def my_get_counts(self, cursor):

		print('Counting train dataset')
		start_time = time.time()
		cursor.execute('select count(*) from dataset_id')
		end_time = time.time()
		print('Time elapsed on counting training dataset: {} seconds.'.format(round(end_time - start_time, 2)))
		count = cursor.fetchall()[0][0]

		return count
