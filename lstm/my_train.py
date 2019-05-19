#! /usr/bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from input_helpers import InputHelper
from siamese_network import SiameseLSTM
from siamese_network_semantic import SiameseLSTMw2v
from random import random
import sqlite3 as lite
import sys
import math
import pickle

# Parameters
# ==================================================

tf.flags.DEFINE_boolean("is_char_based", False, "is character based syntactic similarity. "
                                               "if false then word embedding based semantic similarity is used."
                                               "(default: False)")

tf.flags.DEFINE_string("word2vec_model", "wiki.simple.vec", "word2vec pre-trained embeddings file (default: wiki.simple.vec)")
tf.flags.DEFINE_string("word2vec_format", "text", "word2vec pre-trained embeddings file format (bin/text/textgz)(default: text)")

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_string("database", "../plag.db", "training file (default: ../plag.db)")
tf.flags.DEFINE_string("training_folder", 'ds', "path to folder containing dataset (default: ds)")
tf.flags.DEFINE_integer("hidden_units", 50, "Number of hidden units (default:50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 300)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 1)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 50)")
tf.flags.DEFINE_integer("patience", 20, "Patience for early stopping (default: 20)")
tf.flags.DEFINE_integer("log_every", 1000, "Log results every X steps (default: 100000)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

batch_size = FLAGS.batch_size
num_epochs = FLAGS.num_epochs

print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.database==None:
    print("Input Files List is empty. use -database argument.")
    exit()

max_document_length=15
#max_document_length=sys.maxint # attempt to read all words in a document
inpH = InputHelper()
#train_set, dev_set, vocab_processor,sum_no_of_batches = inpH.getDataSets(FLAGS.database,max_document_length, 10,
#                                                                         FLAGS.batch_size, FLAGS.is_char_based)

num_docs = inpH.get_num_docs(FLAGS.training_folder)

db = lite.connect(FLAGS.database)
cursor = db.cursor()
emb_map, vocab_processor = inpH.getEmbeddingsMap(cursor, max_document_length, num_docs)
train_count, dev_count = inpH.get_counts(FLAGS.training_folder)[0:2]
total_count = train_count + dev_count

sum_no_of_batches = int(math.ceil(float(train_count) / batch_size))
dev_no_of_batches = int(math.ceil(float(dev_count) / batch_size))

train_set = inpH.my_train_batch(emb_map, train_count, FLAGS.batch_size, num_epochs)

dev_set = inpH.my_dev_batch(emb_map, dev_count, FLAGS.batch_size, num_epochs)

# train_set, dev_set, sum_no_of_batches = inpH.myGetDataSets(cursor ,max_document_length, 10,
#                                                                          FLAGS.batch_size, FLAGS.is_char_based, 1000)

trainableEmbeddings=False
if FLAGS.is_char_based==True:
    FLAGS.word2vec_model = False
else:
    if FLAGS.word2vec_model==None:
        trainableEmbeddings=True
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
          "You are using word embedding based semantic similarity but "
          "word2vec model path is empty. It is Recommended to use  --word2vec_model  argument. "
          "Otherwise now the code is automatically trying to learn embedding values (may not help in accuracy)"
          "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:
        inpH.loadW2V(FLAGS.word2vec_model, FLAGS.word2vec_format)

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        if FLAGS.is_char_based:
            siameseModel = SiameseLSTM(
                sequence_length=max_document_length,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_units=FLAGS.hidden_units,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                batch_size=FLAGS.batch_size
            )
        else:
            siameseModel = SiameseLSTMw2v(
                sequence_length=max_document_length,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_units=FLAGS.hidden_units,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                batch_size=FLAGS.batch_size,
                trainableEmbeddings=trainableEmbeddings
            )
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        print(siameseModel.accuracy)
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("initialized siameseModel object")
    
    grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    # grad_summaries = []
    # for g, v in grads_and_vars:
    #     if g is not None:
    #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
    #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
    #         grad_summaries.append(grad_hist_summary)
    #         grad_summaries.append(sparsity_summary)
    # grad_summaries_merged = tf.summary.merge(grad_summaries)
    # print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", siameseModel.loss)
    acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)

    # Train Summaries
    # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    # train_summary_dir = os.path.join(out_dir, "summaries", "train")
    # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    # dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    # dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    
    # my summary dir
    summary_dir = os.path.join(out_dir, "summaries")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Write ids_mapping
    pickle.dump(emb_map, open(os.path.join(checkpoint_dir + '/ids_mapping'), 'w'))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)

    if FLAGS.word2vec_model :
        # initial matrix with random uniform
        initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        #initW = np.zeros(shape=(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        # load any vectors from the word2vec
        print("initializing initW with pre-trained word2vec embeddings")
        for w in vocab_processor.vocabulary_._mapping:
            arr=[]
            s = re.sub('[^0-9a-zA-Z]+', '', w)
            if w in inpH.pre_emb:
                arr=inpH.pre_emb[w]
            elif w.lower() in inpH.pre_emb:
                arr=inpH.pre_emb[w.lower()]
            elif s in inpH.pre_emb:
                arr=inpH.pre_emb[s]
            elif s.isdigit():
                arr=inpH.pre_emb["zero"]
            if len(arr)>0:
                idx = vocab_processor.vocabulary_.get(w)
                initW[idx]=np.asarray(arr).astype(np.float32)
        print("Done assigning intiW. len="+str(len(initW)))
        inpH.deletePreEmb()
        gc.collect()
        sess.run(siameseModel.W.assign(initW))

    def train_step(x1_batch, x2_batch, y_batch, epoch, batch):
        """
        A single training step
        """
        if random()>0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        _, step, loss, accuracy, dist, sim = sess.run([tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.distance, siameseModel.temp_sim],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        if batch*(epoch+1) % FLAGS.log_every == 0:
            print("TRAIN {}: epoch/step {}/{}, loss {:g}, f1 {:g}".format(time_str, epoch, batch, loss, accuracy))
        #train_summary_writer.add_summary(summaries, step)
        # print(y_batch, dist, sim)
        return loss, accuracy

    def dev_step(x1_batch, x2_batch, y_batch, epoch, batch):
        """
        A single training step
        """ 
        if random()>0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        step, loss, accuracy, sim = sess.run([global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.temp_sim],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        if batch*(epoch+1) % FLAGS.log_every == 0:
            print("DEV {}: epoch/batch {}/{}, loss {:g}, f1 {:g}".format(time_str, epoch, batch, loss, accuracy))
        #dev_summary_writer.add_summary(summaries, step)
        return loss, accuracy

    # Generate batches
    # batches=inpH.batch_batch_iter(
    #             list(zip(train_set[0], train_set[1], train_set[2])), 128, FLAGS.batch_size, FLAGS.num_epochs)

    train_batches = train_set
    dev_batches = dev_set
    ptr=0
    max_validation_f1=0.0
    stopping_step = 0
    best_loss = sys.float_info.max

    for epoch in xrange(FLAGS.num_epochs):
        start_time = time.time()

        current_step = tf.train.global_step(sess, global_step)
        losses = []
        f1s = []

        for nn in xrange(sum_no_of_batches):
            train_batch = train_batches.next()
            if len(train_batch)<1:
                continue
            x1_batch,x2_batch, y_batch = zip(*train_batch)
            if len(y_batch)<1:
                continue
            loss, f1 = train_step(x1_batch, x2_batch, y_batch, epoch, nn)
            losses.append(loss)
            f1s.append(f1)
          
        epoch_f1 = np.mean(np.nan_to_num(f1s))
        epoch_loss = np.mean(np.nan_to_num(losses))
        
        with open(os.path.join(summary_dir, 'train_summary'), 'a') as f:
            f.write('\t'.join((str(epoch), str(epoch_loss), str(epoch_f1))) + '\n')
            
        

        if epoch % FLAGS.evaluate_every == 0:
            losses = []
            f1s = []
            
            print("\nEvaluation:")
            for _ in xrange(dev_no_of_batches):
                dev_batch = dev_batches.next()
                if len(dev_batch)<1:
                    continue
                x1_dev_b,x2_dev_b,y_dev_b = zip(*dev_batch)
                if len(y_dev_b)<1:
                    continue
                loss, f1 = dev_step(x1_dev_b, x2_dev_b, y_dev_b, epoch, nn)
                losses.append(loss)
                f1s.append(f1)

            epoch_f1 = np.mean(np.nan_to_num(f1s))
            epoch_loss = np.mean(np.nan_to_num(losses))

            with open(os.path.join(summary_dir, 'dev_summary'), 'a') as f:
                f.write('\t'.join((str(epoch), str(epoch_loss), str(epoch_f1))) + '\n')
        
        if epoch % FLAGS.checkpoint_every == 0:
            if epoch_f1 >= max_validation_f1:
                max_validation_f1 = epoch_f1
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(epoch)+".pb", as_text=False)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_f1, checkpoint_prefix))

        # early stopping
        if epoch_loss < best_loss:
            stopping_step = 0
            best_loss = epoch_loss
        else:
            stopping_step += 1
        if stopping_step >= FLAGS.patience:
            print("Early stopping is trigger at epoch: {} loss:{}".format(epoch, epoch_loss))
            saver.save(sess, checkpoint_prefix, global_step=current_step)
            tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(epoch)+".pb", as_text=False)
            print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(epoch, max_validation_f1, checkpoint_prefix))
            exit(0)

        end_time = time.time()
        print('Time spent on epoch {}: {:.2f} seconds'.format(epoch, end_time-start_time))

    print("End of training.")
    saver.save(sess, checkpoint_prefix, global_step=current_step)
    tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph" + str(epoch) + ".pb", as_text=False)
    print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_f1, checkpoint_prefix))
