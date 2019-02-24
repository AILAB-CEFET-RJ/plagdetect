import tensorflow as tf
import numpy as np

class SiameseLSTMw2v(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
    """
    
    def stackedRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_hidden=hidden_units
        n_layers=3
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        # print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell

        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

            outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)
        return outputs[-1]

    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2

    def tf_f1_score(self, y_true, y_pred):
        """Computes 3 different f1 scores, micro macro
        weighted.
        micro: f1 score accross the classes, as 1
        macro: mean of f1 scores per class
        weighted: weighted average of f1 scores per class,
                weighted from the support of each class


        Args:
            y_true (Tensor): labels, with shape (batch, num_classes)
            y_pred (Tensor): model predictions, same shape as y_true

        Returns:
            tuple(Tensor): (micro, macro, weighted)
                        tuple of the computed f1 scores
        """

        f1 = 0

        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)

        TP = tf.count_nonzero(y_pred * y_true, dtype=tf.float64, name='TP')
        # if tf.is_nan(tf.cast(TP, tf.float64)):
        #     TP = tf.zeros([1], tf.int64)
        FP = tf.count_nonzero(y_pred * (y_true - 1), dtype=tf.float64, name='FP')
        FN = tf.count_nonzero((y_pred - 1) * y_true, dtype=tf.float64, name='FN')

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        f1 = tf.reduce_mean(f1)
        print(f1)


        return f1

    def __init__(
        self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size, trainableEmbeddings):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")
          
        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.constant(0.0, shape=[vocab_size, embedding_size]),
                trainable=trainableEmbeddings,name="W")
            self.embedded_words1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_words2 = tf.nn.embedding_lookup(self.W, self.input_x2)
        print self.embedded_words1
        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1=self.stackedRNN(self.embedded_words1, self.dropout_keep_prob, "side1", embedding_size, sequence_length, hidden_units)
            self.out2=self.stackedRNN(self.embedded_words2, self.dropout_keep_prob, "side2", embedding_size, sequence_length, hidden_units)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
            self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
            self.accuracy = self.tf_f1_score(self.input_y, self.temp_sim)
