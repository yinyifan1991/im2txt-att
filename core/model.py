
from __future__ import division

import tensorflow as tf


class ShowAndTellModel(object):
    def __init__(self, word_to_idx, mode='train', features_L=196,features_D=512, dim_embed=512, dim_hidden=1024, n_time_step=16, 
                  alpha_c=0.0, lstm_dropout_keep_prob=0.5):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            mode: train or evaluation
            features_L, features_D: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM. 
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            lstm_dropout_keep_prob: (optional) The possibility a hidden layer to be kept.
        """
        assert mode in ["train", "eval"]
        self.word_to_idx = word_to_idx
        self.vocab_size = len(word_to_idx)
        self.mode = mode
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.alpha_c = alpha_c
        self.lstm_dropout_keep_prob = lstm_dropout_keep_prob
        self.V = len(word_to_idx)
        self.L = features_L
        self.D = features_D
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self.global_step = 0

        self.weight_initializer = tf.random_uniform_initializer(minval=-0.08, maxval=0.08)
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        

    def _is_train(self):
        return self.mode == 'train'
    
    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            params = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(params, inputs, name='word_vector')
            return x
  
    def _doubly_stochastic_regularize(self, alpha_list, ds_lambda=1):
        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))
        alphas_sum = tf.reduce_sum(alphas, 1)
        temp_sum = tf.reduce_sum((16./196 - alphas_sum)**2)
        regularization = ds_lambda * temp_sum
        return regularization
  
    def _decode_lstm(self, x, h, z):
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE) as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                    inputs=h,
                    num_outputs=self.vocab_size,
                    activation_fn=None,
                    weights_initializer=self.weight_initializer,
                    scope=logits_scope)
            return logits
    
    def _doubly_stochastic(self, z, h):
        with tf.variable_scope('doubly_stochastic', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta') 
            z = tf.multiply(beta, z, name='doubly_stochastic_context') 
            return z, beta
        
    def _process_features(self, features):
        with tf.variable_scope('process_features', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_temp = tf.reshape(features, [-1, self.D])
            features = tf.reshape(tf.matmul(features_temp, w), [-1, self.L, self.D])
            tf.get_variable_scope().reuse_variables()
            return features
        
    def _attention_layer(self, features, h):
        features_proj = self._process_features(features)
        with tf.variable_scope('attention_layer', reuse=tf.AUTO_REUSE):
            w_h = tf.get_variable('w_h', [self.H, self.D], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.D], initializer=self.const_initializer)
            h_proj = tf.matmul(h, w_h)
            h_att = tf.nn.tanh(features_proj + tf.expand_dims(h_proj, 1) + b_h)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)
            b_att = tf.get_variable('b_att', [1], initializer=self.const_initializer)
            h_att_inter = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att) + b_att, [-1, self.L])
            alpha = tf.nn.softmax(h_att_inter)
            z = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')
            
            return z, alpha
            
    
    def _init_lstm_state(self, features):
        with tf.variable_scope('init_lstm'):
            features_mean = tf.reduce_mean(features, 1)
            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            
            c = tf.nn.sigmoid(tf.matmul(features_mean, w_c) + b_c)
            h = tf.nn.sigmoid(tf.matmul(features_mean, w_h) + b_h)

            return c, h
        
    def setup_global_step(self):
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    
        self.global_step = global_step

    def build_model(self, max_len=20):
        features = self.features
        # batch normalize feature vectors
        features = tf.contrib.layers.batch_norm(inputs=features,
                                                decay=0.95,
                                                center=True,
                                                scale=True,
                                                is_training=True,
                                                updates_collections=None,
                                                scope='batch_norm')
        
        c, h = self._init_lstm_state(features)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        loss = 0.0
        alpha_list = []
        if self._is_train():
            captions = self.captions
            batch_size = tf.shape(features)[0]
            input_seq = tf.slice(captions, [0, 0], [-1, self.T])
            target_seq = tf.slice(captions, [0, 1], [-1, self.T])
            mask = tf.to_float(tf.not_equal(target_seq, self._null))
            x = self._word_embedding(inputs=input_seq)

            if self._is_train():
                lstm_cell = tf.contrib.rnn.DropoutWrapper(
                        lstm_cell,
                        input_keep_prob=self.lstm_dropout_keep_prob,
                        output_keep_prob=self.lstm_dropout_keep_prob)
    
            for t in range(self.T):
                z, alpha = self._attention_layer(features, h)
                alpha_list.append(alpha)
    
                z, beta = self._doubly_stochastic(z, h) 
    
                with tf.variable_scope('lstm', reuse=(t!=0)):
                    _, (c, h) = lstm_cell(inputs=tf.concat([x[:,t,:], z], 1), state=[c, h])
    
                logits = self._decode_lstm(x[:,t,:], h, z)
                loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_seq[:, t], logits=logits) * mask[:, t])
               
            if self.alpha_c > 0:
                ds_reg = self._doubly_stochastic_regularize(alpha_list, ds_lambda=self.alpha_c)
                loss += ds_reg
                
            
            self.setup_global_step()    
            return loss / tf.to_float(batch_size)
        else:
            beta_list = []
            sampled_word_list = []
            for t in range(max_len):
                if t == 0:
                    x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
                else:
                    x = self._word_embedding(inputs=sampled_word, reuse=True)  
              
                z, alpha = self._attention_layer(features, h)
                alpha_list.append(alpha)
    
                z, beta = self._doubly_stochastic(z, h) 
                beta_list.append(beta)
    
                with tf.variable_scope('lstm', reuse=(t!=0)):
                    _, (c, h) = lstm_cell(inputs=tf.concat([x, z], 1), state=[c, h])
    
                logits = self._decode_lstm(x, h, z)
                sampled_word = tf.argmax(logits, 1)       
                sampled_word_list.append(sampled_word)     

            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))
            betas = tf.transpose(tf.squeeze(beta_list), (1, 0)) 
            generated_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))  
            return alphas, betas, generated_captions
        
        
