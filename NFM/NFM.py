import pandas as pd
import numpy as np
import tensorflow as tf
import Config
from tools import _get_data, _get_conf, get_label, auc_score
import random
import time


class NFM(object):
    def __init__(self):
        self.total_emb, self.single_size, self.numerical_size, self.multi_size = _get_conf()
        print(self.single_size,self.numerical_size,self.multi_size)
        self.field_size = self.single_size + self.numerical_size + self.multi_size
        self.embedding_length = self.field_size * Config.embedding_size
        self._init_data()
        self._init_placeholder()
        self._init_Variable()
        self._init_Model()
        self.valid_batch = self._get_batch(self.valid, -1)
        self.valid_label = get_label(self.valid_batch[0], 2)
        self.valid_dict = {
            self.ph['single_index']: self.valid_batch[1],
            self.ph['numerical_index']: self.valid_batch[2],
            self.ph['value']: self.valid_batch[-1],
            self.ph['label']: self.valid_label,
            self.train_phase: False
        }
        if not Config.use_numerical_embedding:
            self.ph['numerical_value']: self.valid_batch[3]
        if Config.multi_features:
            for idx, s in enumerate(Config.multi_features):
                self.valid_dict[self.ph['multi_index_%s' % s]] = self.valid_batch[4]
                self.valid_dict[self.ph['multi_value_%s' % s]] = self.valid_batch[5]
        self.global_step = []
        self.global_train_auc = []
        self.global_valid_auc = []

        self._train()
        self._save_loss()

    def _init_data(self):
        self.train = _get_data(Config.train_save_file)
        self.valid = _get_data(Config.valid_save_file)
        self.test = _get_data(Config.test_save_file)

    def _get_batch(self, data, idx):
        start = time.time()
        if idx == -1:
            batch_data = data
        elif (idx + 1) * Config.batch_size <= len(data):
            batch_data = data[idx * Config.batch_size:(idx + 1) * Config.batch_size]
        else:
            batch_data = data[idx * Config.batch_size:]
        final_label = []
        final_single_index = []
        final_numerical_value = []
        final_numerical_index = []
        final_multi_sparse_index = []
        final_multi_sparse_value = []
        final_value = []
        for idx, line in enumerate(batch_data):
            line_index = []
            line_value = []
            line_numerical_value = []
            line_data = line.split(',')
            final_label.append(int(line_data[0]))
            if self.single_size:
                for i in range(1, 1 + self.single_size):
                    single_pair = line_data[i].split(':')
                    line_index.append(int(single_pair[0]))
                    line_value.append(float(single_pair[1]))
            final_single_index.append(line_index)
            line_index = []
            if self.single_size + self.numerical_size:
                for i in range(1 + self.single_size, 1 + self.single_size + self.numerical_size):
                    single_pair = line_data[i].split(':')
                    if not Config.use_numerical_embedding:
                        line_numerical_value.append(float(single_pair[1]))
                    if float(single_pair[1]) == 0:
                        line_index.append(int(9999))
                        line_value.append(float(1))
                    else:
                        line_index.append(int(single_pair[0]))
                        line_value.append(float(single_pair[1]))
            final_numerical_value.append(line_numerical_value)
            final_numerical_index.append(line_index)
            line_index = []
            total_length = 1 + self.single_size + self.numerical_size + self.multi_size
            if self.multi_size:
                for i in range(1 + self.single_size + self.numerical_size, total_length):
                    single_pair = line_data[i].split(':')
                    _multi = [int(x) for x in single_pair[0].split('|')]
                    line_index.append(_multi)
                    for v in _multi:
                        final_multi_sparse_index.append([idx, idx])
                        final_multi_sparse_value.append(v)
                    line_value.append(float(single_pair[1]))
            final_value.append(line_value)
        end = time.time()
        return [final_label, final_single_index, final_numerical_index, final_numerical_value, final_multi_sparse_index,
                final_multi_sparse_value, final_value]

    def _init_placeholder(self):
        self.ph = {}
        self.ph['label'] = tf.placeholder(dtype=tf.int8, shape=[None, 2])
        self.train_phase = tf.placeholder(tf.bool, name="train_phase")
        self.ph['value'] = tf.placeholder(dtype=tf.float32,
                                          shape=[None, self.single_size + self.numerical_size + self.multi_size])
        self.ph['single_index'] = tf.placeholder(dtype=tf.int32, shape=[None, self.single_size])
        self.ph['numerical_index'] = tf.placeholder(dtype=tf.int32, shape=[None, self.numerical_size])
        for s in Config.multi_features:
            self.ph['multi_index_%s' % s] = tf.placeholder(dtype=tf.int64, shape=[None, 2])
            self.ph['multi_value_%s' % s] = tf.placeholder(dtype=tf.int64, shape=[None])
        if not Config.use_numerical_embedding:
            self.ph['numerical_value'] = tf.placeholder(dtype=tf.float32, shape=[None, self.numerical_size])

    def _init_Variable(self):
        self.vr = {}
        self.vr['single_second_embedding'] = tf.get_variable(name='single_second_embedding',
                                                             shape=(10000, Config.embedding_size),
                                                             initializer=tf.glorot_uniform_initializer())
        self.vr['numerical_second_embedding'] = tf.get_variable(name='numerical_second_embedding',
                                                                shape=(10000, Config.embedding_size),
                                                                initializer=tf.glorot_uniform_initializer())
        for s in Config.multi_features:
            self.vr['multi_second_embedding_%s' % s] = tf.get_variable(name='multi_second_embedding_%s' % s,
                                                                       shape=(10000, Config.embedding_size),
                                                                       initializer=tf.glorot_uniform_initializer())

        self.vr['single_first_embedding'] = tf.get_variable(name='single_first_embedding',
                                                            shape=(10000, 1),
                                                            initializer=tf.glorot_uniform_initializer())
        self.vr['numerical_first_embedding'] = tf.get_variable(name='numerical_first_embedding',
                                                               shape=(10000, 1),
                                                               initializer=tf.glorot_uniform_initializer())
        for s in Config.multi_features:
            self.vr['multi_first_embedding_%s' % s] = tf.get_variable(name='multi_first_embedding_%s' % s,
                                                                      shape=(10000, 1),
                                                                      initializer=tf.glorot_uniform_initializer())
        dnn_net = [Config.embedding_size] + Config.dnn_net_size
        for i in range(len(Config.dnn_net_size)):
            self.vr['W_%d' % i] = tf.get_variable(name='W_%d' % i, shape=[dnn_net[i], dnn_net[i + 1]],
                                                  initializer=tf.glorot_uniform_initializer())
            self.vr['b_%d' % i] = tf.get_variable(name='b_%d' % i, shape=[dnn_net[i + 1]],
                                                  initializer=tf.zeros_initializer())
            # output

    def _init_Model(self):
        # first embedding
        if self.single_size:
            first_single_result = tf.reshape(tf.nn.embedding_lookup(self.vr['single_first_embedding'],
                                                                    self.ph['single_index']),
                                             shape=[-1, self.single_size]
                                             )
        if self.numerical_size:
            first_numerical_result = tf.reshape(tf.nn.embedding_lookup(self.vr['numerical_first_embedding'],
                                                                       self.ph['numerical_index']),
                                                shape=[-1, self.numerical_size]
                                                )
        first_multi_result = []
        if Config.multi_features:
            for s in Config.multi_features:
                temp_multi_result = tf.nn.embedding_lookup_sparse(self.vr['multi_first_embedding_%s' % s],
                                                                  tf.SparseTensor(indices=self.ph['multi_index_%s' % s],
                                                                                  values=self.ph['multi_value_%s' % s],
                                                                                  dense_shape=(Config.batch_size,
                                                                                               Config.embedding_size)),
                                                                  None,
                                                                  combiner="sum"
                                                                  )
                first_multi_result.append(temp_multi_result)
                first_multi_result = tf.concat(first_multi_result, axis=1)
        to_concat = []
        if self.single_size:
            to_concat.append(first_single_result)
        if self.numerical_size:
            to_concat.append(first_numerical_result)
        first_embedding_output = tf.concat(to_concat,axis=1)
        y_first_order = tf.multiply(first_embedding_output, self.ph['value'])

        # second embedding
        if self.single_size:
            second_single_result = tf.reshape(tf.nn.embedding_lookup(self.vr['single_second_embedding'],
                                                                     self.ph['single_index']),
                                              shape=[-1, Config.embedding_size * self.single_size]
                                              )
        if self.numerical_size:
            second_numerical_result = tf.reshape(tf.nn.embedding_lookup(self.vr['numerical_second_embedding'],
                                                                        self.ph['numerical_index']),
                                                 shape=[-1, Config.embedding_size * self.numerical_size]
                                                 )
        if Config.multi_features:
            second_multi_result = []
            for s in Config.multi_features:
                temp_multi_result = tf.nn.embedding_lookup_sparse(self.vr['multi_second_embedding_%s' % s],
                                                                  tf.SparseTensor(indices=self.ph['multi_index_%s' % s],
                                                                                  values=self.ph['multi_value_%s' % s],
                                                                                  dense_shape=(Config.batch_size,
                                                                                               Config.embedding_size)),
                                                                  None,
                                                                  combiner="sum"
                                                                  )
                second_multi_result.append(temp_multi_result)
                second_multi_result = tf.concat(second_multi_result, axis=1)
        tobe_concat = []
        if self.single_size:
            tobe_concat.append(second_single_result)
        if self.numerical_size:
            tobe_concat.append(second_numerical_result)
        if self.multi_size:
            tobe_concat.append(second_multi_result)
        # DNN input
        self.DNN_input = tf.concat(tobe_concat,axis=1)
        self.middle_fm_input = self.DNN_input
        # second output
        second_FM_input = tf.reshape(self.middle_fm_input,
                                     shape=[-1, self.single_size + self.numerical_size + self.multi_size,
                                            Config.embedding_size])

        summed_features_emb = tf.reduce_sum(second_FM_input, 1)
        summed_features_emb_square = tf.square(summed_features_emb)
        squared_features_emb = tf.square(second_FM_input)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)
        y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)

        self.shape = tf.shape(self.DNN_input)
        # NFM BI
        self.DNN_input = tf.reshape(self.DNN_input, shape=[-1, self.single_size + self.multi_size + self.numerical_size,
                                                           Config.embedding_size])
        all_embedding = []
        for i in range(1, self.field_size):
            now_embedding = self.DNN_input[:,-i]
            for j in range(self.field_size - i):
                temp_embedding = self.DNN_input[:,j]
                temp_result = tf.multiply(now_embedding, temp_embedding)
                all_embedding.append(temp_result)
        self.DNN_input = tf.reduce_sum(all_embedding, axis=0)




        dnn_output = self.DNN_input
        # DNN output
        for i in range(len(Config.dnn_net_size)):
            self.DNN_input = tf.add(tf.matmul(self.DNN_input, self.vr['W_%d' % i]), self.vr['b_%d' % i])
            self.DNN_input = tf.layers.batch_normalization(self.DNN_input, training=self.train_phase)
            dnn_output = tf.nn.relu(self.DNN_input)



        # final output
        output_length = 0
        to_concat = []
        if Config.FM_layer:
            to_concat.append(y_first_order)
            to_concat.append(y_second_order)
            output_length += self.field_size + Config.embedding_size
        if Config.DNN_layer:
            to_concat.append(dnn_output)
            output_length += Config.dnn_net_size[-1]

        output = tf.concat(to_concat, axis=1)

        self.vr['final_w'] = tf.get_variable(name='final_w', shape=[output_length, 2],
                                             initializer=tf.glorot_uniform_initializer())
        self.vr['final_b'] = tf.get_variable(name='final_b', shape=[2],
                                             initializer=tf.zeros_initializer())
        final_logits = tf.add(tf.matmul(output, self.vr['final_w']), self.vr['final_b'])
        self.softmax_output = tf.nn.softmax(final_logits)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.ph['label'], logits=final_logits))
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=Config.learning_rate).minimize(self.loss)

    def _train(self):
        print('....')
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            allDataLength = len(self.train)
            global_step = 0
            print('total step:%d' % (Config.epochs * (int(allDataLength / Config.batch_size) + 1)))
            for i in range(Config.epochs):
                num_batchs = int(allDataLength / Config.batch_size) + 1
                for j in range(num_batchs):
                    global_step += 1
                    now_batch = self._get_batch(self.train, j)
                    start = time.time()
                    batch_dict = {
                        self.ph['single_index']: now_batch[1],
                        self.ph['numerical_index']: now_batch[2],
                        self.ph['value']: now_batch[-1],
                        self.ph['label']: get_label(now_batch[0], 2),
                        self.train_phase: True
                    }
                    if not Config.use_numerical_embedding:
                        self.ph['numerical_value']: self.valid_batch[3]
                    if Config.multi_features:
                        for idx, s in enumerate(Config.multi_features):
                            batch_dict[self.ph['multi_index_%s' % s]] = now_batch[4]
                            batch_dict[self.ph['multi_value_%s' % s]] = now_batch[5]
                    end = time.time()
                    start = time.time()
                    _out, _loss, _ = self.sess.run((self.softmax_output, self.loss, self.optimizer),
                                                   feed_dict=batch_dict)
                    end = time.time()

                    if global_step % 10 == 0:
                        __out, __loss, __ = self.sess.run((self.softmax_output, self.loss, self.optimizer),
                                                          feed_dict=self.valid_dict)
                        self.global_step.append(global_step)
                        self.global_train_auc.append(_loss)
                        self.global_valid_auc.append(__loss)
                        print('step:', global_step, 'train loss:', _loss, 'valid loss:', __loss, 'valid_auc:',
                              auc_score(__out, get_label(self.valid_batch[0], 2), 2))

    def _save_loss(self):
        loss_result = pd.DataFrame({
            'step': self.global_step,
            'train_auc': self.global_train_auc,
            'valid_auc': self.global_valid_auc
        })
        loss_result.to_csv('NFM_loss.csv', index=False)


if __name__ == '__main__':
    nfm = NFM()
