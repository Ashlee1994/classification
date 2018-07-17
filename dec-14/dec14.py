import os,time
import tensorflow as tf
import numpy as np

class dec14:
    def __init__(self,args):
        self.args = args
        self.variables_deepem()
        self.build_model()

    def variables_deepem(self):
        '''
        self.data_dict = {
            # filter = [kernelsize, kernelsize, last_layser_feature_map_num, feature_map_num]
            # biases = [feature_map_num]
            'conv1':   {'filter': [35,35,1, 32], 'biases': [32]},
            'conv2':   {'filter': [25,25,32,64], 'biases':[64]},
            'conv3':   {'filter': [15,15,64, 128], 'biases':[128]},
            'conv4_1': {'filter': [15,15,128,256], 'biases':[256]},
            'conv4_2': {'filter': [7,7,256,256], 'biases':[256]},
            'conv4_3': {'filter': [7,7,256,256], 'biases':[256]},
            'fc6': {'filter': [], 'biases':[2048]},    # 4096,
            'fc7': {'filter': [2048, 2048], 'biases':[2048]},    # 4096,
            'fc8': {'filter': [2048,    self.args.n_classes], 'biases':[self.args.n_classes]}   # 1
        }
        '''
        # train acc:  0.829375
        # test acc:  0.825833
        # learning_rate: 0.00001
        self.data_dict = {
            # filter = [kernelsize, kernelsize, last_layser_feature_map_num, feature_map_num]
            # biases = [feature_map_num]
            'conv1':   {'filter': [15,15,1, 32], 'biases': [32]},
            'conv2':   {'filter': [10,10,32,64], 'biases': [64]},
            'conv3':   {'filter': [6,6,64, 128], 'biases':[128]},
            'conv4_1': {'filter': [3,3,128,256], 'biases':[256]},
            'conv4_2': {'filter': [3,3,256,256], 'biases':[256]},
            'conv4_3': {'filter': [3,3,256,256], 'biases':[256]},
            'fc6': {'filter': [], 'biases':[2048]},    # 4096,
            'fc7': {'filter': [2048, 2048], 'biases':[2048]},    # 4096,
            'fc8': {'filter': [2048,    self.args.n_classes], 'biases':[self.args.n_classes]}   # 1
        }

    def build_model(self):
        self.X = tf.placeholder(tf.float32, shape = [None, 180, 180, 1])
        self.Y = tf.placeholder(tf.float32, shape = [None,self.args.n_classes])
        self.global_step = tf.Variable(0, name='global_step',trainable=False)

        self.conv1 = self.conv_layer(self.X,           "conv1")
        self.pool1 = self.avg_pool(self.conv1,         'pool1')
        print("shape of conv1: ",             self.conv1.shape)
        print("shape of pool1:   ",           self.pool1.shape)

        self.conv2 = self.conv_layer(self.pool1,       "conv2")
        self.pool2 = self.avg_pool(self.conv2,         'pool2')
        # self.pool2 = self.avg_pool(self.conv2_2,       'pool2')
        print("shape of conv2: ",             self.conv2.shape)
        print("shape of pool2:   ",           self.pool2.shape)

        self.conv3 = self.conv_layer(self.pool2,       "conv3")
        self.pool3 = self.avg_pool(self.conv3,         'pool3')
        # self.pool3 = self.avg_pool(self.conv3_4,       'pool3')
        print("shape of conv3: ",             self.conv3.shape)
        print("shape of pool3:   ",           self.pool3.shape)

        self.conv4_1 = self.conv_layer(self.pool3,   "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.avg_pool(self.conv4_3,       'pool4')
        # self.pool4 = self.avg_pool(self.conv4_4,       'pool4')
        print("shape of conv4_1: ",         self.conv4_1.shape)
        print("shape of conv4_2: ",         self.conv4_2.shape)
        print("shape of conv4_3: ",         self.conv4_3.shape)
        print("shape of pool4:   ",           self.pool4.shape)

        self.fc6 = self.fc_layer(self.pool4, "fc6")

        # self.relu6 = tf.nn.relu(self.fc6)
        # self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.fc6, "fc7")

        # dropout
        if self.args.is_training and self.args.dropout:
            self.fc7 = tf.nn.dropout(self.fc7, self.args.dropout_rate)
        # self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.fc7, "fc8")

        print("shape of fc6:     ", self.fc6.shape)
        print("shape of fc7:     ", self.fc7.shape)
        print("shape of fc8:     ", self.fc8.shape)

        self.logits = self.fc8
        self.pred = tf.nn.softmax(self.logits, name="prob")
        self.accuracy =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred,1),tf.argmax(self.Y, 1)), tf.float32))

        if not self.args.is_training:
            return

        # regularization
        # self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.Y - self.pred)))
        # self.loss = tf.reduce_sum(tf.square(self.Y - self.pred))
        if not self.args.regularization:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels = self.Y))
        else:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels = self.Y)) + \
                    tf.contrib.layers.l2_regularizer(self.args.reg_rate)(self.get_fc_weight('fc6')) + \
                    tf.contrib.layers.l2_regularizer(self.args.reg_rate)(self.get_fc_weight('fc7')) + \
                    tf.contrib.layers.l2_regularizer(self.args.reg_rate)(self.get_fc_weight('fc8'))
                  
        self.lr = tf.maximum(1e-8,tf.train.exponential_decay(self.args.learning_rate, self.global_step, self.args.decay_step, self.args.decay_rate, staircase=True))
        # self.lr = self.args.learning_rate
        # self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        # self.optimizer = tf.train.MomentumOptimizer(self.lr,0.9).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            if name == "fc6":
                self.data_dict['fc6']['filter'] = [dim, 2048]
            
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_conv_filter(self, name):
        return tf.Variable(tf.truncated_normal(self.data_dict[name]['filter'], stddev = 0.02), name="filter")
        # return tf.Variable(self.data_dict[name]['filter'], name="filter")

    def get_bias(self, name):
        # b1 = tf.Variable(tf.zeros([self.args.FL_feature_map]))
        return tf.Variable(tf.zeros(self.data_dict[name]['biases']), name="biases")

    def get_fc_weight(self, name):
        #return tf.Variable(tf.zeros(self.data_dict[name]['filter']), name="weights")
        return tf.Variable(tf.truncated_normal(self.data_dict[name]['filter'], stddev = 0.02), name="weights")
        # return tf.Variable(self.data_dict[name]['filter'], name="weights")
