import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))

# reference: https://blog.csdn.net/u011974639/article/details/78290448

class DenseNet:
    def __init__(self, args):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: Class, that have all required data sets
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            use_bottleneck: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
        self.args = args
        self.global_step = tf.train.get_or_create_global_step()
        self.images = tf.placeholder(tf.float32, shape = [args.batch_size, 180, 180, 1],name='input_images')
        self.labels = tf.placeholder(tf.float32, shape = [args.batch_size, args.num_classes],name='labels')
        # self.learning_rate = args.learning_rate
        self.learning_rate = tf.convert_to_tensor(args.learning_rate, dtype=tf.float32)

        self.is_training = args.is_training

        self.n_classes = args.num_classes
        self.depth = args.depth
        self.growth_rate = args.growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = self.growth_rate * 2
        self.total_blocks = args.total_blocks
        # pool_layers = self.total_blocks + 1
        self.layers_per_block = (self.depth - (self.total_blocks + 1)) // self.total_blocks
        self.use_bottleneck = args.use_bottleneck
        # compression rate at the transition layers
        self.reduction = args.reduction

        if not self.use_bottleneck:
            print("\nBuild %s model with %d blocks, "
                  "%d composite layers each." % (
                      "DenseNet", self.total_blocks, self.layers_per_block))
        if self.use_bottleneck:
            self.layers_per_block = self.layers_per_block // 2
            print("\nBuild %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      "bottleneck DenseNet ", self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = args.keep_prob
        self.weight_decay = args.decay_rate
        self.nesterov_momentum = args.nesterov_momentum
        
        self.dataset_name = args.dataset_name
        self.batches_step = 0

        self._build_graph()
        # self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.use_bottleneck:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.use_bottleneck:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output

    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier(
            [features_total, self.n_classes], name='W')
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_random_normal(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def weight_variable_random_normal(self, shape, name):
        # shape = [kernel_size, kernel_size, in_features, out_features]
        n = shape[0] * shape[0] * shape[3]
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block)
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_classes(output)
        prediction = tf.nn.softmax(logits)

        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(
            cross_entropy + l2_loss * self.weight_decay, global_step=self.global_step)

        self.learning_rate = tf.maximum(1e-8,tf.train.exponential_decay(self.args.learning_rate, self.global_step, self.args.decay_step, self.args.decay_rate, staircase=True))
        self.lr = tf.to_float(self.learning_rate, name='ToFloat')

        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))