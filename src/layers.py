from src.inits import *
import tensorflow as tf
from src.Final_value import batch_size
flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)    #随机生成一个tensor
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)# floor 向下取整     case数据转换格式
    pre_out = tf.sparse_retain(x, dropout_mask)       #为x  中的非空元素    设置对应位的空值
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)     #参数可视化，直方图
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])      #参数可视化直方图





class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.input_dim=input_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights'] = glorot([input_dim, output_dim],
                                                        name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:          #参数可视化
            self._log_vars()

    def _call(self, inputs): ######000000000000000000000000
        x = inputs

        y=[]
        # dropout
        if self.sparse_inputs:                 #sparse_inputs   代表输入 inputs 是否是稀疏矩阵
            for i in range(batch_size):
                y.append(dot(x[i],tf.eye(self.input_dim),sparse=self.sparse_inputs))
                y[i] = tf.nn.dropout(y[i], 1 - self.dropout)
        else:
            for i in range(batch_size):
                y.append(tf.nn.dropout(x[i], 1-self.dropout))
        # convolve
        supports = list()
        pre_sup=[]
        for i in range(batch_size):                   #将一个网络分成若干子网络，每个子网络有自己的权值矩阵
            if not self.featureless:
                pre_sup.append(dot(y[i], self.vars['weights'],
                              sparse=False))
            else:
                pre_sup.append(self.vars['weights'])
            support = dot(self.support[i], pre_sup[i], sparse=True)
            supports.append(support)
        output = supports

        # bias
        if self.bias:
            for i in range(batch_size):
                output[i] += self.vars['bias']

        for i in range(batch_size):
            output[i] = tf.layers.batch_normalization(output[i], training=True)
            output[i]=self.act(output[i])
        return output

