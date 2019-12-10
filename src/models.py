from src.layers import *
from src.metrics import *
from src.Final_value import residue_type_dict
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg    #字典key是否是name   logging的一员
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()      #获取类名
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None, Path=''):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        print('self.name: '+self.name)
        save_path = saver.save(sess, Path+'model_'+str(self.name)+'.ckpt')
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None, Path=''):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = Path+'model_'+str(self.name)+'.ckpt'
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):    #设置variable_scope
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])          # python中类的实例对象可以被调用，此处调用了对象layer
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)


class GCN(Model):
    def __init__(self, placeholders, input_dim, hidden1,learning_rate, weight_decay,batch_size, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']      #input是特征
        self.hidden1 = hidden1
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim =len(residue_type_dict)
        self.placeholders = placeholders
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        self.loss=0
        self.loss_list=[]
        for layer in self.layers:
            for var in layer.vars.values():
                self.loss_list.append(self.weight_decay * tf.nn.l2_loss(var))
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        # for var in self.layers[0].vars.values():
        #     self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        msce=[]
        for i in range(batch_size):

            msce.append(masked_softmax_cross_entropy(self.outputs[i], self.placeholders['labels'][i],
                                         self.placeholders['labels_mask'][i]))

        self.loss += tf.reduce_mean(msce)

    def _accuracy(self):
        acc=[]
        for i in range(batch_size):
            acc.append( masked_accuracy(self.outputs[i], self.placeholders['labels'][i],
                                        self.placeholders['labels_mask'][i]))
        self.accuracy=tf.reduce_mean(acc)

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,  # 输入维度
                                                 output_dim=self.hidden1,    #隐藏层节点数
                                                 placeholders=self.placeholders,  #占位符
                                                 act=tf.nn.relu,            #激活函数
                                                 dropout=False,               #进行dropout
                                                 sparse_inputs=True,         #输入是稀疏的
                                                 logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=self.hidden1,  # 输入维度
                                            output_dim=64,  # 隐藏层节点数
                                            placeholders=self.placeholders,  # 占位符
                                            act=tf.nn.relu,  # 激活函数
                                            dropout=True,  # 进行dropout
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=64,  # 输入维度
                                            output_dim=128,  # 隐藏层节点数
                                            placeholders=self.placeholders,  # 占位符
                                            act=tf.nn.relu,  # 激活函数
                                            dropout=True,  # 进行dropout
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=128,  # 输入维度
                                            output_dim=128,  # 隐藏层节点数
                                            placeholders=self.placeholders,  # 占位符
                                            act=tf.nn.relu,  # 激活函数
                                            dropout=True,  # 进行dropout
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=128,  # 输入维度
                                            output_dim=128,  # 隐藏层节点数
                                            placeholders=self.placeholders,  # 占位符
                                            act=tf.nn.relu,  # 激活函数
                                            dropout=True,  # 进行dropout
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=128,  # 输入维度
                                            output_dim=256,  # 隐藏层节点数
                                            placeholders=self.placeholders,  # 占位符
                                            act=tf.nn.relu,  # 激活函数
                                            dropout=True,  # 进行dropout
                                            logging=self.logging))
        # self.layers.append(GraphConvolution(input_dim=256,  # 输入维度
        #                                     output_dim=256,  # 隐藏层节点数
        #                                     placeholders=self.placeholders,  # 占位符
        #                                     act=tf.nn.relu,  # 激活函数
        #                                     dropout=True,  # 进行dropout
        #                                     logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=256,  # 输入维度
                                            output_dim=256,  # 隐藏层节点数
                                            placeholders=self.placeholders,  # 占位符
                                            act=tf.nn.relu,  # 激活函数
                                            dropout=True,  # 进行dropout
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=256,
                                                 output_dim=self.output_dim,
                                                 placeholders=self.placeholders,
                                                 act=lambda x: x,     #匿名函数  返回x
                                                 dropout=False,
                                                 logging=self.logging))
    # def _build(self):
    #
    #     self.layers.append(GraphConvolution(input_dim=self.input_dim,   #输入维度
    #                                         output_dim=self.hidden1,    #隐藏层节点数
    #                                         placeholders=self.placeholders,  #占位符
    #                                         act=tf.nn.relu,            #激活函数
    #                                         dropout=False,               #进行dropout
    #                                         sparse_inputs=True,         #输入是稀疏的
    #                                         logging=self.logging))
    #     self.layers.append(GraphConvolution(input_dim=self.hidden1,  # 输入维度
    #                                         output_dim=64,  # 隐藏层节点数
    #                                         placeholders=self.placeholders,  # 占位符
    #                                         act=tf.nn.relu,  # 激活函数
    #                                         dropout=True,  # 进行dropout
    #                                         logging=self.logging))
    #     self.layers.append(GraphConvolution(input_dim=64,  # 输入维度
    #                                         output_dim=64,  # 隐藏层节点数
    #                                         placeholders=self.placeholders,  # 占位符
    #                                         act=tf.nn.relu,  # 激活函数
    #                                         dropout=True,  # 进行dropout
    #                                         logging=self.logging))
    #     self.layers.append(GraphConvolution(input_dim=64,  # 输入维度
    #                                         output_dim=128,  # 隐藏层节点数
    #                                         placeholders=self.placeholders,  # 占位符
    #                                         act=tf.nn.relu,  # 激活函数
    #                                         dropout=True,  # 进行dropout
    #                                         logging=self.logging))
    #     self.layers.append(GraphConvolution(input_dim=128,  # 输入维度
    #                                         output_dim=128,  # 隐藏层节点数
    #                                         placeholders=self.placeholders,  # 占位符
    #                                         act=tf.nn.relu,  # 激活函数
    #                                         dropout=True,  # 进行dropout
    #                                         logging=self.logging))
    #     self.layers.append(GraphConvolution(input_dim=128,  # 输入维度
    #                                         output_dim=128,  # 隐藏层节点数
    #                                         placeholders=self.placeholders,  # 占位符
    #                                         act=tf.nn.relu,  # 激活函数
    #                                         dropout=True,  # 进行dropout
    #                                         logging=self.logging))
    #     self.layers.append(GraphConvolution(input_dim=128,  # 输入维度
    #                                         output_dim=256,  # 隐藏层节点数
    #                                         placeholders=self.placeholders,  # 占位符
    #                                         act=tf.nn.relu,  # 激活函数
    #                                         dropout=True,  # 进行dropout
    #                                         logging=self.logging))
    #     self.layers.append(GraphConvolution(input_dim=256,  # 输入维度
    #                                         output_dim=256,  # 隐藏层节点数
    #                                         placeholders=self.placeholders,  # 占位符
    #                                         act=tf.nn.relu,  # 激活函数
    #                                         dropout=True,  # 进行dropout
    #                                         logging=self.logging))
    #     self.layers.append(GraphConvolution(input_dim=256,  # 输入维度
    #                                         output_dim=256,  # 隐藏层节点数
    #                                         placeholders=self.placeholders,  # 占位符
    #                                         act=tf.nn.relu,  # 激活函数
    #                                         dropout=True,  # 进行dropout
    #                                         logging=self.logging))
    #     self.layers.append(GraphConvolution(input_dim=256,
    #                                         output_dim=self.output_dim,
    #                                         placeholders=self.placeholders,
    #                                         act=lambda x: x,     #匿名函数  返回x
    #                                         dropout=False,
    #                                         logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)




class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)