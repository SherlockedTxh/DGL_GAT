import numpy as np
from src.utils import *
from src.models import GCN
import time
import scipy.sparse as sp
from src.Final_value import residue_type_dict
import tensorflow as tf


class Protein_GCN(object):

    def __init__(self, graph_dict,train_dict, learning_rate=0.001, epochs=200,  batch_size=4,
                 hidden1=64, dropout=0.5, weight_decay=5e-4, early_stopping=100,#权值衰减
                 max_degree=3, clf_ratio=0.8,Path_output=''):
        """
                        learning_rate: Initial learning rate
                        epochs: Number of epochs to train
                        hidden1: Number of units in hidden layer 1
                        dropout: Dropout rate (1 - keep probability)
                        weight_decay: Weight for L2 loss on embedding matrix
                        early_stopping: Tolerance for early stopping (# of epochs)
                        max_degree: Maximum Chebyshev polynomial degree
        """
        #self.labels       矩阵，每个节点对应所打的标签
        #self.label_dict   字典，一个label对应自己的编号
        self.graph_dict = graph_dict             #字典：graph_name(network_name)  :   Graph
        self.train_dict=train_dict               #字典：number    :   graph_name(network_name)
        self.clf_ratio = clf_ratio           #训练比例
        self.learning_rate = learning_rate
        self.epochs = epochs                #迭代次数
        self.hidden1 = hidden1
        self.dropout = dropout
        self.weight_decay = weight_decay    #权重衰减
        self.early_stopping = early_stopping
        self.max_degree = max_degree
        self.batch_size=batch_size
        # print(self.clf_ratio)
        # print(self.learning_rate)
        # print(self.epochs)
        # print(self.hidden1)
        # print(self.dropout)
        self.preprocess_data()#self.features={}   self.support={}   self.labels={}
        self.build_train_val_test()
        self.num_batches=int(self.training_size/self.batch_size)
        self.build_placeholders()
        # Create model
        self.model = GCN(
            placeholders=self.placeholders, input_dim=self.input_dim, hidden1=self.hidden1,
            learning_rate=self.learning_rate,weight_decay=self.weight_decay, batch_size=batch_size,logging=True)
        #占位符    输入维度     隐藏层 数   权值衰减
        # Initialize session
        self.sess = tf.Session()
        # Init variables
        self.sess.run(tf.global_variables_initializer())
        print('len(train_mask): ' + str(len(self.train_mask)))
        print('len(test_mask): ' + str(len(self.test_mask)))
        # Train model
        for epoch in range(self.epochs):

            t = time.time()
            for i in range(self.num_batches):
                network_name=[]
                now_train_mask=[]
                network_number=[]
                for ii in range(batch_size):
                    network_name.append(self.number_networkname_dic[self.train_mask[i*self.batch_size+ii]])
                    network_number.append(self.train_mask[i*self.batch_size+ii])
                    now_train_mask.append(np.ones(graph_dict[network_name[ii]].G.number_of_nodes()))
                # Construct feed dictionary
                feed_dict = self.construct_feed_dict(network_number,now_train_mask)
                # print(feed_dict)
                # print((self.train_mask))
                feed_dict.update({self.placeholders['dropout']: self.dropout})
                # Training step

                outs = self.sess.run([self.model.opt_op, self.model.loss, self.model.accuracy], feed_dict=feed_dict)
                cost_val = []
                acc_val=[]
                #Validation
                # print(len(self.val_mask))

                for j in range(int(len(self.val_mask) / self.batch_size)):
                    validation_name_list = []
                    now_t = []
                    nn = []
                    for ii in range(batch_size):
                        validation_name_list.append(self.number_networkname_dic[self.val_mask[j * self.batch_size + ii]])
                        nn.append(self.val_mask[j * self.batch_size + ii])
                        now_t.append(np.ones(graph_dict[validation_name_list[ii]].G.number_of_nodes()))
                    cost, acc, duration = self.evaluate(nn, now_t)
                    cost_val.append(cost)
                    acc_val.append(acc)

                print('Epoch: ' + str(epoch) + ' train.size: ' + str(
                    self.training_size) + ' train_loss: ' + str(outs[1]) + ' train_acc: ' + str(outs[2])
                      + ' validation_cost_mean: ' + str(np.mean(cost_val)) + ' validation_acc_mean: ' + str(
                    np.mean(acc_val)))





            # Print results
            # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            #       "train_acc=", "{:.5f}".format(
            #     outs[2]), "val_loss=", "{:.5f}".format(cost),
            #     "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            # if epoch > self.early_stopping and cost_val[-1] > np.mean(cost_val[-(self.early_stopping+1):-1]):#mean 求均值
            #     print("Early stopping...")
            #     break
        # print("Optimization Finished!")
        #
        # # Testing
        # test_cost, test_acc,test_cost_list, test_mask,test_duration = self.evaluate(self.test_mask)
        # print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        #       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

        #Saving
        # if Path_output.strip()!='':
        #     self.Savemodel(Path_output)
        # else:
        #     print('model dose not save')

    def Savemodel(self,Path_output=''):
        self.model.save(self.sess,Path_output)
    # Define model evaluation function

    def evaluate(self, networkname ,mask):  #########################
        t_test = time.time()        #
        feed_dict_val = self.construct_feed_dict(networkname,mask)
        outs_val = self.sess.run(
            [self.model.loss, self.model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1],(time.time() - t_test)

    def build_placeholders(self): #000000000000000000000000000000000000000
        self.placeholders = {
            'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(_)) for _ in range(self.batch_size)],       #稀疏矩阵占位符
            'features': [tf.sparse_placeholder(tf.float32, name='features_{}'.format(_)) for _ in range(self.batch_size)],
            'labels': [tf.placeholder(tf.float32, shape=(None,len(self.label_dict)), name='labels_{}'.format(_)) for _ in range(self.batch_size)],
            'labels_mask': [tf.placeholder(tf.int32, name='labels_mask_{}'.format(_)) for _ in range(self.batch_size)],
            'dropout': tf.placeholder_with_default(0., shape=()),
            # helper variable for sparse dropout
            'num_features_nonzero': [tf.placeholder(tf.int32, name='num_features_nonzero_{}'.format(_)) for _ in range(self.batch_size)]
        }

    def build_label(self,graph):#0000000000000000000000000
        print(graph)
        g = self.graph_dict[graph].G
        look_up = self.graph_dict[graph].look_up_dict
        labels = []
        label_id = 0
        for node in g.nodes():
            labels.append((node, g.nodes[node]['label']))   #node label  字典
        lll = np.zeros((len(labels), len(residue_type_dict)))    #创建矩阵
        self.label_dict = residue_type_dict
        for node, l in labels:
            node_id = look_up[node]               #填充矩阵
            for ll in l:
                l_id = residue_type_dict[ll]
                lll[node_id][l_id] = 1
        self.labels.append(lll)

    def build_train_val_test(self):##0000000000000000000000000000000000
        """
            build train_mask test_mask val_mask
        """
        train_precent = self.clf_ratio
        self.training_size = int(train_precent * len(self.train_dict))
        self.validation_size = int(0.5*(1-train_precent)*len(self.train_dict))+1
        self.test_size = len(self.train_dict)-self.training_size-self.validation_size
        # print('train_size '+str(self.training_size))
        # print('validation_size '+str(self.validation_size))
        # print('test_size '+str(self.test_size))
        state = np.random.get_state()
        np.random.seed(0)
        shuffle_indices = np.random.permutation(      #随机排列序列
            np.arange(len(self.train_dict)))
        # print('shuffle_indices')
        # print(shuffle_indices)
        np.random.set_state(state)

        # print('self.train_dict: '+str(len(self.train_dict)))
        # print('self.training_size: '+str(self.training_size))
        def sample_mask(begin, end):
            mask = np.zeros(len(self.train_dict))
            for i in range(begin, end):
                mask[shuffle_indices[i]] = 1
            return mask

        # nodes_num = len(self.labels)
        # self.train_mask = sample_mask('train', nodes_num)
        # self.val_mask = sample_mask('valid', nodes_num)
        # self.test_mask = sample_mask('test', nodes_num)
        self.train_mask = shuffle_indices[0:self.training_size]
        self.val_mask = shuffle_indices[self.training_size: self.training_size+self.validation_size]
        self.test_mask = shuffle_indices[self.training_size+self.validation_size: len(self.train_dict)]

    def preprocess_data(self):##0000000000000000000000000
        """
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
            y_train, y_val, y_test can merge to y
        """
        #look_up_dict   节点  序号 字典
        #look_back_list  节点列表
        self.features=[]
        self.support=[]
        self.labels=[]
        self.number_networkname_dic={}
        graph_dict = self.graph_dict
        number=0
        for graph in graph_dict.keys():
            self.number_networkname_dic[number]=graph
            number+=1
            look_back = graph_dict[graph].look_back_list
            features = np.vstack([graph_dict[graph].G.nodes[look_back[i]]['feature']
                                       for i in range(graph_dict[graph].G.number_of_nodes())])
            #特征
            features=preprocess_features(features)
            self.features.append(features)     #得到的特征是元组：(坐标，值，类型)
            self.input_dim=features[2][1]
            self.build_label(graph)        #建立一个node  *   label的矩阵    self.labels[node_id][l_id] = 1 代表此节点有标签l_id
            adj = nx.adjacency_matrix(graph_dict[graph].G)  # the type of graph
            self.support.append(preprocess_adj(adj))     #邻接矩阵规范化，变成元组

    def construct_feed_dict(self, network_number,now_train_mask):#00000000000000000000000000000
        """Construct feed dictionary."""
        from collections import OrderedDict
        #feed_dict = dict()
        feed_dict = OrderedDict()
        # labels=[]
        # now_train_mask_list=[]
        # features_list=[]
        # surpport_list=[]
        # nfn_list=[]
        # labels.append(self.labels[network_name[i]])
        # features_list.append(self.features[network_name[i]])
        # surpport_list.append(self.support[network_name[i]])
        # nfn_list.append(self.features[network_name[i]][1].shape)
        feed_dict.update({self.placeholders['labels'][i]: self.labels[network_number[i]] for i in range(self.batch_size) })    #
        feed_dict.update({self.placeholders['labels_mask'][i]: now_train_mask[i] for i in range(self.batch_size)})

        feed_dict.update(
            {self.placeholders['num_features_nonzero'][i]: self.features[network_number[i]][1].shape[0] for i in range(self.batch_size)})
        feed_dict.update(
            {self.placeholders['support'][i]: self.support[network_number[i]] for i in range(self.batch_size) })
        feed_dict.update({self.placeholders['features'][i]: self.features[network_number[i]] for i in range(self.batch_size)})

        return feed_dict
