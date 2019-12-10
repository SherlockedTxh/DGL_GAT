
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import networkx as nx
# class dd(object):
#     def __init__(self):
#         self.cc=[6]
#
#     def ddd(self):
#         ccc=self.cc
#         ccc.append(9)
#         print(self.cc)
#     def get_class_name(self):
#         print('dasdasdasd:'+self.__class__.__name__.lower())
# #sad=dd()
# #sad.get_class_name()

#
#     if isinstance(sparse_mx, list):   #是否是list类型
#         for i in range(len(sparse_mx)):
#             sparse_mx[i] = to_tuple(sparse_mx[i])
#     else:
#         sparse_mx = to_tuple(sparse_mx)
#
#     return sparse_mx
#
# aa=np.vstack([[1,2,3],[4,5,6]])
# l=[[1.,2.,3.,4.],[5.,6.,7.,8.]]
#
# cc=np.vstack([l[i] for i in range(2)])

# s=np.array(cc.sum(1))
# dd=np.power(s,-1).flatten()
# uashd=np.isinf(dd)
log=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])

# y_=tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])
# cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=log, labels=y_))#dont forget tf.reduce_sum()!!
# sess = tf.Session()
# res=sess.run(cross_entropy2)
# print(res)

# row  = np.array([0, 3, 1, 0 ,2])
# col  = np.array([0, 3, 1, 2, 2])
# data = np.array([4, 5, 7, 9, 8])
# cc=sp.coo_matrix((data,(row,col)),shape=(4,4))
# print(cc.toarray())
#
# print(dd.toarray())
#
#
#
# dic={'ddd':123}
#print(dic.get('ddd'))
#print(dic.get('sd'))
#print(c00_max.transpose().toarray())
#print(c00_max.transpose().dot(c00_max).toarray())
#print(np.dot(c00_max.transpose(),c00_max))
#print(c00_max.row)
#print(c00_max.col)
#print(c00_max.data)
#coords = np.vstack((c00_max.row, c00_max.col)).transpose()
#print(coords)



#seed=0
#random_tensor=0.5
#num_features_nonzero=tf.placeholder(tf.int32)
#tt=tf.random_uniform(num_features_nonzero,seed=seed)
#random_tensor=random_tensor+tt

#with tf.Session() as sess:
#   print(sess.run(tt,feed_dict={num_features_nonzero:[10]}))
#   print(sess.run(random_tensor,feed_dict={num_features_nonzero:[10]}))

# with tf.variable_scope('ddd'):
#     with tf.variable_scope('www'):
#         v = tf.get_variable('v',[1])
#         print(v.name)
#         print(v)
#     v = tf.get_variable('v', [1])
#     print(v.name)
#     print(v)
########测试tf.layers.batch_normalization
# def sparse_to_tuple(sparse_mx):
#     """Convert sparse matrix to tuple representation."""
#     def to_tuple(mx):
#         if not sp.isspmatrix_coo(mx):  #要是coo类型的系数矩阵
#             mx = mx.tocoo()
#         coords = np.vstack((mx.row, mx.col)).transpose()   #矩阵转置  获得非零位置坐标
#         values = mx.data
#         shape = mx.shape
#         return coords, values, shape
# row  = np.array([0, 1])
# col  = np.array([0, 1])
# data = np.array([2., 3.])
# aa=sp.coo_matrix((data,(row,col)),shape=(2,2))
# aa=sparse_to_tuple(aa)
# row  = np.array([0, 1])
# col  = np.array([1, 0])
# data = np.array([2., 3.])
# bb=sp.coo_matrix((data,(row,col)),shape=(2,2))
# bb=sparse_to_tuple(bb)
# row  = np.array([0, 1, 2])
# col  = np.array([0, 1, 1])
# data = np.array([2., 3., 1.])
# cc=sp.coo_matrix((data,(row,col)),shape=(3,2))
# cc=sparse_to_tuple(cc)
# input=[aa,bb,cc]
# batch_size=3
# placeholders={
#     'feature': [tf.sparse_placeholder(tf.float32) for _ in range(batch_size)]
# }
# x = tf.layers.batch_normalization(placeholders['feature'], training=False)
# sess = tf.Session()# Init variables
# sess.run(tf.global_variables_initializer())
# feed_dict={}
# feed_dict.update({placeholders['feature'][i]: input[i] for i in range(batch_size)})
#
# y=sess.run(x,feed_dict=feed_dict)
# print(y[0])

for i in range(10):
    if i==3:
        continue
    print(i)