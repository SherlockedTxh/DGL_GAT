"""Graph utilities."""

# from time import time
import networkx as nx
import os
import pickle as pkl
import numpy as np
import scipy.sparse as sp

__author__ = "Gao Rui"


class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0
    def get_edges(self):
        t=list(self.G.edges)
        for i in range(len(t)):
            e=t[i]
            print(e[0]+e[1]+str(self.G[e[0]][e[1]]['weight']))   #边是以元祖的形式存储
        print(t[1][0])

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_g(self, g):
        self.G = g
        self.encode_node()

    def read_adjlist_file(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def read_undirected_unweighted(self, line):
        line_split=line.split()
        src=line_split[0]
        dst=line_split[1]
        self.G.add_edge(src,dst)
        self.G.add_edge(dst,src)
        self.G[src][dst]['weight'] = 1.0
        self.G[dst][src]['weight'] = 1.0

    def read_undirected_weighted(self, line):
        line_split=line.split()
        src=line_split[0]
        dst=line_split[1]
        w=float(line_split[2])
        self.G.add_edge(src,dst)
        self.G.add_edge(dst,src)
        self.G[src][dst]['weight'] = w
        self.G[dst][src]['weight'] = w

    def read_directed_unweighted(self, line):
        line_split=line.split()
        src=line_split[0]
        dst=line_split[1]
        self.G.add_edge(src,dst)
        self.G[src][dst]['weight'] = 1.0

    def read_directed_weighted(self, line):
        line_split=line.split()
        src=line_split[0]
        dst=line_split[1]
        w=float(line_split[2])
        self.G.add_edge(src,dst)
        self.G[src][dst]['weight'] = w
    def read_edgelist_file(self, path='', weighted=False, directed=False):    #创建网络
        self.G = nx.DiGraph()
        if os.path.exists(path):
            fin = open(path, 'r')
            entrylist=list(fin)
            func = self.read_undirected_unweighted
            if directed and not weighted:
                func= self.read_directed_unweighted
            if not directed and weighted:
                func=self.read_undirected_weighted
            if directed and weighted:
                func = self.read_directed_weighted

            for line in entrylist:
                func(line)
            fin.close()
            self.encode_node()
        else:
            print('the file: '+path+' is not exsit!')

    def read_node_label_file(self, path):
        fp = open(path, 'r')
        entrylist=list(fp)
        for l in entrylist:
            line = l.split()
            if self.G.has_node(line[0]):
                self.G.nodes[line[0]]['label'] = line[1:]
        fp.close()

    def read_node_features_file(self, path):
        fp = open(path, 'r')
        for l in fp.readlines():
            line = l.split()
            if self.G.has_node(line[0]):
                self.G.nodes[line[0]]['feature'] = np.array(
                    [float(x) for x in line[1:]])
        fp.close()

    def read_node_status_file(self, path):
        fp = open(path, 'r')
        for l in fp.readlines():
            line = l.split()
            self.G.nodes[line[0]]['status'] = line[1]  # train test valid
        fp.close()

    def read_edge_label_file(self, path):
        fp = open(path, 'r')
        for l in fp.readlines():
            line = l.split()
            self.G[line[0]][line[1]]['label'] = line[2:]
        fp.close()
