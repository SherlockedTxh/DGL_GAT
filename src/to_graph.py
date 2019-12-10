import numpy as np
from Final_value import windows_dir_pre
import os
import json

nodes = []
links = []


def npy_to_graph(npy, graph):
    graph_num = []
    cnt = 1
    for i in range(1, len(graph)):
        if graph[i] == graph[i - 1]:
            cnt += 1
        else:
            graph_num.append(cnt)
            cnt = 1
    # print(graph_num)
    cnt = 0
    count = 0
    while count < len(graph_num):
        for i in range(cnt, graph_num[count] + cnt):
            nodes.append({"id": i + 1})
            for j in range(i + 1, graph_num[count] + cnt):
                if cal_distance(npy[i][0], npy[i][1], npy[i][2], npy[j][0], npy[j][1], npy[j][2]) <= 9.0:
                    links.append({"source": i, "target": j})  # one way
                    # print(npy[i][0], npy[i][1], npy[i][2], npy[j][0], npy[j][1], npy[j][2])
                    print("source:", i, "target:", j)
        cnt += graph_num[count]
        count += 1


def cal_distance(a1, a2, a3, b1, b2, b3):
    return ((a1 - b1) ** 2 + (a2 - b2) ** 2 + (a3 - b3) ** 2) ** 0.5


if __name__ == '__main__':
    in_dir = windows_dir_pre + '/output/graph_1000'
    out_dir = windows_dir_pre + '/output/graph_1000'
    train = np.load(in_dir + '/train_feats.npy')
    valid = np.load(in_dir + '/valid_feats.npy')
    test = np.load(in_dir + '/test_feats.npy')
    train_graph_id = np.load(in_dir + '/train_graph_id.npy')
    valid_graph_id = np.load(in_dir + '/valid_graph_id.npy')
    test_graph_id = np.load(in_dir + '/test_graph_id.npy')

    npy_to_graph(valid, valid_graph_id)
    valid_graph_dict = {"directed": True, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}
    valid_graph_json = json.dumps(valid_graph_dict)
    fileObject = open(in_dir + '/valid_graph.json', 'w')
    fileObject.write(valid_graph_json)
    fileObject.close()
    print("valid done!")

    nodes.clear()
    links.clear()

    npy_to_graph(test, test_graph_id)
    test_graph_dict = {"directed": True, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}
    test_graph_json = json.dumps(test_graph_dict)
    fileObject = open(in_dir + '/test_graph.json', 'w')
    fileObject.write(test_graph_json)
    fileObject.close()

    nodes.clear()
    links.clear()

    npy_to_graph(train, train_graph_id)
    train_graph_dict = {"directed": True, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}
    train_graph_json = json.dumps(train_graph_dict)
    fileObject = open(in_dir + '/train_graph.json', 'w')
    fileObject.write(train_graph_json)
    fileObject.close()