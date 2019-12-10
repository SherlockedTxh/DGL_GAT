import Final_value
import collections
import os
import numpy as np


features = []
graph_id = []
labels = []


def read_data(iid, read_datalist):
    for line1 in read_datalist:
        line = line1.split()
        label = []
        for j in range(20):
            label.append(0)
        label[Final_value.residue_type_dict[line[1]]] = 1
        labels.append(label)  # amino acid type
        graph_id.append(iid)  # graph id
        feats = []
        # sum = 0
        # for j in range(0, 18):
        #     sum += float(line[j+2])
        # for j in range(0, 18):
        #     feats.append(float(line[j+2])/sum)
        # features.append(feats)  # Normalization features
        for j in range(0, 18):
            feats.append(float(line[j+2]))
        features.append(feats)  # no Normalization features


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    in_dir = Final_value.windows_dir_pre + '/output/position_and_feature_with_dssp'
    out_dir = Final_value.windows_dir_pre + '/output/graph_1000'
    files = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]

    for i in range(0, 1000):
        infile = open(in_dir + '/' + files[i])
        datalist = list(infile)
        read_data(i, datalist)
        np.save(out_dir + '/train_feats', np.asarray(features))
        np.save(out_dir + '/train_graph_id', np.asarray(graph_id))
        np.save(out_dir + '/train_labels', np.asarray(labels))

    features.clear()
    graph_id.clear()
    labels.clear()

    for i in range(1000, 1100):
        infile = open(in_dir + '/' + files[i])
        datalist = list(infile)
        read_data(i, datalist)
        np.save(out_dir + '/valid_feats', np.asarray(features))
        np.save(out_dir + '/valid_graph_id', np.asarray(graph_id))
        np.save(out_dir + '/valid_labels', np.asarray(labels))

    features.clear()
    graph_id.clear()
    labels.clear()

    for i in range(1100, 1200):
        infile = open(in_dir + '/' + files[i])
        datalist = list(infile)
        read_data(i, datalist)
        np.save(out_dir + '/test_feats', np.asarray(features))
        np.save(out_dir + '/test_graph_id', np.asarray(graph_id))
        np.save(out_dir + '/test_labels', np.asarray(labels))


