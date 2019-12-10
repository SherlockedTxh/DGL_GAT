import  time
from src.graph import Graph
from src.Protein_gcn import *
import random
import numpy as np
from src.PDB_to_feature import load_train_txt_file
from src.Final_value import *
def main():
    t1 = time.time()
    print('Load train set ID...')
    train_set=load_train_txt_file()
    train_set.remove('1a0a')
    graph_dict={}
    train_dict={}
    train_ID=0
    for network_name in train_set:
        train_dict[train_ID]=network_name
        train_ID+=1
        print(str(train_ID)+" Read graph ..."+network_name+"'s edgelist_file!")
        g = Graph()
        Path_edgelist_file = windows_dir_pre + '/output/edgelist/edgelist_'+network_name+'.txt'
        g.read_edgelist_file(path=Path_edgelist_file, weighted=True, directed=False)

        print(str(train_ID)+' Read graph ...'+network_name+"'s node_label_file!")
        Path_node_label_file = windows_dir_pre + '/output/residue_type/residue_type_'+network_name+'.txt'
        g.read_node_label_file(Path_node_label_file)
        print(str(train_ID)+' Read graph ...' + network_name + "'s feature_file!")
        Path_node_feature_file = windows_dir_pre + '/output/feature/feature_'+network_name+'.txt'
        g.read_node_features_file(Path_node_feature_file)
        graph_dict[network_name]=g
        if train_ID==200:
            break
    Protein_GCN_model=Protein_GCN(graph_dict=graph_dict,train_dict=train_dict,learning_rate=learning_rate,
                                  epochs=epochs,hidden1=hidden1,dropout=dropout, weight_decay=weight_decay, early_stopping=early_stopping,#权值衰减
                 max_degree=max_degree, clf_ratio=clf_ratio,Path_output=Path_output,batch_size=batch_size)
    # if args.graph_format == 'adjlist':
    #     g.read_adjlist(filename=args.input)
    # elif args.graph_format == 'edgelist':
    #     g.read_edgelist(filename=args.input, weighted=args.weighted,
    #                     directed=args.directed)


    t2 = time.time()

    # for node in g.G.nodes():
    #     print(node+' '+str(g.G.nodes[node]['feature'])+' '+str(g.G.nodes[node]['label']))




if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main()
