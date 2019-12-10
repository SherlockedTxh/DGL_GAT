import os
import collections

residue_type_dict={'HIS':0, 'LYS':1, 'ARG':2, 'ASP':3,  'GLU':4,  'SER':5,  'THR':6,  'ASN':7,  'GLN':8,  'ALA':9,  'VAL':10,  'LEU':11,
                       'ILE':12,  'MET':13,  'PHE':14,  'TYR':15,  'TRP':16, 'PRO':17,  'GLY':18,  'CYS':19}

# windows_dir_pre =os.path.dirname(os.path.abspath('.'))
windows_dir_pre = "/mnt/md1/a503tongxueheng/3DCNN_data_process/data"
learning_rate=0.002
epochs=1000
hidden1=64
dropout=0.5
weight_decay=5e-4
early_stopping=100
max_degree=3
clf_ratio=0.8
Path_output=windows_dir_pre + '/output/model/'
batch_size=20