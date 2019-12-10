# -*- encoding: utf-8 -*-

import os

# 特征存储顺序为支链亲水性，分子量，等电点，解离常数（羧基），解离常数（氨基），这些是氨基酸本身的性质

#pdb文件路径
dir = "/mnt/md1/a503tongxueheng/3DCNN_data_process/data/PDB_family_train"

amnio_fearture_dict = {
    'GLY':[-0.4,75.07,6.06,2.35,9.78],
    'CYS':[2.5,121.16,5.05,1.92,10.70],
    'ARG':[-4.5,174.20,10.76,1.82,8.99],
    'SER':[-0.8,105.09,5.68,2.19,9.21],
    'THR':[-0.7,119.12,5.60,2.09,9.10],
    'LYS':[-3.9,146.19,9.60,2.16,9.06],
    'MET':[1.9,149.21,5.74,2.13,9.28],
    'ALA':[1.8,89.09,6.11,2.35,9.87],
    'LEU':[3.8,131.17,6.01,2.33,9.74],
    'ILE':[4.5,131.17,6.05,2.32,9.76],
    'VAL':[4.2,117.15,6.00,2.39,9.74],
    'ASP':[-3.5,133.10,2.85,1.99,9.90],
    'GLU':[-3.5,147.13,3.15,2.10,9.47],
    'HIS':[-3.2,155.16,7.60,1.80,9.33],
    'ASN':[-3.5,132.12,5.41,2.14,8.72],
    'PRO':[-1.6,115.13,6.30,1.95,10.64],
    'GLN':[-3.5,146.15,5.65,2.17,9.13],
    'PHE':[2.8,165.19,5.49,2.20,9.31],
    'TRP':[-0.9,204.23,5.89,2.46,9.41],
    'TYR':[-1.3,181.19,5.64,2.20,9.21]
}

def grab_pdb(entry_list,filename):
    f = open("/mnt/md1/a503tongxueheng/GAT_feature/"+filename[:-4]+".txt", "w")
    for line_content in entry_list:
        line = line_content.split()
        
        if line[0] == "ATOM":
            res = (line_content[17:20])
            if res in amnio_fearture_dict.keys():
                atom=(line_content[13:16].strip(' '))
                chain = line_content[21:26]
                chain_ID = chain[0]
                res_Seq = chain[1:].strip(' ')
                res_Seq = int(res_Seq)
                value = amnio_fearture_dict[res]
                if(atom == 'CA'):
                    f.write(chain_ID+"_"+str(res_Seq)+" "+str(value[0])+" "+str(value[1])+" "+str(value[2])+" "+str(value[3])+" "+str(value[4])+"\n")
    f.close()

if __name__ == '__main__':

    #输入、输出路径
    pdb_dir = dir + ''
    out_dir = '/mnt/md1/a503tongxueheng/GAT_feature'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for filename in os.listdir(pdb_dir):
        pdb_file = open(pdb_dir+'/'+filename)
        if(filename[-3:] == 'pdb'):
            infile = list(pdb_file)
            grab_pdb(infile,filename)
    
    print("complete!")