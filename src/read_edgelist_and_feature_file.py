import os
import collections
from src import PDB_to_feature as ptf
from src.Final_value import windows_dir_pre
from src.Final_value import residue_type_dict as res_type_dict
distance_threshold=float(9)

def read_position_and_feature_file(path):
    residue_position_dict=collections.OrderedDict()
    residue_feature_dict=collections.OrderedDict()
    residue_type_dict=collections.OrderedDict()
    flag=True
    if os.path.exists(path):
        fp=open(path)
        position_and_feature_file=list(fp)
        for line in position_and_feature_file:
            line_split=line.split()
            residue_position_dict[line_split[0]] = (
            float(line_split[2]), float(line_split[3]), float(line_split[4]))
            residue_feature_dict[line_split[0]] = (line_split[5],line_split[6], line_split[7], line_split[8], line_split[9],line_split[10]
                                                   ,line_split[11],line_split[12],line_split[13],line_split[14])
            residue_type_dict[line_split[0]] = line_split[1]
        fp.close()
    else:
        flag=False
        print(path+' is not exist!')
    return residue_position_dict,residue_feature_dict,residue_type_dict,flag

def cal_distance(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)**0.5
if __name__=='__main__':
    Path_position_and_feature_file=windows_dir_pre+'/output/position_and_feature_with_dssp/'
    Path_edgelist=windows_dir_pre+'/output/edgelist/'
    Path_feature=windows_dir_pre+'/output/feature/'
    Path_residue_type=windows_dir_pre+'/output/residue_type/'
    PDB_filename=ptf.load_train_txt_file()
    string_error_name=''
    ss=0
    num=0
    for PDB_ID in PDB_filename:
        edgelist=""
        feature=""
        residue_type=""
        num+=1
        (residue_position_dict,residue_feature_dict,residue_type_dict,flag)=read_position_and_feature_file(Path_position_and_feature_file+PDB_ID+'.txt')
        if not flag:
            string_error_name+=PDB_ID+'\n'
            ss+=1
            continue
        else:
            residue_ID_list=list(residue_position_dict.keys())
            len_residue_ID_list=len(residue_ID_list)
            for i in range(len_residue_ID_list):
                temp_feature=residue_feature_dict[residue_ID_list[i]]
                feature=feature+residue_ID_list[i]+"  "+temp_feature[0]+"  "+temp_feature[1]+"  "+temp_feature[2]+"  "+temp_feature[3]\
                        +"  "+temp_feature[4]+"  "+temp_feature[5]+"  "+temp_feature[6]+"  "+temp_feature[7]+"  "+temp_feature[8] \
                        + "  " + temp_feature[9]+"\n"
                residue_type=residue_type+residue_ID_list[i]+"  "+residue_type_dict[residue_ID_list[i]]+"\n"
                for j in range(i+1,len_residue_ID_list):
                    temp_distance=cal_distance(residue_position_dict[residue_ID_list[i]],residue_position_dict[residue_ID_list[j]])
                    try:
                        if temp_distance<=9.0:
                            edgelist=edgelist+residue_ID_list[i] + "  " + residue_ID_list[j] + "  " + str(1/temp_distance)+"\n"
                    except Exception as e:
                        print(temp_distance)
                        print(PDB_ID+" "+residue_ID_list[i]+"  "+residue_ID_list[j])
            ptf.write_str_to_file(edgelist,Path_edgelist+"edgelist_"+PDB_ID+".txt")
            ptf.write_str_to_file(feature,Path_feature+"feature_"+PDB_ID+".txt")
            ptf.write_str_to_file(residue_type,Path_residue_type+"residue_type_"+PDB_ID+".txt")
        print(str(num) + ": "+str(ss))
    ptf.write_str_to_file(string_error_name,windows_dir_pre + '/output/error_file_name.txt')