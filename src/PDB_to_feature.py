
import os
import collections
import glob
from Final_value import windows_dir_pre

GLY = []
CYS = []
ARG = []
SER = []
THR = []
LYS = []
MET = []
ALA = []
LEU = []
ILE = []
VAL = []
ASP = []
GLU = []
HIS = []
ASN = []
PRO = []
GLN = []
PHE = []
TRP = []
TYR = []

res_container_dict = {0: HIS, 1: LYS, 2: ARG, 3: ASP, 4: GLU, 5: SER, 6: THR, 7: ASN, 8: GLN, 9: ALA, 10: VAL, 11: LEU,
                      12: ILE, 13: MET, 14: PHE, 15: TYR, 16: TRP, 17: PRO, 18: GLY, 19: CYS}



class PDB_residue:#
    def __init__(self, has_CA, residue_type, residue_ID, CA_x, CA_y, CA_z, num_C ,num_N ,num_O , num_S, index):
        self.has_CA=has_CA
        self.residue_type = residue_type
        self.residue_ID = residue_ID
        self.CA_x = CA_x
        self.CA_y = CA_y
        self.CA_z = CA_z
        self.num_C=num_C
        self.num_N=num_N
        self.num_O=num_O
        self.num_S=num_S
        self.index = index

    def get_has_CA(self):
        return self.has_CA
    def get_residue_type(self):
        return self.residue_type
    def get_residue_ID(self):
        return self.residue_ID
    def get_CA_x(self):
        return self.CA_x
    def get_CA_y(self):
        return self.CA_y
    def get_CA_z(self):
        return self.CA_z
    def get_num_N(self):
        return self.num_N
    def get_num_C(self):
        return self.num_C
    def get_num_O(self):
        return self.num_O
    def get_num_S(self):
        return self.num_S
    def get_index(self):
        return self.index

    def set_CA_x(self,CA_x):
        self.CA_x=CA_x
    def set_CA_y(self,CA_y):
        self.CA_y=CA_y
    def set_CA_z(self,CA_z):
        self.CA_z=CA_z
    def set_has_CA(self,has_CA):
        self.has_CA=has_CA
    def set_num_C(self,num_C):
        self.num_C=num_C
    def set_num_N(self,num_N):
        self.num_N=num_N
    def set_num_O(self,num_O):
        self.num_O=num_O
    def set_num_S(self,num_S):
        self.num_S=num_S
    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def load_train_txt_file():
    PDB_train_Set = set()
    if os.path.exists(windows_dir_pre + '/PDB_family_train/PDB_family_train.txt'):
        PDB_train_file = open(windows_dir_pre + '/PDB_family_train/PDB_family_train.txt')
        pdb_train_dir = windows_dir_pre + '/PDB_family_train'
        for line in PDB_train_file:
            PDB_ID = line.split()[0]
            PDB_train_Set.add(PDB_ID.lower())
        PDB_train_file.close()
    else:
        print('The file: '+windows_dir_pre+'/PDB_family_train/PDB_family_train.txt'+'is not exist!')

    return PDB_train_Set


def read_PDB(entry_list):
    ID_dict = collections.OrderedDict()
    index=0
    for line1 in entry_list:
        line = line1[0:6].strip(' ')
        if line == 'ATOM':  # or line[0]=="HETATM":
            atom_name = (line1[13:16].strip(' '))
            residue_type = (line1[17:21].strip(' '))
            residue_ID = line1[21:27]
            chain = residue_ID[0]
            chain_number = residue_ID[1:].strip(' ')
            residue_ID = chain+'_'+chain_number
            new_pos = [line1[30:38], line1[38:46], line1[46:54]]
            atom_type=line1[77]
            CA_x="0000.000"
            CA_y="0000.000"
            CA_z="0000.000"
            num_C=0
            num_N=0
            num_O=0
            num_S=0
            has_CA = 0
            if atom_type=='C':
                num_C=1
            if atom_type=='N':
                num_N=1
            if atom_type=='O':
                num_O=1
            if atom_type=='S':
                num_S=1
            if atom_name == 'CA':
                CA_x=new_pos[0]
                CA_y=new_pos[1]
                CA_z=new_pos[2]
                has_CA=1
            if residue_ID not in ID_dict.keys():
                if atom_name == "CA":
                    residue = PDB_residue(has_CA, residue_type, residue_ID, CA_x, CA_y, CA_z, num_C ,num_N ,num_O , num_S, index)
                    index=index+1
                    ID_dict[residue_ID] = residue
            else:
                residue=ID_dict[residue_ID]
                temp_C = residue.get_num_C()+num_C
                temp_N = residue.get_num_N()+num_N
                temp_O = residue.get_num_O()+num_O
                temp_S = residue.get_num_S()+num_S
                residue.set_num_C(temp_C)
                residue.set_num_O(temp_O)
                residue.set_num_N(temp_N)
                residue.set_num_S(temp_S)
                if residue.get_has_CA()==0:
                    residue.set_has_CA(has_CA)
                    residue.set_CA_x (CA_x)
                    residue.set_CA_y (CA_y)
                    residue.set_CA_z (CA_z)
                ID_dict[residue_ID] = residue


    return ID_dict

def write_str_to_file(string0,path):
    fp = open(path, 'w', encoding='utf-8')
    fp.write(string0)
    fp.close()



if __name__ == '__main__':

    mode = 'S'  # using 'carbon', 'oxygen', 'sulfur', and 'nitrogen' channels
    num_of_channels = 4
    print("begin load_train_txt_file()......")
    PDB_train_Set = load_train_txt_file()
    PDB_number = 0
    for PDB_ID in PDB_train_Set:
        PDB_path = windows_dir_pre + '/PDB_family_train/'+PDB_ID+'.pdb'
        output_path = windows_dir_pre + '/output/position_and_feature/'
        string0 = ""
        if os.path.exists(PDB_path):
            residue_file = open(PDB_path)
            infile = list(residue_file)
            ID_dict = read_PDB(infile)
            for residue_ID in ID_dict.keys():
                residue = ID_dict[residue_ID]
                string0 = string0 + str(
                    residue.get_has_CA()) + " " + residue.get_residue_type() + " " + residue.get_residue_ID() + " " + str(
                    residue.get_CA_x()) + " " + str(residue.get_CA_y()) + " " + str(residue.get_CA_z()) + " " + str(
                    residue.get_num_C()) + " " + str(residue.get_num_N()) + " " + str(residue.get_num_O()) + " " + str(
                    residue.get_num_S()) + " " + str(residue.get_index()) + "\n"
            # print(string0)
            write_str_to_file(string0, output_path + PDB_ID+'.txt')
        else:
            print('The file: ' + windows_dir_pre + '/PDB_family_train/'+PDB_ID+'.pdb' + 'is not exist!')

# class PDB_residue:#
#     def __init__(self, has_CA, residue_type, residue_ID, CA_x, CA_y, CA_z, num_C ,num_N ,num_O , num_S, index):
#         self.has_CA=has_CA
#         self.residue_type = residue_type
#         self.residue_ID = residue_ID
#         self.CA_x = CA_x
#         self.CA_y = CA_y
#         self.CA_z = CA_z
#         self.num_C=num_C
#         self.num_N=num_N
#         self.num_O=num_O
#         self.num_S=num_S
#         self.index = index
#
#     def get_has_CA(self):
#         return self.has_CA
#     def get_residue_type(self):
#         return self.residue_type
#     def get_residue_ID(self):
#         return self.residue_ID
#     def get_CA_x(self):
#         return self.CA_x
#     def get_CA_y(self):
#         return self.CA_y
#     def get_CA_z(self):
#         return self.CA_z
#     def get_num_N(self):
#         return self.num_N
#     def get_num_C(self):
#         return self.num_C
#     def get_num_O(self):
#         return self.num_O
#     def get_num_S(self):
#         return self.num_S
#     def get_index(self):
#         return self.index
#
#     def set_CA_x(self,CA_x):
#         self.CA_x=CA_x
#     def set_CA_y(self,CA_y):
#         self.CA_y=CA_y
#     def set_CA_z(self,CA_z):
#         self.CA_z=CA_z
#     def set_has_CA(self,has_CA):
#         self.has_CA=has_CA
#     def set_num_C(self,num_C):
#         self.num_C=num_C
#     def set_num_N(self,num_N):
#         self.num_N=num_N
#     def set_num_O(self,num_O):
#         self.num_O=num_O
#     def set_num_S(self,num_S):
#         self.num_S=num_S
#     def __eq__(self, other):
#         return self.__dict__ == other.__dict__
#
#
#
# def read_PDB(entry_list,output_path):
#     ID_dict = collections.OrderedDict()
#     index=0
#     output_filename=''
#     num=0
#     for line1 in entry_list:
#         line = line1.split()
#         if line[0] == 'ATOM':  # or line[0]=="HETATM":
#             atom_name = (line1[13:17].strip(' '))
#             residue_type = (line1[17:21].strip(' '))
#             residue_ID = line1[21:27]
#             chain_number = residue_ID[1:].strip(' ')
#             residue_ID = chain_number
#             new_pos = [line1[30:38], line1[38:46], line1[46:54]]
#             atom_type=line1[77]
#             CA_x="0000.000"
#             CA_y="0000.000"
#             CA_z="0000.000"
#             num_C=0
#             num_N=0
#             num_O=0
#             num_S=0
#             has_CA = 0
#             if atom_type=='C':
#                 num_C=1
#             if atom_type=='N':
#                 num_N=1
#             if atom_type=='O':
#                 num_O=1
#             if atom_type=='S':
#                 num_S=1
#             if atom_name == 'CA':
#                 CA_x=new_pos[0]
#                 CA_y=new_pos[1]
#                 CA_z=new_pos[2]
#                 has_CA=1
#             if residue_ID not in ID_dict.keys():
#                 residue = PDB_residue(has_CA, residue_type, residue_ID, CA_x, CA_y, CA_z, num_C ,num_N ,num_O , num_S, index)
#                 index=index+1
#                 ID_dict[residue_ID] = residue
#             else:
#                 residue=ID_dict[residue_ID]
#                 temp_C = residue.get_num_C()+num_C
#                 temp_N = residue.get_num_N()+num_N
#                 temp_O = residue.get_num_O()+num_O
#                 temp_S = residue.get_num_S()+num_S
#                 residue.set_num_C(temp_C)
#                 residue.set_num_O(temp_O)
#                 residue.set_num_N(temp_N)
#                 residue.set_num_S(temp_S)
#                 if residue.get_has_CA()==0:
#                     residue.set_has_CA(has_CA)
#                     residue.set_CA_x (CA_x)
#                     residue.set_CA_y (CA_y)
#                     residue.set_CA_z (CA_z)
#                 ID_dict[residue_ID] = residue
#         elif line[0] == 'ENDMDL':
#             if num%50==0 or num==49999:
#                 string0 = ""
#                 for residue_ID in ID_dict.keys():
#                     residue = ID_dict[residue_ID]
#                     string0 = string0 + str(
#                         residue.get_has_CA()) + " " + residue.get_residue_type() + " " + residue.get_residue_ID() + " " + str(
#                         residue.get_CA_x()) + " " + str(residue.get_CA_y()) + " " + str(residue.get_CA_z()) + " " + str(
#                         residue.get_num_C()) + " " + str(residue.get_num_N()) + " " + str(residue.get_num_O()) + " " + str(
#                         residue.get_num_S()) + " " + str(residue.get_index()) + "\n"
#                     # print(string0)
#                 write_str_to_file(string0, output_path + output_filename )
#                 print(output_filename)
#             index = 0
#             num+=1
#             ID_dict.clear()
#         elif line[0] == 'MODEL':
#             index = 0
#             time_point=line[1]
#             output_filename='point_'+time_point+'.txt'
#
#
#     return ID_dict
#
# def write_str_to_file(string0,path):
#     fp=open(path,'w',encoding='utf-8')
#     fp.write(string0)
#     fp.close()
#
# def _find_data_files( search_path):
#     all_files=glob.glob(search_path+'point*')
#     return [name for name in all_files ]
#
# def read_position_and_feature_file(path):
#     residue_position_dict=collections.OrderedDict()
#     residue_feature_dict=collections.OrderedDict()
#     residue_type_dict=collections.OrderedDict()
#     if os.path.exists(path):
#         fp=open(path)
#         position_and_feature_file=list(fp)
#         for line in position_and_feature_file:
#             line_split=line.split()
#             if int(line_split[0])==1:
#                 residue_position_dict[line_split[2]]=(float(line_split[3]),float(line_split[4]),float(line_split[5]))
#                 residue_feature_dict[line_split[2]]=(line_split[6],line_split[7],line_split[8],line_split[9])
#                 residue_type_dict[line_split[2]]=line_split[1]
#         fp.close()
#     else:
#         print(path+' is not exist!')
#     return residue_position_dict,residue_feature_dict,residue_type_dict
# def cal_distance(a,b):
#     return ((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)**0.5
# if __name__ == '__main__':
#
#     dynamcis=['dynamcis_1','dynamcis_18']
#     for dyn in dynamcis:
#         point_path = windows_dir_pre + '/dynamcis/output/position_and_feature/'+dyn+'/'
#         Path_edgelist = windows_dir_pre + '/dynamcis/output/edgelist/'+dyn+'/'
#         Path_residue_type = windows_dir_pre + '/dynamcis/output/residue_type/'+dyn+'/'
#         point_txt_file_list=_find_data_files(search_path=point_path)
#         point_num=1
#         for point_txt in point_txt_file_list:
#             if os.path.exists(point_txt):
#                 edgelist = ""
#                 feature = ""
#                 residue_type = ""
#                 (residue_position_dict, residue_feature_dict, residue_type_dict) = read_position_and_feature_file(
#                     point_txt)
#                 residue_ID_list = list(residue_position_dict.keys())
#                 len_residue_ID_list = len(residue_ID_list)
#                 for i in range(len_residue_ID_list):
#                     residue_type = residue_type + residue_ID_list[i] + "  " + residue_type_dict[
#                         residue_ID_list[i]] + "\n"
#                     for j in range(i + 1, len_residue_ID_list):
#                         temp_distance = cal_distance(residue_position_dict[residue_ID_list[i]],
#                                                      residue_position_dict[residue_ID_list[j]])
#                         edgelist = edgelist + residue_ID_list[i] + "  " + residue_ID_list[j] + "  " + str(
#                                 temp_distance) + "\n"
#                 write_str_to_file(edgelist, Path_edgelist + "edgelist_point_" + str(point_num) + ".txt")
#                 write_str_to_file(residue_type, Path_residue_type + "residue_type_point_" + str(point_num) + ".txt")
#                 point_num+=50
#
#             else:
#                 print('The file: ' + point_txt+ ' is not exist!')
#
#     # for dyn in dynamcis:
#     #     PDB_path = windows_dir_pre + '/dynamcis/'+dyn+'.pdb'
#     #     output_path = windows_dir_pre + '/dynamcis/output/position_and_feature/'+dyn+'/'
#     #     if os.path.exists(PDB_path):
#     #         residue_file = open(PDB_path)
#     #         infile = list(residue_file)
#     #         ID_dict = read_PDB(infile,output_path=output_path)
#     #         residue_file.close()
#     #
#     #     else:
#     #         print('The file: ' + windows_dir_pre + '/dynamcis/'+dyn+'.pdb' + ' is not exist!')
#
