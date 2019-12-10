
import collections
import os
from Final_value import windows_dir_pre
from Final_value import outpath
from Final_value import outpath_value
from Final_value import final_path


def load_train_txt_file():
    PDB_train_Set = set()
    if (os.path.exists(windows_dir_pre + '/PDB_family_train/PDB_family_train.txt')):
        PDB_train_file = open(windows_dir_pre + '/PDB_family_train/PDB_family_train.txt')
        pdb_train_dir = windows_dir_pre + '/PDB_family_train'
        for line in PDB_train_file:
            PDB_ID = line.split()[0]
            PDB_train_Set.add(PDB_ID.lower())
        PDB_train_file.close()
    else:
        print('The file: ' + windows_dir_pre + '/PDB_family_train/PDB_family_train.txt' + 'is not exist!')
    return PDB_train_Set


def read_dssp(entrylist):
    ID_dict = collections.OrderedDict()
    flag = False
    for line1 in entrylist:
        line = line1.split()
        if line[0] == '#':
            flag = True
            continue
        if flag:
            if line1[13] != '!':
                residuenum = line1[5:11].strip(' ')
                chain = line1[11:13].strip(' ')
                residue_ID = chain + '_' + residuenum
                acc = line1[34:38].strip(' ')
                tco = line1[85:91].strip(' ')
                kappa = line1[91:97].strip(' ')
                alpha = line1[97:103].strip(' ')
                phi = line1[103:109].strip(' ')
                psi = line1[109:115].strip(' ')
                ID_dict[residue_ID] = (acc, tco, kappa, alpha, phi, psi)
    return ID_dict


def read_feature(entrylist):
    ID_dict = collections.OrderedDict()
    for line1 in entrylist:
        line = line1.split()
        ID_dict[line[2]] = (line[1], line[3], line[4], line[5], line[6], line[7], line[8], line[9])
    return ID_dict


# feature_process
def read_value_feature(entrylist):
    ID_dict = collections.OrderedDict()
    for line1 in entrylist:
        line = line1.split()
        ID_dict[line[0]] = (line[1], line[2], line[3], line[4], line[5])
    return ID_dict


def write_str_to_file(string0, path):
    fp = open(path, 'w', encoding='utf-8')
    fp.write(string0)
    fp.close()


if __name__ == '__main__':
    # print("begin load_train_txt_file()......")
    # PDB_train_Set = load_train_txt_file()
    # print(len(PDB_train_Set))
    # for PDB_ID in PDB_train_Set:
    for filename in os.listdir(windows_dir_pre):
        # print(PDB_ID)
        feature_path = outpath + filename + '.txt'
        # dssp_path = windows_dir_pre + '/output/dssp/dssp/' + PDB_ID + '.txt'
        value_path = outpath_value + filename + '.txt'
        output_path = final_path
        if os.path.exists(feature_path) and os.path.exists(value_path):
            feature_file = open(feature_path)
            # dssp_file = open(dssp_path)
            value_file = open(value_path)
            feature_infile = list(feature_file)
            # dssp_infile = list(dssp_file)
            value_infile = list(value_file)
            feature_dict = read_feature(feature_infile)
            # dssp_dict = read_dssp(dssp_infile)
            value_dict = read_value_feature(value_infile)
            output_string = ''
            for residue_ID in feature_dict.keys():
                if residue_ID not in feature_dict.keys() or residue_ID not in value_dict.keys():
                    print("Error! " + residue_ID + " is not in feature_dict of " + filename)
                else:
                    feature = feature_dict[residue_ID]
                    # dssp = dssp_dict[residue_ID]
                    value = value_dict[residue_ID]
                    output_string += residue_ID + " " + feature[0] + " " + feature[1] + " " + feature[2] + " " + \
                                     feature[3] + " " + feature[4] + " " + feature[5] + " " + feature[6] + " " + \
                                     feature[7] + " " + value[0] + " " + \
                                     value[1] + " " + value[2] + " " + value[3] + " " + value[4] + "\n"

            write_str_to_file(output_string, output_path + filename + '.txt')
            feature_file.close()
            # dssp_file.close()
        else:
            print("Error! " + "No file of " + filename)
