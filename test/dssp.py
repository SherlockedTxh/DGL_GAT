"""
This example client takes a PDB file, sends it to the REST service, which
creates HSSP data. The HSSP data is then output to the console.

Example:

    python pdb_to_hssp.py 1crn.pdb http://www.cmbi.umcn.nl/xssp/
"""

import argparse
import json
import requests
import time
import os
from Final_value import windows_dir_pre
def load_train_txt_file():
    PDB_train_Set = set()
    if(os.path.exists(windows_dir_pre+'/PDB_family_train/PDB_family_train.txt')):
        PDB_train_file = open(windows_dir_pre + '/PDB_family_train/PDB_family_train.txt')
        pdb_train_dir = windows_dir_pre + '/PDB_family_train'
        for line in PDB_train_file:
            PDB_ID = line.split()[0]
            PDB_train_Set.add(PDB_ID.lower())
        PDB_train_file.close()
    else:
        print('The file: '+windows_dir_pre+'/PDB_family_train/PDB_family_train.txt'+'is not exist!')

    return PDB_train_Set
def load_train_txt_file_read():
    PDB_train_Set = set()
    if(os.path.exists(windows_dir_pre+'/output/dssp/has_read.txt')):
        PDB_train_file = open(windows_dir_pre + '/output/dssp/has_read.txt')
        for line in PDB_train_file:
            PDB_ID = line.split()[0]
            PDB_train_Set.add(PDB_ID.lower())
        PDB_train_file.close()
    else:
        print('The file: '+windows_dir_pre+ '/output/dssp/has_read.txt'+'is not exist!')

    return PDB_train_Set
def write_str_to_file(string0,path):
    fp=open(path,'w',encoding='utf-8')
    fp.write(string0)
    fp.close()
def pdb_to_hssp(pdb_file_path, rest_url):
    # Read the pdb file data into a variable
    files = {'file_': open(pdb_file_path, 'rb')}

    # Send a request to the server to create hssp data from the pdb file data.
    # If an error occurs, an exception is raised and the program exits. If the
    # request is successful, the id of the job running on the server is
    # returned.
    url_create = '{}api/create/pdb_file/dssp/'.format(rest_url)
    r = requests.post(url_create, files=files)
    r.raise_for_status()
    job_id = json.loads(r.text)['id']
    print("Job submitted successfully. Id is: '{}'".format(job_id))

    # Loop until the job running on the server has finished, either successfully
    # or due to an error.
    ready = False
    while not ready:
        # Check the status of the running job. If an error occurs an exception
        # is raised and the program exits. If the request is successful, the
        # status is returned.
        url_status = '{}api/status/pdb_file/dssp/{}/'.format(rest_url,
                                                                  job_id)
        r = requests.get(url_status)
        r.raise_for_status()

        status = json.loads(r.text)['status']
        print("Job status is: '{}'".format(status))

        # If the status equals SUCCESS, exit out of the loop by changing the
        # condition ready. This causes the code to drop into the `else` block
        # below.
        #
        # If the status equals either FAILURE or REVOKED, an exception is raised
        # containing the error message. The program exits.
        #
        # Otherwise, wait for five seconds and start at the beginning of the
        # loop again.
        if status == 'SUCCESS':
            ready = True
        elif status in ['FAILURE', 'REVOKED']:
            raise Exception(json.loads(r.text)['message'])
        else:
            time.sleep(10)
    else:
        # Requests the result of the job. If an error occurs an exception is
        # raised and the program exits. If the request is successful, the result
        # is returned.
        url_result = '{}api/result/pdb_file/dssp/{}/'.format(rest_url,
                                                                  job_id)
        r = requests.get(url_result)
        r.raise_for_status()
        result = json.loads(r.text)['result']

        # Return the result to the caller, which prints it to the screen.
        return result


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Create HSSP from a PDB file')
    # parser.add_argument('pdb_file_path')
    # parser.add_argument('rest_url')
    rest_url='http://www.cmbi.umcn.nl/xssp/'
    # args = parser.parse_args()
    print("begin load_train_txt_file()......")
    PDB_train_Set = load_train_txt_file()
    PDB_has_read_Set=load_train_txt_file_read()
    PDB_number = 0
    string=''
    string_exception=''
    for PDB_ID in PDB_train_Set:
        PDB_path = windows_dir_pre + '/PDB_family_train/' + PDB_ID + '.pdb'
        output_path = windows_dir_pre + '/output/dssp/'
        if PDB_ID not in PDB_has_read_Set:
            print(PDB_ID+" "+str(PDB_number))
            PDB_number+=1
            string+=PDB_ID+"\n"
            try:
                result = pdb_to_hssp(PDB_path, rest_url)
                write_str_to_file(result, output_path + PDB_ID + '.txt')
                write_str_to_file(string, output_path + 'has_read.txt')
                time.sleep(2)
            except Exception as e:
                print(str(e))
                string_exception+=PDB_ID+"\n"
                write_str_to_file(string_exception, output_path + 'exception_PDB_ID.txt')
                print('Exception occurs up! Wait 10 seconds!')
                time.sleep(10)

        else:
            print(PDB_ID + "  in has_read.txt")
    # rest_url = 'http://www.cmbi.umcn.nl/xssp/'
    # # args = parser.parse_args()
    # print("begin load_train_txt_file()......")
    # PDB_train_Set = load_train_txt_file()
    # PDB_number = 0
    # string = ''
    # for PDB_ID in PDB_train_Set:
    #     PDB_path = windows_dir_pre + '/PDB_family_train/' + PDB_ID + '.pdb'
    #     output_path = windows_dir_pre + '/output/dssp/'
    #     print(PDB_ID + " " + str(PDB_number))
    #     PDB_number += 1
    #     string += PDB_ID + "\n"
    #     result = pdb_to_hssp(PDB_path, rest_url)
    #     write_str_to_file(result, output_path + PDB_ID + '.txt')
    #     write_str_to_file(string, output_path + 'has_read.txt')
    #     time.sleep(2)

