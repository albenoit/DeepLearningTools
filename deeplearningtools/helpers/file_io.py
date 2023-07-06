# ========================================
# FileName: file_io.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit (plus some colleagues and interns such as Louis Klein on spring 2017)
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A set of tools to preprocess data and build up input data pipelines
# for DeepLearningTools.
# =========================================

import os
import glob
from subprocess import check_output

def extractFilenames(root_dir, file_extension="*.jpg", raiseOnEmpty=True):
    '''
    Utility function to extract filenames based on a root directory and file extension.

    Given a root directory and file extension, this function walks through the directory tree
    to create a list of files that match the specified extension.

    :param root_dir: The root folder from which files should be searched.
    :type root_dir: str
    :param file_extension: The extension of the files to search for. Defaults to ".jpg".
    :type file_extension: str
    :param raiseOnEmpty: A boolean flag indicating whether an exception should be raised if no file is found. Defaults to True.
    :type raiseOnEmpty: bool

    :return: A sorted list of filenames.
    :rtype: list[str]

    :raises ValueError: If no files are found and `raiseOnEmpty` is set to True.
    '''
    files  = []
    msg='extractFilenames: from working directory {wd}, looking for files {path} with extension {ext}'.format(wd=os.getcwd(),
                                                                                                                path=root_dir,
                                                                                                                ext=file_extension)
    #print(msg)
    for root, dirnames, filenames in os.walk(root_dir):
        file_proto=os.path.join(root, file_extension)
        print('-> Parsing folder : '+file_proto)
        newfiles = glob.glob(file_proto)
        if len(newfiles)>0:
            print('----> Found files:'+str(len(newfiles)))
        files.extend(newfiles)

    if len(files)==0 and raiseOnEmpty is True:
        raise ValueError('No files found at '+msg)
    else:
        print('Found files : '+str(len(files)))
    return sorted(files)

def count_lines(path, skip_header):
    """
    Count the number of lines in files that match a specified path.

    Given a file path pattern, this function counts the total number of lines across all files
    that match the specified path. It also provides an option to skip a certain number of lines
    from the beginning of each file.

    :param path: The file path pattern to match.
    :type path: str
    :param skip_header: The number of lines to skip from the beginning of each file.
    :type skip_header: int

    :return: The total number of lines across all files.
    :rtype: int
    """
    lines=0
    for file in glob.glob(path):
        print('file=', file)
        lines+=int(check_output(["wc", "-l", file]).split()[0])-int(skip_header)
    return lines 
