import re
import os
import csv
import glob


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def read_csv_file(csv_file):
    """
    Open and read a csv file and returns a list containing
    patient IDs and corresponding labels.
    """
    patient_id = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data_list = list(reader)
    for i in range(1,len(data_list)):   
        row = data_list[i]
        name = row
        patient_id.append(name)
    return patient_id



def get_filepath(path, pattern):
    """
    Path: path to the data folder
    Pattern: string pattern in filenames of interest
    Returns the full path of each file containing
    pattern in the filename.
    """
    data_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if pattern in file:
                file_path = os.path.join(root, file)
                data_list.append(file_path)
    data_list.sort(key=natural_sort_key)
    return data_list

def match_img_label(img_path, csv_contents):
    '''
    match the data directory to the .csv contents

    Parameters
    ----------
    img_path : list
        each item in the list is a full path to a data.
    csv_contents : list
        name and label of each data subject.

    Returns
    -------
    img_with_label : list
        each item consists of full directory to the data and corresponding label.

    '''
    img_with_label = []
    for image_path in img_path:
        for csv_data in csv_contents:
            subject_name_img = os.path.split(image_path)[0].split('/')[-1]
            subject_name_csv = csv_data[0]
            subject_label = csv_data[1]
            if subject_name_img == subject_name_csv:
                temp = []
                temp.append(image_path)
                temp.append(subject_label)
                img_with_label.append(temp)
    return img_with_label
