# -*- coding: utf-8 -*-
import os
import re

def _extract_number(f):
    s = re.findall("\d+$",f)
    return int(s[0])

def get_max_index_checkpoint(path):
    """
    Return int: suffix of checkpoint name
    """
    list_of_files = os.listdir(path)
    # list_of_files = ["checkpoint_1","checkpoint_10","checkpoint_2", "checkpoint_22"]

    n = max([_extract_number(f) for f in list_of_files]) if list_of_files else None
    if n is None:
        return 0

    return n

def get_last_checkpoint(path):
    """
    Return file_name with the largest suffix number
    """
    list_of_files = os.listdir(path)
    # list_of_files = ["checkpoint_1","checkpoint_10","checkpoint_2", "checkpoint_22"]

    file_numbers = {_extract_number(f):f for f in list_of_files}
    if file_numbers:
        n = max(file_numbers.keys())
        return file_numbers[n]
    else:
        return None
