# -*- coding: utf-8 -*-
import os
import re
from utils import constants as ct

from datetime import datetime

def _extract_date(f):
  date_string = re.search(f"^{ct.DATE_REGEXP}",f)[0]
  return datetime.strptime(date_string, ct.DATE_FORMAT)

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

def get_last_checkpoint_by_date(path):
  """
  Return file_name with the largest suffix number
  """
  list_of_files = os.listdir(path)

  file_dates = {_extract_date(f): f for f in list_of_files}
  if file_dates:
    key = sorted(file_dates.keys(), reverse=True)[0]
    return file_dates[key]
  else:
    return None
