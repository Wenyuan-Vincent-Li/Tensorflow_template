import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from tensorflow.python.tools import inspect_checkpoint as chkp

def fn_inspect_checkpoint(ckpt_filepath, **kwargs):
  name = kwargs.get('tensor_name', '')
  if name == '':
    all_tensors = True
  else:
    all_tensors = False
  chkp.print_tensors_in_checkpoint_file(ckpt_filepath, name, all_tensors)