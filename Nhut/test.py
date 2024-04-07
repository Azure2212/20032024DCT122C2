import os
import pandas as pd

current_dir = os.path.abspath(globals().get("__file__","./"))
root_dir = os.path.abspath(f'{current_dir}/../../../../input/emotion/rafdb_basic.hdf5')

data = pd.read_hdf(root_dir)
print(data[(data['gender'] == 2 )& (data['type'] == 'train')])