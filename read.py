import os
current_dir = os.path.abspath(globals().get("__file__","./"))
root_dir = os.path.abspath(f'{current_dir}/../../../')
dataset_dir = root_dir + '/input/dataset-radfdb-basic/rafdb_basic.hdf5'

import pandas as pd
# In cả dataset
df = pd.read_hdf(dataset_dir)
print(df)
