#### hien bang thong ke
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

# Read the HDF5 file into a DataFrame
df = pd.read_hdf("/kaggle/working/sgu24project/rafdb_basic.hdf5")

# Select specific columns and print the corresponding rows
selected_columns = ["type","gender", "age", "emotion", "race"]
df_train = df.loc[:, selected_columns]

profile = ProfileReport(
    df_train, title="Pandas Profiling Report for rafdb_basic dataset"
)
# Save the report to an HTML file
#profile.to_file("/kaggle/working/MH_EDA/rafdb_basic_profile.html")
profile.to_widgets()