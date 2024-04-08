import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
current_dir = os.path.abspath(globals().get("__file__", "./"))
root_dir = os.path.abspath(f'(current_dir)/../../../')
dataset_dir = root_dir + '/input/emotion/rafdb_basic.hdf5'

df = pd.read_hdf(dataset_dir)
TT_type = 'test'
TT_df = df[df['type'] == TT_type]


# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='age', y='gender',z='race', data=TT_df)

# Customize labels and title
plt.xlabel('Expression')
plt.ylabel('Confidence')
plt.title('Violin Plot of Confidence by Expression')

# Show the plot
plt.show()
