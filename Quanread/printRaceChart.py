import os
import pandas as pd
import matplotlib.pyplot as plt
current_dir = os.path.abspath(globals().get("__file__","./"))
root_dir = os.path.abspath(f'(current_dir)/../../../')
dataset_dir = root_dir + '/input/emotion/rafdb_basic.hdf5'

df = pd.read_hdf(dataset_dir)

race_counts = df['race'].value_counts()

# Create bar chart
bars = plt.bar(race_counts.index, race_counts.values,color='purple')


# Add labels and title
plt.xlabel('Race Categories')
plt.ylabel('Count')
plt.title('Distribution of Race')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
# Show plot
plt.show()