import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the HDF5 file into a DataFrame
df = pd.read_hdf("/kaggle/working/sgu24project/rafdb_basic.hdf5")
df = df.loc[:, [ "age", "emotion","gender", "race"]]

# Tính ma trận tương quan
methods=['spearman', 'pearson', 'kendall']
for i in range(len(methods)):
    corr_matrix=df.corr(methods[i])
    # Vẽ heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='tab20b', fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Heatmap ({methods[i]})')
    plt.show()