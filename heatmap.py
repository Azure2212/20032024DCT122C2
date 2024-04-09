import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the HDF5 file into a DataFrame
df = pd.read_hdf("/kaggle/input/dataset-ty/rafdb_basic.hdf5")
df = df.loc[:, [ "age", "emotion","gender", "race"]]

# Tính toán phần trăm người trong từng nhóm tuổi và chủng tộc
total_count = df.groupby(['age', 'race']).size() #gr các nhóm người theo độ tuổi
percentages = total_count / total_count.groupby(level=0).sum() #tính xác xuất nhóm người theo độ tuổi
percentages = percentages.reset_index(name='percentage') # Đổi định dạng của Series thành DataFrame

# Tạo pivot table từ DataFrame đã tính toán
pivot_table = percentages.pivot_table(index='age', columns='race', values='percentage')

# Vẽ heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='tab20b', fmt='.2f', linewidths=0.5)
plt.title('Percentage of People by Age and Race')
plt.xlabel('Race')
plt.ylabel('Age')
plt.show()
