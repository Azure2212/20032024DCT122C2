import os
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.abspath(globals().get("__file__", "./"))
root_dir = os.path.abspath(f'{current_dir}/../../../')
dataset_dir = root_dir + '/input/dataset-radfdb-basic/rafdb_basic.hdf5'

# Đọc file HDF5 vào DataFrame
df = pd.read_hdf(dataset_dir)

# Thống kê và tạo histogram cho cột 'age'
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.hist(df['age'], bins=10, color='blue', edgecolor='black')
plt.title('Age Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Thống kê và tạo histogram cho cột 'race'
plt.subplot(2, 2, 2)
plt.hist(df['race'], bins=5, color='red', edgecolor='black')
plt.title('Race Histogram')
plt.xlabel('Race')
plt.ylabel('Frequency')

# Thống kê và tạo histogram cho cột 'gender'
plt.subplot(2, 2, 3)
plt.hist(df['gender'], bins=2, color='green', edgecolor='black')
plt.title('Gender Histogram')
plt.xlabel('Gender')
plt.ylabel('Frequency')

# Thống kê và tạo histogram cho cột 'label' (emotion)
plt.subplot(2, 2, 4)
plt.hist(df['emotion'], bins=7, color='orange', edgecolor='black')
plt.title('Emotion Histogram')
plt.xlabel('Emotion')
plt.ylabel('Frequency')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
