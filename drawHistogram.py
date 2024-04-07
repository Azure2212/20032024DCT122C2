import os
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.abspath(globals().get("__file__", "./"))
root_dir = os.path.abspath(f'{current_dir}/../../../')
dataset_dir = root_dir + '/input/dataset-radfdb-basic/rafdb_basic.hdf5'

# Đọc file HDF5 vào DataFrame
df = pd.read_hdf(dataset_dir)

# Thống kê và tạo histogram cho cột 'age'
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(df['age'].replace({0: '0-3', 1: '4-19', 2: '20-39', 3: '40-69', 4:'70+'}), bins=5, color='blue', edgecolor='black')
plt.title('Age Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Thống kê và tạo histogram cho cột 'race'
plt.subplot(2, 2, 2)
plt.hist(df['race'].replace({0: 'Caucasian', 1: 'African-American', 2: 'Asian'}), bins=3, color='red', edgecolor='black')
plt.title('Race Histogram')
plt.xlabel('Race')
plt.ylabel('Frequency')

# Thống kê và tạo histogram cho cột 'gender'
plt.subplot(2, 2, 3)
# Sử dụng labels thay vì giá trị số
plt.hist(df['gender'].replace({0: 'male', 1: 'female', 2: 'unsure'}), bins=3, color='green', edgecolor='black')
plt.title('Gender Histogram')
plt.xlabel('Gender')
plt.ylabel('Frequency')


# Thống kê và tạo histogram cho cột 'label' (emotion)
plt.subplot(2, 2, 4)
plt.hist(df['emotion'].replace({1: 'Surprise', 2: 'Fear', 3: 'Disgust', 4: 'Happiness', 5: 'Sadness', 6: 'Anger', 7: 'Neutral'}), bins=7, color='orange', edgecolor='black')
plt.title('Emotion Histogram')
plt.xlabel('Emotion')
plt.ylabel('Frequency')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
