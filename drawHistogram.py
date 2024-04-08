import os
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.abspath(globals().get("__file__", "./"))
root_dir = os.path.abspath(f'{current_dir}/../../../')
dataset_dir = root_dir + '/input/dataset-radfdb-basic/rafdb_basic.hdf5'

# Đọc file HDF5 vào DataFrame
df = pd.read_hdf(dataset_dir)

# Tách dữ liệu thành dữ liệu huấn luyện và dữ liệu kiểm tra dựa trên cột 'type'
train_df = df[df['type'] == 'train']
test_df = df[df['type'] == 'test']


plt.figure(figsize=(24, 10))

# Histogram cho dữ liệu train
plt.subplot(2, 4, 1)
plt.hist(train_df['age'].replace({0: '0-3', 1: '4-19', 2: '20-39', 3: '40-69', 4:'70+'}), bins=5, color='blue', edgecolor='black')
plt.title('Train - Age Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(2, 4, 2)
plt.hist(train_df['race'].replace({0: 'Caucasian', 1: 'African-American', 2: 'Asian'}), bins=3, color='red', edgecolor='black')
plt.title('Train - Race Histogram')
plt.xlabel('Race')
plt.ylabel('Frequency')

plt.subplot(2, 4, 3)
plt.hist(train_df['gender'].replace({0: 'male', 1: 'female', 2: 'unsure'}), bins=3, color='green', edgecolor='black')
plt.title('Train - Gender Histogram')
plt.xlabel('Gender')
plt.ylabel('Frequency')

plt.subplot(2, 4, 4)
plt.hist(train_df['emotion'].replace({1: 'Surprise', 2: 'Fear', 3: 'Disgust', 4: 'Happiness', 5: 'Sadness', 6: 'Anger', 7: 'Neutral'}), bins=7, color='orange', edgecolor='black')
plt.title('Train - Emotion Histogram')
plt.xlabel('Emotion')
plt.ylabel('Frequency')

# Histogram cho dữ liệu test
plt.subplot(2, 4, 5)
plt.hist(test_df['age'].replace({0: '0-3', 1: '4-19', 2: '20-39', 3: '40-69', 4:'70+'}), bins=5, color='blue', edgecolor='black')
plt.title('Test - Age Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(2, 4, 6)
plt.hist(test_df['race'].replace({0: 'Caucasian', 1: 'African-American', 2: 'Asian'}), bins=3, color='red', edgecolor='black')
plt.title('Test - Race Histogram')
plt.xlabel('Race')
plt.ylabel('Frequency')

plt.subplot(2, 4, 7)
plt.hist(test_df['gender'].replace({0: 'male', 1: 'female', 2: 'unsure'}), bins=3, color='green', edgecolor='black')
plt.title('Test - Gender Histogram')
plt.xlabel('Gender')
plt.ylabel('Frequency')

plt.subplot(2, 4, 8)
plt.hist(test_df['emotion'].replace({1: 'Surprise', 2: 'Fear', 3: 'Disgust', 4: 'Happiness', 5: 'Sadness', 6: 'Anger', 7: 'Neutral'}), bins=7, color='orange', edgecolor='black')
plt.title('Test - Emotion Histogram')
plt.xlabel('Emotion')
plt.ylabel('Frequency')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
