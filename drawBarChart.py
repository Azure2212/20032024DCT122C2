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

# Thiết lập kích thước biểu đồ
plt.figure(figsize=(24, 10))

# Bar chart cho dữ liệu huấn luyện về 'age'
plt.subplot(2, 4, 1)
train_age_counts = train_df['age'].value_counts().sort_index()
plt.bar(train_age_counts.index, train_age_counts.values, color='blue', edgecolor='black')
plt.title('Train - Age Bar Chart')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xticks(train_age_counts.index, ['0-3', '4-19', '20-39', '40-69', '70+'])

# Bar chart cho dữ liệu huấn luyện về 'race'
plt.subplot(2, 4, 2)
train_race_counts = train_df['race'].value_counts().sort_index()
plt.bar(train_race_counts.index, train_race_counts.values, color='red', edgecolor='black')
plt.title('Train - Race Bar Chart')
plt.xlabel('Race')
plt.ylabel('Frequency')
plt.xticks(train_race_counts.index, ['Caucasian', 'African-American', 'Asian'])

# Bar chart cho dữ liệu huấn luyện về 'gender'
plt.subplot(2, 4, 3)
train_gender_counts = train_df['gender'].value_counts().sort_index()
plt.bar(train_gender_counts.index, train_gender_counts.values, color='green', edgecolor='black')
plt.title('Train - Gender Bar Chart')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(train_gender_counts.index, ['male', 'female', 'unsure'])

# Bar chart cho dữ liệu huấn luyện về 'emotion'
plt.subplot(2, 4, 4)
train_emotion_counts = train_df['emotion'].value_counts().sort_index()
plt.bar(train_emotion_counts.index, train_emotion_counts.values, color='orange', edgecolor='black')
plt.title('Train - Emotion Bar Chart')
plt.xlabel('Emotion')
plt.ylabel('Frequency')
plt.xticks(train_emotion_counts.index, ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'])

# Bar chart cho dữ liệu kiểm tra về 'age'
plt.subplot(2, 4, 5)
test_age_counts = test_df['age'].value_counts().sort_index()
plt.bar(test_age_counts.index, test_age_counts.values, color='blue', edgecolor='black')
plt.title('Test - Age Bar Chart')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xticks(test_age_counts.index, ['0-3', '4-19', '20-39', '40-69', '70+'])

# Bar chart cho dữ liệu kiểm tra về 'race'
plt.subplot(2, 4, 6)
test_race_counts = test_df['race'].value_counts().sort_index()
plt.bar(test_race_counts.index, test_race_counts.values, color='red', edgecolor='black')
plt.title('Test - Race Bar Chart')
plt.xlabel('Race')
plt.ylabel('Frequency')
plt.xticks(test_race_counts.index, ['Caucasian', 'African-American', 'Asian'])

# Bar chart cho dữ liệu kiểm tra về 'gender'
plt.subplot(2, 4, 7)
test_gender_counts = test_df['gender'].value_counts().sort_index()
plt.bar(test_gender_counts.index, test_gender_counts.values, color='green', edgecolor='black')
plt.title('Test - Gender Bar Chart')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(test_gender_counts.index, ['male', 'female', 'unsure'])

# Bar chart cho dữ liệu kiểm tra về 'emotion'
plt.subplot(2, 4, 8)
test_emotion_counts = test_df['emotion'].value_counts().sort_index()
plt.bar(test_emotion_counts.index, test_emotion_counts.values, color='orange', edgecolor='black')
plt.title('Test - Emotion Bar Chart')
plt.xlabel('Emotion')
plt.ylabel('Frequency')
plt.xticks(test_emotion_counts.index, ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'])

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
