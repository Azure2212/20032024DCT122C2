fer2013_path = '/kaggle/input/fer2013'
folders = ['train','test']
emotion_folders = ['angry','disgust','fear','happy','neutral','sad','surprise']
emotion_mapping = {1: 'angry',2: 'disgust',3: 'fear',4: 'happy',5: 'neutral',6: 'sad',7: 'surprise'}

import os
import pandas as pd

#tạo format cho bảng
column_table = ["id","name","type","path","label"]
values_of_each_column = { "id":[], "name":[], "type":[], "path":[], "label":[] }
for folder in folders: #train or test
    folder_path = fer2013_path + '/' + folder
    for emotion_folder in emotion_folders:
        emotion_folder_path = folder_path + '/' + emotion_folder
        files = os.listdir(emotion_folder_path)
        for file in files:
            values_of_each_column["id"].append(file)
            values_of_each_column["name"].append(file)
            values_of_each_column["type"].append(folder)
            values_of_each_column["path"].append(emotion_folder_path)
            x = key = next((k for k, v in emotion_mapping.items() if v == emotion_folder), None)
            values_of_each_column["label"].append(x)

#print(values_of_each_column)
df = pd.DataFrame(values_of_each_column, columns=column_table)
print(df)

#dùng tabulate định dạng bảng
#from tabulate import tabulate
#table_str = tabulate(df, headers='keys', tablefmt='grid')
#print(table_str)

#PIE CHART
import matplotlib.pyplot as plt

#Train
train_label_counts = df[df['type'] == 'train']['label'].value_counts()

#Test
test_label_counts = df[df['type'] == 'test']['label'].value_counts()

# Mapping numerical labels to emotion names
emotion_names = [emotion_folders[label-1] for label in train_label_counts.index]

#Pie chart cho Train dataset
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.pie(train_label_counts, labels=emotion_names, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Emotions in Train Dataset')
plt.axis('equal')  

#Pie chart cho Test dataset
plt.subplot(1, 2, 2)
plt.pie(test_label_counts, labels=emotion_names, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Emotions in Test Dataset')
plt.axis('equal')  

plt.tight_layout()
plt.show()


#BAR CHART

train_label_counts = df[df['type'] == 'train']['label'].value_counts()
test_label_counts = df[df['type'] == 'test']['label'].value_counts()

# Mapping numerical labels to emotion names
emotion_names = [emotion_folders[label-1] for label in train_label_counts.index]

#TRAIN
plt.figure(figsize=(10, 6))
plt.bar(emotion_names, train_label_counts, color='skyblue')
plt.xlabel('Emotions', fontweight='bold')
plt.ylabel('Number of Images', fontweight='bold')
plt.title('Distribution of Emotions in Train Dataset', fontweight='bold')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()

#TEST
plt.figure(figsize=(10, 6))
plt.bar(emotion_names, test_label_counts, color='salmon')
plt.xlabel('Emotions', fontweight='bold')
plt.ylabel('Number of Images', fontweight='bold')
plt.title('Distribution of Emotions in Test Dataset', fontweight='bold')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()


#STACK BAR CHART

train_label_counts = df[df['type'] == 'train'].groupby('label').size()
train_emotion_names = [emotion_folders[label-1] for label in train_label_counts.index]

test_label_counts = df[df['type'] == 'test'].groupby('label').size()
test_emotion_names = [emotion_folders[label-1] for label in test_label_counts.index]

# Set the number of emotions
num_emotions = len(emotion_folders)

# Set the width of the bars
bar_width = 0.6

# Set the positions of the bars on the x-axis
r = range(num_emotions)

# Plotting the stacked bar chart for train dataset
plt.figure(figsize=(10, 6))
plt.bar(r, train_label_counts, color='skyblue', edgecolor='white', width=bar_width, label='Train')
plt.bar(r, test_label_counts, bottom=train_label_counts, color='salmon', edgecolor='white', width=bar_width, label='Test')
plt.xlabel('Emotions', fontweight='bold')
plt.ylabel('Number of Images', fontweight='bold')
plt.title('Distribution of Emotions in Train and Test Datasets', fontweight='bold')
plt.xticks(r, emotion_folders, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

#Area Chart

train_label_counts = df[df['type'] == 'train']['label'].value_counts()
test_label_counts = df[df['type'] == 'test']['label'].value_counts()

# Mapping numerical labels to emotion names
emotion_names = [emotion_folders[label-1] for label in train_label_counts.index]

# Plotting the area chart for both train and test datasets
plt.figure(figsize=(10, 6))

plt.fill_between(emotion_names, train_label_counts, color='skyblue', alpha=0.4, label='Train')
plt.plot(emotion_names, train_label_counts, color='skyblue', alpha=0.6)

plt.fill_between(emotion_names, test_label_counts, color='salmon', alpha=0.4, label='Test')
plt.plot(emotion_names, test_label_counts, color='salmon', alpha=0.6)

# Adding labels and title
plt.xlabel('Emotions', fontweight='bold')
plt.ylabel('Number of Images', fontweight='bold')
plt.title('Distribution of Emotions in Train and Test Datasets', fontweight='bold')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Adding legend
plt.legend()

plt.tight_layout()
plt.show()


#Scatter plot
import numpy as np
import matplotlib.pyplot as plt

# Grouping by 'label' and counting occurrences for train dataset
train_label_counts = df[df['type'] == 'train']['label'].value_counts()

# Grouping by 'label' and counting occurrences for test dataset
test_label_counts = df[df['type'] == 'test']['label'].value_counts()

# Mapping numerical labels to emotion names
emotion_names = [emotion_folders[label-1] for label in train_label_counts.index]

# Generate x positions for train and test data points
train_x = np.arange(len(emotion_names))
test_x = train_x + 0.3  # Shift test points to the right for better visualization

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(train_x, train_label_counts, color='skyblue', label='Train', marker='o')
plt.scatter(test_x, test_label_counts, color='salmon', label='Test', marker='x')

# Adding labels and title
plt.xlabel('Emotions', fontweight='bold')
plt.ylabel('Number of Images', fontweight='bold')
plt.title('Distribution of Emotions in Train and Test Datasets', fontweight='bold')

# Adjust x-axis ticks and labels
plt.xticks(train_x + 0.15, emotion_names, rotation=45, ha='right')

# Adding legend
plt.legend()

plt.tight_layout()
plt.show()