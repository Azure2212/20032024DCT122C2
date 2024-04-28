import os
import pandas as pd
import matplotlib.pyplot as plt

fer2013_path = '/kaggle/input/fer2013'
folders = ['train', 'test']
emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_mapping = {1: 'angry', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}


train_counts = []
test_counts = []


for folder in folders:
    for i, emotion_folder in enumerate(emotion_folders):
        emotion_folder_path = os.path.join(fer2013_path, folder, emotion_folder)
        file_count = len(os.listdir(emotion_folder_path))
        if folder == 'train':
            train_counts.append(file_count)
        elif folder == 'test':
            test_counts.append(file_count)


fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.flatten()

for i, emotion in enumerate(emotion_folders):
    #Scatter plot for Train
    axs[i].scatter(train_counts[i], test_counts[i], color='blue', marker='o', label='Train')

    #Scatter plot for Test
    axs[i].scatter(test_counts[i], train_counts[i], color='red', marker='o', label='Test')

    axs[i].set_title(f'Comparison of "{emotion}" Emotion', fontweight='bold')
    axs[i].set_xlabel('Train Dataset', fontweight='bold')
    axs[i].set_ylabel('Test Dataset', fontweight='bold')
    axs[i].legend()  # Add legend

    # Adding a diagonal line for reference
    max_count = max(train_counts[i], test_counts[i])
    axs[i].plot([0, max_count], [0, max_count], color='black', linestyle='--')

plt.delaxes(axs[7]) 
plt.delaxes(axs[8])  
plt.tight_layout()
plt.show()
