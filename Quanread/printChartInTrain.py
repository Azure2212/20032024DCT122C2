import os
import pandas as pd
import matplotlib.pyplot as plt
current_dir = os.path.abspath(globals().get("__file__","./"))
root_dir = os.path.abspath(f'(current_dir)/../../../')
dataset_dir = root_dir + '/input/emotion/rafdb_basic.hdf5'


df = pd.read_hdf(dataset_dir)

train_df = df[df['type'] == 'train']
#----------------------------------------------EMOTION----------------------------------------------#
emotion_counts = train_df['emotion'].value_counts()

# Create bar chart emotion
bars = plt.bar(emotion_counts.index, emotion_counts.values,color='red')


# Add labels and title
plt.xlabel('Emotion Categories')
plt.ylabel('Count')
plt.title('Distribution of Emotion Categories in train')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
# Show plot
plt.show()
#----------------------------------------------END OF EMOTION----------------------------------------------#


#----------------------------------------------RACE----------------------------------------------#
race_counts = train_df['race'].value_counts()
# Create bar chart race
bars = plt.bar(race_counts.index, race_counts.values,color='purple')


# Add labels and title
plt.xlabel('Race Categories')
plt.ylabel('Count')
plt.title('Distribution of Race in train')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
# Show plot
plt.show()
#----------------------------------------------END OF RACE----------------------------------------------#


#----------------------------------------------GENDER----------------------------------------------#
gender_counts = train_df['gender'].value_counts()
# Create bar chart gender
bars = plt.bar(gender_counts.index, gender_counts.values, color='skyblue')


# Add labels and title
plt.xlabel('Gender Categories in train')
plt.ylabel('Count')
plt.title('Distribution of Gender in train')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
# Show plot
plt.show()
#----------------------------------------------END OF GENDER----------------------------------------------#
