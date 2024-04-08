import os
import pandas as pd
import matplotlib.pyplot as plt
current_dir = os.path.abspath(globals().get("__file__","./"))
root_dir = os.path.abspath(f'(current_dir)/../../../')
dataset_dir = root_dir + '/input/emotion/rafdb_basic.hdf5'


df = pd.read_hdf(dataset_dir)
#print(df)
TT_type = 'test'
TT_df = df[df['type'] == TT_type]
#----------------------------------------------EMOTION----------------------------------------------#
emotion_counts = TT_df['emotion'].value_counts() #value counts sẽ thống kê từng giá trị ví dụ ở đây ta có 1: 1290, 2: 281, 3: 717, 4: 4772, ...

#add info
emotion_descriptions ={
    1:'Surprise', 2: 'Fear', 3: 'Disgust', 4:'Happiness', 5:'Sadness', 6: 'Anger', 7: 'Neutral'
}
# Create bar chart emotion
plt.figure(figsize=(8, 9)) #Điều chỉnh kích thước
bars = plt.bar(emotion_counts.index, emotion_counts.values,color='red') #emotion_counts.index: danh sách các nhãn(labels) của các cột
                                                                        #emotion_counts.value: danh sách các giá trị(values) của các cột
# Add labels and title

plt.xlabel('Emotion Categories')
plt.ylabel('Count')
plt.title('Distribution of Emotion Categories in ' + TT_type )

for  i, bar in enumerate(bars):
    height = bar.get_height()#lấy chiều cao cột hiện tại
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}\n{emotion_descriptions[emotion_counts.index[i]]}', ha='center', va='bottom')
# Show plot
plt.show()
#----------------------------------------------END OF EMOTION----------------------------------------------#

#----------------------------------------------GENDER----------------------------------------------#
gender_counts = TT_df['gender'].value_counts()
#add info
gender_descriptions = {0:'male',1:'female',2:'unsure'}
# Create bar chart gender

plt.figure(figsize=(8, 9))
bars = plt.bar(gender_counts.index, gender_counts.values, color='skyblue')

# Add labels and title
plt.xlabel('Gender Categories in ' + TT_type )
plt.ylabel('Count')
plt.title('Distribution of Gender in ' + TT_type )

for i,bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}\n{gender_descriptions[gender_counts.index[i]]}', ha='center', va='bottom')
# Show plot
plt.show()
#----------------------------------------------END OF GENDER----------------------------------------------#



#----------------------------------------------RACE----------------------------------------------#
race_counts = TT_df['race'].value_counts()
#add info
race_descriptions = {0.0:'Caucasian', 1.0:'African-American', 2.0:'Asian'}
# Create bar chart race
plt.figure(figsize=(8, 9))
bars = plt.bar(race_counts.index, race_counts.values,color='purple')


# Add labels and title
plt.xlabel('Race Categories in ' + TT_type)
plt.ylabel('Count')
plt.title('Distribution of Race in ' + TT_type )

for i,bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}\n{race_descriptions[race_counts.index[i]]}', ha='center', va='bottom')
# Show plot
plt.show()
#----------------------------------------------END OF RACE----------------------------------------------#





#----------------------------------------------AGE----------------------------------------------#
age_counts = TT_df['age'].value_counts()
#add info
age_description = {0:'0-3', 1:'4-19',2:'20-39',3:'40-69',4:'70+'}
# Create bar chart gender
plt.figure(figsize=(8, 9))
bars = plt.bar(age_counts.index, age_counts.values, color='pink')


# Add labels and title
plt.xlabel('Age Categories in ' + TT_type )
plt.ylabel('Count')
plt.title('Distribution of Age in ' + TT_type )

for i,bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}\n{age_description[age_counts.index[i]]}', ha='center', va='bottom')
# Show plot
plt.show()
#----------------------------------------------END OF AGE----------------------------------------------#